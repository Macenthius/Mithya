# Mithya/train.py
"""
Optimized training script for Mithya Deepfake Detection.
Adds AMP, faster DataLoader settings, async checkpointing, and a few production niceties.
"""

import argparse
import logging
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import os

# Project imports
from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, WEIGHTS_DIR,
    EARLY_STOPPING_PATIENCE, VIT_IMAGE_SIZE, EFFICIENTNET_IMAGE_SIZE
)
from data.dataset import DeepfakeDataset
from data.augmentations import get_train_augs, get_valid_augs
from models.vit_detector import ViTDetector
from models.efficient_detector import EfficientNetV2Detector

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS = {
    "vit": (ViTDetector, VIT_IMAGE_SIZE),
    "efficientnet": (EfficientNetV2Detector, EFFICIENTNET_IMAGE_SIZE),
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def async_save(state_dict: Dict[str, Any], path: Path):
    """Save a checkpoint in a background thread to avoid blocking."""
    def _save():
        try:
            torch.save(state_dict, str(path))
            logger.info(f"Async saved checkpoint: {path}")
        except Exception as e:
            logger.exception(f"Failed to async-save checkpoint to {path}: {e}")

    threading.Thread(target=_save, daemon=True).start()


def get_dataloaders(train_df, val_df, image_size, batch_size, num_workers, is_cuda):
    train_augs = get_train_augs(image_size)
    valid_augs = get_valid_augs(image_size)

    train_dataset = DeepfakeDataset(train_df, train_augs)
    val_dataset = DeepfakeDataset(val_df, valid_augs)

# heuristics for number of workers
    if num_workers is None:
        if os.name == 'nt': # Check if we're on Windows
            logger.info("Windows detected. Setting num_workers=0 to avoid multiprocessing errors.")
            num_workers = 0
        else:
            # We can be a bit more aggressive for modern systems
            cpu_count = os.cpu_count() or 4
            num_workers = min(cpu_count, 8)

    if num_workers == 0:
        # persistent_workers requires num_workers > 0
        kwargs = {'num_workers': 0, 'pin_memory': is_cuda}
    else:
        kwargs = {
            'num_workers': num_workers,
            'pin_memory': is_cuda,
            'persistent_workers': True,
            'prefetch_factor': 2
        }

    # Use persistent_workers when num_workers > 0 (speedup for repeated epochs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2, # Often can use larger batch for validation
        shuffle=False,
        **kwargs
    )
    return train_loader, val_loader


def train_one_epoch(loader, model, optimizer, scheduler, loss_fn, device,
                    scaler=None, accumulation_steps=1, use_tqdm=True):
    model.train()
    running_loss = 0.0
    it = enumerate(loader)
    if use_tqdm:
        it = tqdm(it, total=len(loader), desc="Training", leave=False)

    optimizer.zero_grad()
    for i, batch in it:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            outputs = model(images)
            loss = loss_fn(outputs, labels) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # gradient accumulation step
        if ((i + 1) % accumulation_steps) == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Note: for OneCycleLR, step per optimizer.step() is correct
            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * accumulation_steps)  # un-normalize for reporting
        if use_tqdm:
            lr_now = optimizer.param_groups[0]['lr']
            it.set_postfix(loss=running_loss / (i + 1), lr=lr_now)

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate_one_epoch(loader, model, loss_fn, device, use_tqdm=True):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    it = loader
    if use_tqdm:
        it = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in it:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).unsqueeze(1)

            # AMP is not strictly needed for inference but doesn't hurt
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Binary threshold 0.5 for accuracy/f1; AUC uses continuous preds
    try:
        all_preds_binary = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, all_preds_binary)
        f1 = f1_score(all_labels, all_preds_binary)
        auc = roc_auc_score(all_labels, all_preds)
    except Exception as e:
        logger.warning(f"Metric computation failed (likely one class in batch): {e}")
        accuracy = f1 = auc = float('nan')

    val_loss = running_loss / len(loader)
    metrics = {"accuracy": float(accuracy), "f1_score": float(f1), "auc": float(auc)}
    return val_loss, metrics


def main(args):
    # Resolve device
    device = torch.device(DEVICE)
    is_cuda = (device.type == 'cuda')
    
    if is_cuda:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU for training.")

    # cuDNN tuning
    if is_cuda:
        torch.backends.cudnn.benchmark = True

    if args.model not in MODELS:
        raise ValueError(f"Model '{args.model}' not supported. Choose from {list(MODELS.keys())}")

    ModelClass, image_size = MODELS[args.model]
    WEIGHTS_DIR_PATH = Path(WEIGHTS_DIR)
    WEIGHTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    output_path = WEIGHTS_DIR_PATH / f"{args.model}_best.pth"
    metrics_path = WEIGHTS_DIR_PATH / f"{args.model}_metrics.json"

    # Data
    logger.info("Loading dataset CSV...")
    df = pd.read_csv(args.data_csv)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_loader, val_loader = get_dataloaders(
        train_df, val_df,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_cuda=is_cuda
    )

    logger.info(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples.")

    # Model, optimizer, loss, scheduler
    logger.info(f"Initializing {args.model} model (pretrained={args.pretrained})...")
    model = ModelClass(pretrained=args.pretrained).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = FocalLoss().to(device)

    # Calculate steps_per_epoch for accumulation
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)

    # AMP scaler
    use_amp = args.use_amp and is_cuda
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP).")

    # Optional gradient accumulation
    accumulation_steps = max(1, args.accumulation_steps)
    if accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {accumulation_steps} steps.")

    best_val_auc = 0.0
    epochs_no_improve = 0
    history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")

        train_loss = train_one_epoch(
            train_loader, model, optimizer, scheduler, loss_fn,
            device, scaler=scaler, accumulation_steps=accumulation_steps, use_tqdm=args.verbose
        )

        val_loss, metrics = validate_one_epoch(val_loader, model, loss_fn, device, use_tqdm=args.verbose)

        logger.info(f"Epoch {epoch + 1} summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics -> Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc']:.4f}")

        epoch_time = time.time() - epoch_start
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "metrics": metrics,
            "epoch_time_sec": epoch_time
        })

        # Early stopping and async checkpoint
        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            logger.info(f"Validation AUC improved to {best_val_auc:.4f}. Saving checkpoint to {output_path} (async).")
            async_save(model.state_dict(), output_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement. Best AUC: {best_val_auc:.4f}. Streak: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        # Save metrics JSON each epoch (atomic replace)
        try:
            tmp_metrics = metrics_path.with_suffix('.tmp')
            tmp_metrics.write_text(json.dumps({"history": history, "best_val_auc": best_val_auc}, indent=2))
            tmp_metrics.replace(metrics_path)
        except Exception as e:
            logger.exception(f"Failed to write metrics JSON: {e}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time / 60:.2f} minutes. Best AUC: {best_val_auc:.4f}")
    # final save of metrics
    try:
        metrics_path.write_text(json.dumps({"history": history, "best_val_auc": best_val_auc}, indent=2))
    except Exception as e:
        logger.exception(f"Failed to write final metrics JSON: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deepfake detection model (optimized).")

    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys(), help="Model to train.")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to CSV with image paths and labels.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Per-device batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Max LR for OneCycleLR.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers (auto by default).")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps (1 = no accumulation).")
    parser.add_argument("--use_amp", action='store_true', help="Enable mixed precision (recommended when using GPU).")
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained', help='Train model from scratch (no pretrained weights).')
    parser.add_argument("--verbose", action='store_true', help="Show tqdm progress bars.")

    args = parser.parse_args()
    main(args)