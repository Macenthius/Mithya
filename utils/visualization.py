# Mithya/utils/visualization.py

"""
Utility functions for plotting training results and other visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")

def plot_training_curves(history: List[Dict[str, Any]], model_name: str, save_path: Path):
    """
    Plots and saves the training/validation loss and AUC/Accuracy curves.

    Args:
        history (List[Dict[str, Any]]): The history list loaded from metrics.
        model_name (str): The name of the model (e.g., "ViT") for the title.
        save_path (Path): The full path to save the output .png file.
    """
    if not history:
        logger.warning("History is empty. Cannot plot training curves.")
        return

    try:
        # Extract data from the history
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        
        # Extract metrics, handling potential NaNs from the first epoch if validation fails
        val_auc = [h['metrics'].get('auc', float('nan')) for h in history]
        val_acc = [h['metrics'].get('accuracy', float('nan')) for h in history]

        # Create a 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        fig.suptitle(f"'{model_name.upper()}' Model Training History", fontsize=16, weight='bold')

        # --- Plot 1: Loss ---
        ax1.plot(epochs, train_loss, 'o-', label='Training Loss', color='b')
        ax1.plot(epochs, val_loss, 's-', label='Validation Loss', color='r')
        ax1.set_ylabel("Loss (FocalLoss)")
        ax1.set_title("Training vs. Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # --- Plot 2: Metrics (AUC and Accuracy on twin axes) ---
        color_auc = 'g'
        ax2.plot(epochs, val_auc, 'o-', label='Validation AUC', color=color_auc)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Validation AUC", color=color_auc)
        ax2.tick_params(axis='y', labelcolor=color_auc)
        ax2.legend(loc='upper left')
        
        # Create a second y-axis for accuracy
        ax2b = ax2.twinx()
        color_acc = 'm'
        ax2b.plot(epochs, val_acc, 's-', label='Validation Accuracy', color=color_acc, alpha=0.6)
        ax2b.set_ylabel("Validation Accuracy", color=color_acc)
        ax2b.tick_params(axis='y', labelcolor=color_acc)
        ax2b.legend(loc='upper right')
        
        ax2.set_title("Validation Metrics (AUC & Accuracy)")
        ax2.grid(True)
        
        # Final layout adjustments
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        
        # Save the figure
        plt.savefig(save_path)
        logger.info(f"Successfully saved training plot to {save_path}")
        plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to plot training curves: {e}", exc_info=True)