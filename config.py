# Mithya/config.py

"""
Central configuration file for the Mithya Deepfake Detection System.
Contains paths, model parameters, training settings, and other constants.
"""

import torch
from pathlib import Path

# --- Core Paths ---
# Use pathlib for OS-agnostic path handling
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = PROJECT_ROOT / "weights"  # For storing trained model weights
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for path in [DATA_DIR, MODELS_DIR, WEIGHTS_DIR, LOGS_DIR, RESULTS_DIR]:
    path.mkdir(exist_ok=True)


# --- Hardware Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Data & Preprocessing ---
VIDEO_FPS = 30  # FPS to sample videos
NUM_FRAMES_PER_VIDEO = 32  # Number of frames to extract from each video for analysis
FACE_CROP_SIZE = 224
TEMPORAL_SEQUENCE_LENGTH = 16 # For 3D CNN
TEMPORAL_INPUT_SIZE = 112


# --- Model Configurations ---
# Vision Transformer (ViT)
VIT_MODEL_NAME = 'vit_base_patch16_224'
VIT_IMAGE_SIZE = 224

# EfficientNetV2
EFFICIENTNET_MODEL_NAME = 'tf_efficientnetv2_b0'
EFFICIENTNET_IMAGE_SIZE = 224

# 3D CNN (ResNet3D)
R3D_MODEL_NAME = 'r3d_18'
R3D_IMAGE_SIZE = TEMPORAL_INPUT_SIZE

# Common Model Parameters
NUM_CLASSES = 1  # Binary classification (REAL vs. FAKE)
DROPOUT_RATE = 0.2


# --- Training Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LOSS_ALPHA = 0.25  # Alpha for Focal Loss
LOSS_GAMMA = 2.0  # Gamma for Focal Loss


# --- Ensemble Weights ---
# MODIFIED: Weights re-balanced for ViT and EfficientNet only.
ENSEMBLE_WEIGHTS = {
    'vit': 0.60,
    'efficientnet': 0.40,
    'temporal': 0.0,
    'frequency': 0.0,
    'face_consistency': 0.0
}
assert sum(ENSEMBLE_WEIGHTS.values()) == 1.0, "Ensemble weights must sum to 1.0"


# --- Inference Settings ---
DETECTION_THRESHOLD = 0.5  # Threshold for classifying as FAKE
FRAME_SKIP_RATE = 2  # Process every Nth frame for faster real-time inference


# --- API & App Settings ---
API_HOST = "0.0.0.0"
API_PORT = 8000
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}