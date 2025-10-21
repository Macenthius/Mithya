# Mithya/data/augmentations.py

"""
Data augmentation pipelines using the Albumentations library.
Defines separate pipelines for training and validation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import VIT_IMAGE_SIZE  # Use one of the standard sizes as default

def get_train_augs(image_size: int = VIT_IMAGE_SIZE) -> A.Compose:
    """
    Creates a composition of augmentations for the training dataset.

    These augmentations are designed to simulate a wide range of real-world
    scenarios, including different lighting, compression, and camera artifacts.

    Args:
        image_size (int): The target size (height and width) for the images.

    Returns:
        A.Compose: An Albumentations composition object.
    """
    return A.Compose([
        # --- Geometric Distortions ---
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=15,
            p=0.5
        ),

        # --- Color and Brightness ---
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.RGBShift(p=0.3),

        # --- Blur and Noise ---
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.4),
        A.ISONoise(p=0.3),
        A.GaussNoise(p=0.3),

        # --- Compression Artifacts (Crucial for deepfake detection) ---
        A.OneOf([
            A.ImageCompression(quality_lower=70, quality_upper=90, p=0.8),
            A.JpegCompression(quality_lower=70, quality_upper=90, p=0.8),
        ], p=0.5),
        
        # --- Sizing and Final Conversion ---
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_augs(image_size: int = VIT_IMAGE_SIZE) -> A.Compose:
    """
    Creates a composition of augmentations for the validation/testing dataset.

    This pipeline only performs the necessary resizing and normalization
    to ensure consistency without introducing random variations.

    Args:
        image_size (int): The target size (height and width) for the images.

    Returns:
        A.Compose: An Albumentations composition object.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])