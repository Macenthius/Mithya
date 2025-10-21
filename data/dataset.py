# Mithya/data/dataset.py

"""
Custom PyTorch Dataset class for loading deepfake data.
"""

import cv2
import torch
import pandas as pd
import logging
from torch.utils.data import Dataset
from typing import Dict, Any, Callable

# Configure logging
logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    A custom dataset to load, preprocess, and augment image data for
    the deepfake detection task.
    """
    def __init__(self, df: pd.DataFrame, augmentations: Callable):
        """
        Initializes the dataset object.

        Args:
            df (pd.DataFrame): A DataFrame with at least two columns:
                               'path' (path to the image) and
                               'label' (0 for REAL, 1 for FAKE).
            augmentations (Callable): An albumentations composition to apply to the images.
        """
        if 'path' not in df.columns or 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'path' and 'label' columns.")
        
        self.df = df
        self.augmentations = augmentations
        
        # For quick access
        self.image_paths = self.df['path'].values
        self.labels = self.df['label'].values

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample.

        Returns:
            Dict[str, Any]: A dictionary containing the processed image tensor
                            and its corresponding label tensor.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            # Load image using OpenCV (loads as BGR)
            image = cv2.imread(image_path)
            if image is None:
                raise IOError(f"Could not read image at path: {image_path}")
            
            # Convert from BGR to RGB as albumentations and PyTorch expect RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentations
            augmented = self.augmentations(image=image)
            image_tensor = augmented['image']
            
            return {
                "image": image_tensor,
                "label": torch.tensor(label, dtype=torch.float32)
            }

        except Exception as e:
            logger.error(f"Error loading or processing image {image_path}: {e}")
            # Return a dummy sample or skip, here we re-raise to be caught by dataloader
            # Or alternatively, return a placeholder:
            # return {"image": torch.zeros(3, 224, 224), "label": torch.tensor(0.0, dtype=torch.float32)}
            # For now, let's just log and see if dataloader handles it
            # It's better to clean data beforehand, but this is a safeguard.
            # Returning the 0th item as a fallback
            return self.__getitem__(0)