# Mithya/models/temporal_3dcnn.py

"""
3D Convolutional Neural Network (3D CNN) for temporal deepfake analysis.
Uses a 3D ResNet architecture to capture motion and time-based artifacts.
"""

import torch
import torch.nn as nn
import logging
from torchvision.models import video as video_models
from typing import Dict

# Import from our project
from config import R3D_MODEL_NAME, NUM_CLASSES

# Configure logging
logger = logging.getLogger(__name__)


class TemporalDetector(nn.Module):
    """
    Temporal deepfake detector using a pre-trained R3D-18 (3D ResNet) model.
    """
    def __init__(self, pretrained: bool = True):
        """
        Initializes the TemporalDetector model.

        Args:
            pretrained (bool): Whether to load pre-trained weights (Kinetics-400). Defaults to True.
        """
        super(TemporalDetector, self).__init__()
        self.model_name = R3D_MODEL_NAME

        try:
            if self.model_name == 'r3d_18':
                self.model = video_models.r3d_18(weights='KINETICS400_V1' if pretrained else None)
            else:
                # Placeholder for other 3D models if needed in the future
                raise NotImplementedError(f"Model {self.model_name} is not supported.")

            # Replace the final fully connected layer for our binary classification task
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

            logger.info(f"Initialized {self.model_name} model.")
            if pretrained:
                logger.info(f"Loaded pre-trained weights for {self.model_name}.")

        except Exception as e:
            logger.error(f"Failed to create video model '{self.model_name}': {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the 3D CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        try:
            # Input format for torchvision video models is (N, C, T, H, W)
            return self.model(x)
        except Exception as e:
            logger.error(f"Error during forward pass in TemporalDetector: {e}")
            return torch.zeros((x.size(0), NUM_CLASSES), device=x.device)

    def get_config(self) -> Dict:
        """Returns the model's configuration."""
        return {
            "model_name": self.model_name,
            "num_classes": NUM_CLASSES,
            "pretrained_dataset": "Kinetics-400"
        }