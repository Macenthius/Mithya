# Mithya/models/efficient_detector.py

"""
EfficientNetV2 model for deepfake detection.
This serves as the secondary, faster image-based detector.
"""

import timm
import torch
import torch.nn as nn
import logging
from typing import Dict

# Import from our project
from config import EFFICIENTNET_MODEL_NAME, NUM_CLASSES, DROPOUT_RATE

# Configure logging
logger = logging.getLogger(__name__)

class EfficientNetV2Detector(nn.Module):
    """
    EfficientNetV2-B0 model based on the timm library.
    Known for its high efficiency (speed and accuracy).
    """
    def __init__(self, pretrained: bool = True):
        """
        Initializes the EfficientNetV2Detector model.

        Args:
            pretrained (bool): Whether to load pre-trained weights. Defaults to True.
        """
        super(EfficientNetV2Detector, self).__init__()
        self.model_name = EFFICIENTNET_MODEL_NAME

        try:
            # Create the EfficientNetV2 model from timm
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=NUM_CLASSES,
                drop_rate=DROPOUT_RATE
            )
            
            logger.info(f"Initialized {self.model_name} model.")
            if pretrained:
                logger.info(f"Loaded pre-trained weights for {self.model_name}.")

        except Exception as e:
            logger.error(f"Failed to create timm model '{self.model_name}': {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        try:
            return self.model(x)
        except Exception as e:
            logger.error(f"Error during forward pass in EfficientNetV2Detector: {e}")
            return torch.zeros((x.size(0), NUM_CLASSES), device=x.device)
            
    def get_config(self) -> Dict:
        """Returns the model's configuration."""
        return {
            "model_name": self.model_name,
            "num_classes": NUM_CLASSES,
            "dropout_rate": DROPOUT_RATE
        }