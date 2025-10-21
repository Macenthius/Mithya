# Mithya/models/vit_detector.py

"""
Vision Transformer (ViT) model for deepfake detection.
This serves as the primary image-based detector in the ensemble.
"""

import timm
import torch
import torch.nn as nn
import logging
from typing import Dict

# Import from our project
from config import VIT_MODEL_NAME, NUM_CLASSES, DROPOUT_RATE

# Configure logging
logger = logging.getLogger(__name__)

class ViTDetector(nn.Module):
    """
    Vision Transformer model based on the timm library.
    Pre-trained on ImageNet-21k for powerful feature extraction.
    """
    def __init__(self, pretrained: bool = True):
        """
        Initializes the ViTDetector model.

        Args:
            pretrained (bool): Whether to load pre-trained weights. Defaults to True.
        """
        super(ViTDetector, self).__init__()
        self.model_name = VIT_MODEL_NAME
        
        try:
            # Create the ViT model from timm
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=NUM_CLASSES,
                drop_rate=DROPOUT_RATE
            )
            
            # For binary classification, the output is a single logit.
            # timm handles the num_classes change automatically.
            # We use BCEWithLogitsLoss during training, so no sigmoid here.
            
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
            logger.error(f"Error during forward pass in ViTDetector: {e}")
            # Return a tensor of zeros with the expected shape to prevent crashing downstream
            return torch.zeros((x.size(0), NUM_CLASSES), device=x.device)

    def get_config(self) -> Dict:
        """Returns the model's configuration."""
        return {
            "model_name": self.model_name,
            "num_classes": NUM_CLASSES,
            "dropout_rate": DROPOUT_RATE
        }