# Mithya/models/ensemble.py

"""
Ensemble model that combines predictions from multiple deepfake detectors.
This class orchestrates the entire inference pipeline for a given video.
"""

import torch
import numpy as np
import logging
import cv2
from typing import Dict, List, Any, Tuple, Optional
from torchvision import transforms

# Import from our project
from config import (
    DEVICE, ENSEMBLE_WEIGHTS, DETECTION_THRESHOLD, NUM_FRAMES_PER_VIDEO,
    FACE_CROP_SIZE, TEMPORAL_SEQUENCE_LENGTH, TEMPORAL_INPUT_SIZE
)
from utils.video_processor import extract_frames
from utils.face_extractor import FaceExtractor
from models.vit_detector import ViTDetector
from models.efficient_detector import EfficientNetV2Detector
from models.temporal_3dcnn import TemporalDetector
from models.frequency_analyzer import analyze_frequency

# Configure logging
logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Orchestrates the deepfake detection process using an ensemble of models.
    """
    def __init__(self, device: torch.device = DEVICE):
        """
        Initializes all models and necessary utilities.

        Args:
            device (torch.device): The device (CPU or GPU) to run the models on.
        """
        self.device = device
        
        # Initialize models
        try:
            self.vit_detector = ViTDetector(pretrained=True).to(self.device)
            self.efficientnet_detector = EfficientNetV2Detector(pretrained=True).to(self.device)
            self.face_extractor = FaceExtractor()
            
            # MODIFIED: Make temporal detector optional
            try:
                self.temporal_detector = TemporalDetector(pretrained=True).to(self.device)
            except Exception as e:
                logger.warning(f"Could not initialize TemporalDetector: {e}. Temporal analysis will be disabled.")
                self.temporal_detector = None

        except Exception as e:
            logger.error(f"Failed to initialize one or more models: {e}")
            raise
            
        # Define image transformations
        self.spatial_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.temporal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

    def load_weights(
        self, 
        vit_path: str, 
        effnet_path: str, 
        temporal_path: Optional[str] = None
    ) -> None:
        """
        Loads trained weights for the neural network models.

        Args:
            vit_path (str): Path to the ViTDetector's .pth file.
            effnet_path (str): Path to the EfficientNetV2Detector's .pth file.
            temporal_path (Optional[str]): Path to the TemporalDetector's .pth file.
        """
        # MODIFIED: Made temporal_path optional
        models_and_paths = {
            "ViT": (self.vit_detector, vit_path),
            "EfficientNetV2": (self.efficientnet_detector, effnet_path),
        }
        
        if temporal_path and self.temporal_detector:
            models_and_paths["Temporal"] = (self.temporal_detector, temporal_path)

        for name, (model, path) in models_and_paths.items():
            try:
                model.load_state_dict(torch.load(path, map_location=self.device))
                logger.info(f"Successfully loaded weights for {name} from {path}")
            except FileNotFoundError:
                logger.error(f"Weight file not found for {name} at {path}. Model will use pre-trained weights.")
            except Exception as e:
                logger.error(f"Error loading weights for {name}: {e}")
        
        # Set all models to evaluation mode
        self.vit_detector.eval()
        self.efficientnet_detector.eval()
        if self.temporal_detector:
            self.temporal_detector.eval()

    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Performs end-to-end deepfake detection on a single video file.

        Args:
            video_path (str): The path to the video file.

        Returns:
            Dict[str, Any]: A dictionary containing the final score, prediction,
                            and a breakdown of individual model scores.
        """
        frames = extract_frames(video_path, num_frames=NUM_FRAMES_PER_VIDEO)
        if not frames:
            return self._format_error("Video could not be processed or no frames were extracted.")

        all_face_crops = []
        for frame in frames:
            faces = self.face_extractor.crop_faces(frame, target_size=(FACE_CROP_SIZE, FACE_CROP_SIZE))
            if faces:
                all_face_crops.extend(faces)

        if not all_face_crops:
            return self._format_error("No faces were detected in the video.")

        scores = {
            'vit': 0.0, 'efficientnet': 0.0, 'temporal': 0.0,
            'frequency': 0.0, 'face_consistency': 0.0
        }

        rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in all_face_crops]

        spatial_batch = torch.stack([self.spatial_transform(crop) for crop in rgb_crops]).to(self.device)
        vit_logits = self.vit_detector(spatial_batch)
        effnet_logits = self.efficientnet_detector(spatial_batch)
        vit_preds = torch.sigmoid(vit_logits).cpu().numpy().flatten().tolist()
        effnet_preds = torch.sigmoid(effnet_logits).cpu().numpy().flatten().tolist()
        
        if vit_preds: scores['vit'] = np.mean(vit_preds)
        if effnet_preds: scores['efficientnet'] = np.mean(effnet_preds)

        # MODIFIED: Check if temporal detector exists before running
        if self.temporal_detector:
            temporal_sequence = self._prepare_temporal_sequence(frames)
            if temporal_sequence is not None:
                temporal_batch = temporal_sequence.to(self.device)
                temporal_logits = self.temporal_detector(temporal_batch)
                temporal_pred = torch.sigmoid(temporal_logits).cpu().numpy().flatten().tolist()
                if temporal_pred: scores['temporal'] = np.mean(temporal_pred)

        final_score = self._calculate_ensemble_score(scores)
        
        return {
            "status": "success",
            "video_path": video_path,
            "ensemble_score": final_score,
            "final_prediction": "FAKE" if final_score >= DETECTION_THRESHOLD else "REAL",
            "detection_threshold": DETECTION_THRESHOLD,
            "individual_scores": scores
        }

    def _prepare_temporal_sequence(self, frames: List[np.ndarray]) -> torch.Tensor | None:
        """Prepares a tensor for the 3D CNN."""
        if len(frames) < TEMPORAL_SEQUENCE_LENGTH:
            return None
            
        sequence_frames = []
        for i in range(len(frames) - TEMPORAL_SEQUENCE_LENGTH + 1):
            temp_seq = []
            for j in range(TEMPORAL_SEQUENCE_LENGTH):
                frame = frames[i+j]
                faces = self.face_extractor.crop_faces(frame, target_size=(TEMPORAL_INPUT_SIZE, TEMPORAL_INPUT_SIZE))
                if not faces:
                    break 
                temp_seq.append(cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB))
            
            if len(temp_seq) == TEMPORAL_SEQUENCE_LENGTH:
                sequence_frames = temp_seq
                break 

        if not sequence_frames:
            return None

        transformed_frames = [self.temporal_transform(frame) for frame in sequence_frames]
        sequence_tensor = torch.stack(transformed_frames).permute(1, 0, 2, 3)
        return sequence_tensor.unsqueeze(0) 

    def _calculate_ensemble_score(self, scores: Dict[str, float]) -> float:
        """Calculates the weighted average score from individual model outputs."""
        final_score = 0.0
        for model_name, score in scores.items():
            final_score += score * ENSEMBLE_WEIGHTS.get(model_name, 0.0)
        return final_score
    
    def _format_error(self, message: str) -> Dict[str, Any]:
        """Creates a standardized error response dictionary."""
        logger.warning(message)
        return {
            "status": "error",
            "message": message,
            "ensemble_score": -1.0,
            "final_prediction": "ERROR"
        }