# Mithya/utils/face_extractor.py

"""
Face extraction utility using Google's MediaPipe.
Provides a reliable, easy-to-install, and high-performance face detector.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)

class FaceExtractor:
    """
    A wrapper class for MediaPipe's Face Detection model.
    """

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.5):
        """
        Initializes the MediaPipe Face Detector.

        Args:
            model_selection (int): 0 for short-range model (2 meters), 1 for full-range (5 meters).
            min_detection_confidence (float): Minimum confidence value for a detection to be considered successful.
        """
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Face Detection: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in a single BGR image.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  contains 'box', 'confidence', and 'landmarks' for a detected face.
                                  Returns an empty list if no faces are detected or an error occurs.
        """
        if image is None:
            logger.warning("Input image is None.")
            return []

        try:
            # MediaPipe expects RGB, so convert from BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(image_rgb)

            detected_faces = []
            if results.detections:
                img_h, img_w, _ = image.shape
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    bbox = int(bboxC.xmin * img_w), int(bboxC.ymin * img_h), \
                           int(bboxC.width * img_w), int(bboxC.height * img_h)
                    
                    detected_faces.append({
                        'box': bbox,
                        'confidence': detection.score[0],
                        'landmarks': detection.location_data.relative_keypoints
                    })
            return detected_faces
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []

    def crop_faces(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        padding_factor: float = 0.2
    ) -> List[np.ndarray]:
        """
        Detects and crops faces from an image.

        Args:
            image (np.ndarray): The input image in BGR format.
            target_size (Optional[Tuple[int, int]]): The size (width, height) to resize cropped faces.
                                                     If None, original crop size is kept.
            padding_factor (float): Factor to expand the bounding box for more context.

        Returns:
            List[np.ndarray]: A list of cropped face images.
        """
        detections = self.detect_faces(image)
        if not detections:
            return []

        cropped_faces = []
        img_h, img_w, _ = image.shape

        for detection in detections:
            x, y, w, h = detection['box']

            # Add padding
            pad_w = int(w * padding_factor)
            pad_h = int(h * padding_factor)
            x1 = max(0, x - pad_w // 2)
            y1 = max(0, y - pad_h // 2)
            x2 = min(img_w, x + w + pad_w // 2)
            y2 = min(img_h, y + h + pad_h // 2)

            face_crop = image[y1:y2, x1:x2]

            if face_crop.size == 0:
                logger.warning(f"Generated an empty crop for box {detection['box']}. Skipping.")
                continue

            if target_size:
                face_crop = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
            
            cropped_faces.append(face_crop)

        return cropped_faces