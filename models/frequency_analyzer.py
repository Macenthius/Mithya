# Mithya/models/frequency_analyzer.py

"""
Frequency domain analysis utility for detecting GAN-based artifacts.
Deepfakes often exhibit unnatural patterns in the frequency spectrum.
"""

import cv2
import numpy as np
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

def analyze_frequency(
    face_image: np.ndarray, 
    high_pass_cutoff: float = 0.1
) -> Optional[float]:
    """
    Analyzes the frequency spectrum of a face image to detect high-frequency artifacts.

    This method calculates the ratio of energy in high-frequency components to
    the total energy in the frequency spectrum. A higher ratio can indicate
    the presence of artificial textures common in deepfakes.

    Args:
        face_image (np.ndarray): A cropped face image in BGR format.
        high_pass_cutoff (float): The fraction of the spectrum size to consider as
                                  the start of the high-frequency region (e.g., 0.1
                                  means the outer 90% of the frequency space).

    Returns:
        Optional[float]: A score between 0 and 1 representing the high-frequency
                         anomaly level. Returns None if an error occurs.
    """
    try:
        if face_image is None or face_image.size == 0:
            logger.warning("Input face image is empty.")
            return None

        # 1. Convert to grayscale for frequency analysis
        gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # 2. Apply 2D Fast Fourier Transform (FFT)
        f_transform = np.fft.fft2(gray_image)

        # 3. Shift the zero-frequency component to the center
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # 4. Calculate the magnitude spectrum (log scale for visualization/intuition)
        magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
        
        # 5. Create a high-pass filter mask
        rows, cols = gray_image.shape
        crow, ccol = rows // 2 , cols // 2
        
        radius = int(min(crow, ccol) * high_pass_cutoff)
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 0, -1) # Create a black circle in the center

        # 6. Calculate energy
        # Total energy is the sum of magnitudes squared
        total_energy = np.sum(magnitude_spectrum)
        if total_energy < 1e-6:
            return 0.0 # Avoid division by zero for black images

        # High-frequency energy is where the mask is 1
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        
        # 7. Calculate the ratio as the anomaly score
        anomaly_score = high_freq_energy / total_energy
        
        return anomaly_score

    except Exception as e:
        logger.error(f"An error occurred during frequency analysis: {e}")
        return None