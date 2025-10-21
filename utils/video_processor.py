# Mithya/utils/video_processor.py

"""
Utility for processing video files.
Includes functions for extracting frames from videos using OpenCV.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_frames(
    video_path: str,
    num_frames: int,
    target_size: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """
    Extracts a specified number of frames, evenly spaced, from a video file.

    Args:
        video_path (str): The path to the video file.
        num_frames (int): The total number of frames to extract.
        target_size (Optional[Tuple[int, int]]): The target (width, height) to resize frames.
                                                  If None, original size is kept.

    Returns:
        List[np.ndarray]: A list of extracted frames as NumPy arrays (in BGR format).
                          Returns an empty list if the video cannot be opened or processed.
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            logger.warning(f"Video file has no frames: {video_path}")
            return []

        # Determine which frame indices to capture
        if total_frames <= num_frames:
            # If video has fewer frames than requested, take all of them
            indices = np.arange(total_frames).astype(int)
        else:
            # Evenly space the frames to capture
            indices = np.linspace(0, total_frames - 1, num=num_frames).astype(int)
        
        frame_count = 0
        captured_indices = set()
        
        for idx in sorted(indices):
            if idx in captured_indices:
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                if target_size:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                frames.append(frame)
                captured_indices.add(idx)
            else:
                logger.warning(f"Could not read frame at index {idx} from {video_path}")
        
        cap.release()
        
        if not frames:
            logger.warning(f"No frames were extracted from {video_path}")
            
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing video {video_path}: {e}")
        # Ensure capture is released in case of error
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return []

    return frames