import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import cv2  # This import is necessary for cv2.imwrite

# Add project root to Python path to import our modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.video_processor import extract_frames
from utils.face_extractor import FaceExtractor
from config import NUM_FRAMES_PER_VIDEO, FACE_CROP_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_videos(video_dir: Path, output_dir: Path, label: int):
    """
    Processes a directory of videos, extracts frames, detects faces,
    saves the crops, and returns metadata for the CSV.
    """
    face_extractor = FaceExtractor()
    metadata = []
    
    video_files = sorted(list(video_dir.glob("*.mp4")))
    if not video_files:
        logger.warning(f"No .mp4 files found in {video_dir}")
        return []

    for video_path in tqdm(video_files, desc=f"Processing label {label} in {video_dir.name}"):
        try:
            # 1. Extract frames from the video
            frames = extract_frames(str(video_path), num_frames=NUM_FRAMES_PER_VIDEO)
            if not frames:
                logger.warning(f"No frames extracted from {video_path}. Skipping.")
                continue

            # 2. Extract and save face crops from each frame
            frame_has_face = False
            for i, frame in enumerate(frames):
                # We crop faces with a standard size for our models
                faces = face_extractor.crop_faces(frame, target_size=(FACE_CROP_SIZE, FACE_CROP_SIZE))
                if faces:
                    frame_has_face = True
                
                for j, face_crop in enumerate(faces):
                    # Create a unique filename for each face
                    face_filename = f"{video_path.stem}_frame{i}_face{j}.jpg"
                    save_path = output_dir / face_filename
                    
                    # Save the face crop
                    cv2.imwrite(str(save_path), face_crop)
                    
                    # Store metadata for the CSV file, using relative path
                    metadata.append({
                        'path': str(save_path.relative_to(project_root)),
                        'label': label
                    })
            
            if not frame_has_face:
                logger.warning(f"No faces detected in any frame for video {video_path}")

        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")

    return metadata

def main(args):
    base_video_path = Path(args.base_dir)
    output_faces_path = project_root / "data" / "processed_faces"
    
    # Define paths for real and fake videos
    real_video_dirs = [base_video_path / "original_sequences/youtube/c23/videos"]
    fake_video_dirs = [
        base_video_path / "manipulated_sequences/Deepfakes/c23/videos",
        base_video_path / "manipulated_sequences/Face2Face/c23/videos",
        base_video_path / "manipulated_sequences/FaceSwap/c23/videos",
    ]
    
    all_metadata = []

    # --- Process REAL videos ---
    logger.info("Starting processing of REAL videos...")
    real_output_dir = output_faces_path / "real"
    real_output_dir.mkdir(exist_ok=True, parents=True)
    for v_dir in real_video_dirs:
        if v_dir.exists():
            all_metadata.extend(process_videos(v_dir, real_output_dir, 0))
        else:
            logger.warning(f"Directory not found, skipping: {v_dir}")


    # --- Process FAKE videos ---
    logger.info("Starting processing of FAKE videos...")
    fake_output_dir = output_faces_path / "fake"
    fake_output_dir.mkdir(exist_ok=True, parents=True)
    for v_dir in fake_video_dirs:
        if v_dir.exists():
            all_metadata.extend(process_videos(v_dir, fake_output_dir, 1))
        else:
            logger.warning(f"Directory not found, skipping: {v_dir}")

    # --- Create and save the final DataFrame ---
    if not all_metadata:
        logger.error("No data was processed. No faces were found or extracted. Exiting without creating CSV.")
        return
        
    df = pd.DataFrame(all_metadata)
    
    # --- Balance the dataset ---
    # Downsample the majority class to match the minority class for a balanced training set
    label_counts = df['label'].value_counts()
    min_class_count = label_counts.min()
    
    if min_class_count > 0:
        logger.info(f"Balancing dataset. Using {min_class_count} samples per class.")
        df_balanced = df.groupby('label').sample(n=min_class_count, random_state=42)
        final_df = df_balanced
    else:
        logger.warning("One class has zero samples. Not balancing the dataset.")
        final_df = df
    
    csv_path = project_root / "data" / "ff_dataset.csv"
    final_df.to_csv(csv_path, index=False)
    
    logger.info("--- Preprocessing Complete ---")
    logger.info(f"Total REAL faces found: {label_counts.get(0, 0)}")
    logger.info(f"Total FAKE faces found: {label_counts.get(1, 0)}")
    logger.info(f"Saved balanced dataset with {len(final_df)} total samples to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the FaceForensics++ dataset.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/ff_raw_videos",
        help="The path to the directory where you stored the downloaded data."
    )
    args = parser.parse_args()
    main(args)