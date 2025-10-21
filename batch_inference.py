# Mithya/batch_inference.py

"""
Batch inference script for processing a directory of media files.
"""
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm

# Import from our project
from config import (
    DEVICE, WEIGHTS_DIR, ALLOWED_VIDEO_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS, RESULTS_DIR
)
from models.ensemble import EnsembleModel

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    """
    Main function to run the batch inference process.
    """
    # --- Input Validation ---
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Model Loading ---
    logger.info("Initializing the EnsembleModel...")
    try:
        ensemble = EnsembleModel(device=DEVICE)
        
        vit_w = args.vit_weights or WEIGHTS_DIR / "vit_best.pth"
        eff_w = args.effnet_weights or WEIGHTS_DIR / "efficientnet_best.pth"
        temp_w = args.temporal_weights or WEIGHTS_DIR / "temporal_best.pth"
        
        logger.info("Loading model weights...")
        ensemble.load_weights(str(vit_w), str(eff_w), str(temp_w))
        logger.info("Model ready for inference.")
    except Exception as e:
        logger.error(f"Failed to initialize or load the model: {e}")
        return

    # --- Batch Processing ---
    supported_extensions = ALLOWED_VIDEO_EXTENSIONS.union(ALLOWED_IMAGE_EXTENSIONS)
    files_to_process = [p for p in input_dir.glob("**/*") if p.suffix.lower() in supported_extensions]
    
    if not files_to_process:
        logger.warning(f"No supported media files found in {input_dir}")
        return

    logger.info(f"Found {len(files_to_process)} files to process. Starting batch inference...")
    
    for file_path in tqdm(files_to_process, desc="Batch Processing"):
        try:
            results = ensemble.predict_video(str(file_path)) # Using predict_video for all as a robust default
            
            if results and results.get("status") == "success":
                output_filename = output_dir / f"{file_path.stem}_results.json"
                with open(output_filename, 'w') as f:
                    json.dump(results, f, indent=4)
            else:
                logger.warning(f"Prediction failed for {file_path}: {results.get('message', 'Unknown error')}")

        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {file_path}: {e}")

    logger.info(f"Batch inference complete. Results are saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch deepfake detection on a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory with media files.")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR), help="Directory to save the output JSON results.")
    parser.add_argument("--vit_weights", type=str, help=f"Path to ViT weights. Defaults to file in {WEIGHTS_DIR}.")
    parser.add_argument("--effnet_weights", type=str, help=f"Path to EfficientNetV2 weights. Defaults to file in {WEIGHTS_DIR}.")
    parser.add_argument("--temporal_weights", type=str, help=f"Path to Temporal CNN weights. Defaults to file in {WEIGHTS_DIR}.")
    
    args = parser.parse_args()
    main(args)