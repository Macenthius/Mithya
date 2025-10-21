# Mithya/inference.py

"""
Command-line inference script for single file deepfake detection.

This script loads the trained ensemble model and runs detection on a
given video or image file, outputting the results in JSON format.
"""

import argparse
import logging
import json
from pathlib import Path
import torch

# Import from our project
from config import (
    DEVICE, WEIGHTS_DIR, ALLOWED_VIDEO_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS
)
# NOTE: This script assumes the EnsembleModel class has been updated
# with a `predict_image` method for single image inference.
from models.ensemble import EnsembleModel

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_default_weight_paths() -> dict:
    """Gets the default paths for the best-performing models."""
    return {
        "vit_weights": WEIGHTS_DIR / "vit_best.pth",
        "effnet_weights": WEIGHTS_DIR / "efficientnet_best.pth",
        "temporal_weights": WEIGHTS_DIR / "temporal_best.pth", # Assumes a temporal model is trained
    }

def main(args):
    """
    Main function to run the inference process.
    """
    # --- Input Validation ---
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # --- Model Loading ---
    logger.info("Initializing the EnsembleModel...")
    try:
        # For this script, we assume a slightly modified EnsembleModel
        # which can handle image predictions gracefully.
        ensemble = EnsembleModel(device=DEVICE)
        
        default_weights = get_default_weight_paths()
        vit_w = args.vit_weights or default_weights["vit_weights"]
        eff_w = args.effnet_weights or default_weights["effnet_weights"]
        temp_w = args.temporal_weights or default_weights["temporal_weights"]
        
        logger.info("Loading model weights...")
        ensemble.load_weights(vit_path=str(vit_w), effnet_path=str(eff_w), temporal_path=str(temp_w))
        logger.info("Model ready for inference.")

    except Exception as e:
        logger.error(f"Failed to initialize or load the model: {e}")
        return

    # --- Prediction ---
    file_extension = input_path.suffix.lower()
    results = None
    
    logger.info(f"Starting prediction for {input_path}...")
    
    if file_extension in ALLOWED_VIDEO_EXTENSIONS:
        results = ensemble.predict_video(str(input_path))
    elif file_extension in ALLOWED_IMAGE_EXTENSIONS:
        # This requires predict_image to be implemented in EnsembleModel
        # The hypothetical implementation would run spatial/frequency models
        # and re-normalize the ensemble weights.
        if hasattr(ensemble, 'predict_image'):
             results = ensemble.predict_image(str(input_path))
        else:
             logger.warning("Single image prediction is not implemented in this version of EnsembleModel. Skipping.")
             # Fallback to a single-frame video prediction
             results = ensemble.predict_video(str(input_path))

    else:
        logger.error(f"Unsupported file type: {file_extension}. Please use video or image files.")
        return

    if not results:
        logger.error("Prediction failed. Please check the logs.")
        return

    # --- Output Results ---
    pretty_results = json.dumps(results, indent=4)
    logger.info(f"Prediction Results:\n{pretty_results}")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to {output_path}")
        except IOError as e:
            logger.error(f"Could not write results to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake detection on a single video or image.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input video or image file.")
    parser.add_argument("-o", "--output", type=str, help="Optional. Path to save the output JSON results.")
    parser.add_argument("--vit_weights", type=str, help=f"Path to ViT weights. Defaults to pre-trained file in {WEIGHTS_DIR}.")
    parser.add_argument("--effnet_weights", type=str, help=f"Path to EfficientNetV2 weights. Defaults to pre-trained file in {WEIGHTS_DIR}.")
    parser.add_argument("--temporal_weights", type=str, help=f"Path to Temporal CNN weights. Defaults to pre-trained file in {WEIGHTS_DIR}.")
    
    args = parser.parse_args()
    main(args)