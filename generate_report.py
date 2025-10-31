# Mithya/generate_report.py

"""
Generates a summary report and training plots for a given model.
This is the final script to run before making your presentation.
"""

import argparse
from pathlib import Path
import logging

# Import from our project
from utils.metrics import load_training_history
from utils.visualization import plot_training_curves
from config import WEIGHTS_DIR, RESULTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    model_name = args.model
    metrics_file = Path(WEIGHTS_DIR) / f"{model_name}_metrics.json"
    
    logger.info(f"Loading metrics from {metrics_file}...")
    history, best_auc = load_training_history(metrics_file)

    if not history:
        logger.error(f"No history found in {metrics_file}. Exiting.")
        return

    # --- Generate Console Report ---
    logger.info(f"--- Training Summary Report: {model_name.upper()} ---")
    
    # Find the best epoch's data
    best_epoch_data = max(history, key=lambda x: x['metrics'].get('auc', 0.0))
    best_epoch_num = best_epoch_data['epoch']
    
    # Print a clean, copy-paste-ready report
    print("\n" + "="*40)
    print(f"  MODEL: {model_name.upper()}")
    print("="*40)
    print(f"Total Epochs Run:      {len(history)}")
    print(f"Best Epoch:            {best_epoch_num}")
    print("\n--- Best Epoch Performance ---")
    print(f"  Validation AUC:      {best_epoch_data['metrics']['auc']:.4f}")
    print(f"  Validation Accuracy: {best_epoch_data['metrics']['accuracy']:.4f}")
    print(f"  Validation F1-Score: {best_epoch_data['metrics']['f1_score']:.4f}")
    print(f"  Validation Loss:     {best_epoch_data['val_loss']:.4f}")
    print("="*40 + "\n")

    # --- Generate Visualization ---
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_save_path = Path(RESULTS_DIR) / f"{model_name}_training_curves.png"
    
    logger.info(f"Generating training plot and saving to {plot_save_path}...")
    plot_training_curves(history, model_name, plot_save_path)
    
    logger.info("Report generation complete.")
    logger.info(f"Find your plot at: {plot_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training report and plots.")
    parser.add_argument("--model", type=str, required=True, choices=["vit", "efficientnet"],
                        help="The model to generate a report for (e.g., 'vit').")
    args = parser.parse_args()
    main(args)