# Mithya/utils/metrics.py

"""
Utility functions for loading and parsing model training metrics.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

def load_training_history(metrics_path: Path) -> Tuple[List[Dict[str, Any]], float]:
    """
    Loads the training history JSON file created by 'train.py'.

    Args:
        metrics_path (Path): The path to the _metrics.json file.

    Returns:
        Tuple[List[Dict[str, Any]], float]: 
            - A list of history dictionaries, one for each epoch.
            - The best validation AUC recorded.
    """
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return [], 0.0
    
    try:
        data = json.loads(metrics_path.read_text())
        history = data.get("history", [])
        best_auc = data.get("best_val_auc", 0.0)
        
        if not history:
            logger.warning(f"Metrics file {metrics_path} contains no 'history' data.")
            
        return history, best_auc
    
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {metrics_path}. File may be corrupt.")
        return [], 0.0
    except Exception as e:
        logger.error(f"Failed to load or parse metrics file {metrics_path}: {e}")
        return [], 0.0