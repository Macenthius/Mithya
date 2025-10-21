# Mithya/download_models.py

"""
Utility script to download pre-trained model weights.
"""
import requests
from tqdm import tqdm
import os
import logging

from config import WEIGHTS_DIR

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model URLs (Replace with actual URLs to your trained models) ---
# For demonstration, these are placeholders. You would host your .pth files
# on a service like GitHub Releases, Google Drive, or AWS S3.
MODEL_URLS = {
    "vit_best.pth": "https://www.example.com/models/vit_best.pth",
    "efficientnet_best.pth": "https://www.example.com/models/efficientnet_best.pth",
    "temporal_best.pth": "https://www.example.com/models/temporal_best.pth",
}

def download_file(url: str, dest_path: str):
    """
    Downloads a file from a URL to a destination path with a progress bar.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                size = f.write(chunk)
                bar.update(size)
        logger.info(f"Successfully downloaded {os.path.basename(dest_path)}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}. Error: {e}")
        logger.warning("This is a placeholder script. Please replace the URLs with your actual model weights.")
        if os.path.exists(dest_path):
            os.remove(dest_path) # Clean up partial download

def main():
    """

    Main function to download all required models.
    """
    logger.info(f"Starting download of model weights to {WEIGHTS_DIR}...")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        destination = WEIGHTS_DIR / filename
        if destination.exists():
            logger.info(f"{filename} already exists. Skipping.")
        else:
            download_file(url, str(destination))
            
    logger.info("Model download process finished.")

if __name__ == "__main__":
    main()