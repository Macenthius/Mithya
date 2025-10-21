# Mithya/api.py

"""
FastAPI REST API for the Mithya Deepfake Detection System.

To run this API server:
1. Make sure you have all requirements installed: pip install -r requirements.txt
2. Make sure you have downloaded the model weights.
3. Run the command in your terminal: uvicorn api:app --reload --port 8000
"""

import uvicorn
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any

from config import DEVICE, WEIGHTS_DIR, ALLOWED_VIDEO_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS
from models.ensemble import EnsembleModel

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Application Lifespan (Model Loading) ---
# Use a dictionary to hold the model instance
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    logger.info("Loading Mithya EnsembleModel for API...")
    try:
        model = EnsembleModel(device=DEVICE)
        vit_weights = WEIGHTS_DIR / "vit_best.pth"
        effnet_weights = WEIGHTS_DIR / "efficientnet_best.pth"
        temporal_weights = WEIGHTS_DIR / "temporal_best.pth"
        
        if not all([vit_weights.exists(), effnet_weights.exists(), temporal_weights.exists()]):
            raise RuntimeError("Model weights not found! Please run `python download_models.py` first.")
            
        model.load_weights(str(vit_weights), str(effnet_weights), str(temporal_weights))
        ml_models["ensemble_model"] = model
        logger.info("Model loaded and ready.")
    except Exception as e:
        logger.error(f"Fatal error during model loading: {e}")
        # If model fails to load, the API shouldn't start properly
        raise RuntimeError(f"Could not load ML model: {e}")
    
    yield
    
    # Clean up the ML models and release the resources
    ml_models.clear()
    logger.info("Cleaned up ML models.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Mithya Deepfake Detection API",
    description="An API for detecting deepfakes in videos and images.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API Schema ---
class DetectionResponse(BaseModel):
    status: str = Field(..., example="success")
    final_prediction: str = Field(..., example="FAKE")
    ensemble_score: float = Field(..., example=0.8734)
    individual_scores: Dict[str, float]
    message: str | None = Field(None, example="Analysis complete.")

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Perform a basic health check of the API."""
    return JSONResponse(content={"status": "ok", "model_loaded": "ensemble_model" in ml_models})

@app.post("/detect/video", response_model=DetectionResponse, summary="Detect Deepfake in Video/Image")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Accepts a video or image file, analyzes it, and returns the detection result.
    """
    model = ml_models.get("ensemble_model")
    if not model:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    file_extension = Path(file.filename).suffix.lower()
    supported_extensions = ALLOWED_VIDEO_EXTENSIONS.union(ALLOWED_IMAGE_EXTENSIONS)
    if file_extension not in supported_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{file_extension}'. Supported types are {supported_extensions}")

    try:
        # Save uploaded file to a temporary path to be processed
        with tempfile.NamedTemporaryFile(delete=True, suffix=file_extension) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
            
            # Run prediction
            logger.info(f"Processing file: {file.filename}")
            results = model.predict_video(tmp_file_path) # Using predict_video as the universal method

        if results and results.get("status") == "success":
            return DetectionResponse(
                status="success",
                final_prediction=results["final_prediction"],
                ensemble_score=results["ensemble_score"],
                individual_scores=results["individual_scores"],
                message="Analysis complete."
            )
        else:
            error_message = results.get("message", "Unknown processing error")
            raise HTTPException(status_code=500, detail=error_message)

    except Exception as e:
        logger.error(f"An error occurred during file processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# To run: uvicorn api:app --reload