# Mithya/app.py

"""
Streamlit Web Application for the Mithya Deepfake Detection System.
"""

import streamlit as st
import tempfile
from pathlib import Path
import logging

# Import from our project
from config import DEVICE, WEIGHTS_DIR, ALLOWED_VIDEO_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS
from models.ensemble import EnsembleModel

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_model():
    """
    Loads the EnsembleModel and its weights.
    The @st.cache_resource decorator ensures this function is run only once.
    """
    try:
        logger.info("Loading Mithya EnsembleModel...")
        model = EnsembleModel(device=DEVICE)
        
        vit_weights = WEIGHTS_DIR / "vit_best.pth"
        effnet_weights = WEIGHTS_DIR / "efficientnet_best.pth"
        
        # MODIFIED: Removed the check for temporal_weights
        if not all([vit_weights.exists(), effnet_weights.exists()]):
            st.error("Model weights not found! Please ensure 'vit_best.pth' and 'efficientnet_best.pth' are in the /weights directory.")
            return None
            
        # MODIFIED: Pass temporal_path=None
        model.load_weights(
            vit_path=str(vit_weights), 
            effnet_path=str(effnet_weights), 
            temporal_path=None
        )
        
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Main Application UI ---
st.set_page_config(page_title="Mithya Deepfake Detection", layout="wide")

st.title("🛡️ Mithya - Advanced Deepfake Detection System")
st.markdown("Upload a video or image file to analyze it for signs of manipulation.")

# Load the model
model = load_model()

# Sidebar for additional information
st.sidebar.header("About Mithya")
st.sidebar.info(
    "Mithya uses a state-of-the-art ensemble of deep learning models, "
    "including Vision Transformers and 3D-CNNs, to provide a robust "
    "and accurate deepfake detection score."
)
st.sidebar.markdown("---")
st.sidebar.write("Developed with Gemini 2.5 Pro")


if model is None:
    st.warning("Model could not be loaded. The application is not functional.")
else:
    # File uploader
    allowed_types = list(ext.strip('.') for ext in ALLOWED_VIDEO_EXTENSIONS.union(ALLOWED_IMAGE_EXTENSIONS))
    uploaded_file = st.file_uploader(
        "Choose a file...",
        type=allowed_types
    )

    if uploaded_file is not None:
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Display the uploaded file
        file_type = Path(uploaded_file.name).suffix.lower()
        if file_type in ALLOWED_VIDEO_EXTENSIONS:
            st.video(tmp_file_path)
        else:
            st.image(tmp_file_path)

        # Analyze button and processing
        if st.button("Analyze File", use_container_width=True):
            with st.spinner('Analyzing... This may take a moment.'):
                try:
                    # Based on the file type, call the appropriate prediction method
                    if file_type in ALLOWED_VIDEO_EXTENSIONS:
                        results = model.predict_video(tmp_file_path)
                    else: # It's an image
                        if hasattr(model, 'predict_image'):
                             results = model.predict_image(tmp_file_path)
                        else:
                             results = model.predict_video(tmp_file_path)

                    # Display results
                    if results and results.get("status") == "success":
                        st.subheader("Analysis Results")
                        score = results['ensemble_score']
                        prediction = results['final_prediction']
                        
                        if prediction == "FAKE":
                            st.error(f"**Verdict: FAKE** (Confidence: {score:.2%})")
                        else:
                            st.success(f"**Verdict: REAL** (Confidence: {1 - score:.2%})")
                        
                        st.progress(score)
                        
                        with st.expander("Show Detailed Score Breakdown"):
                            st.json(results['individual_scores'])
                            
                    else:
                        st.warning(f"Analysis failed: {results.get('message', 'An unknown error occurred.')}")
                
                except Exception as e:
                    logger.error(f"Error during analysis: {e}")
                    st.error(f"A critical error occurred during the analysis process: {e}")