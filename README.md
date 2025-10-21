Here is a complete, professional `README.md` file for your project.

-----

# 🛡️ Mithya - Advanced Deepfake Detection System

Mithya (from Sanskrit: *mithyā*, meaning "false" or "untrue") is a state-of-the-art deepfake detection system designed to identify manipulated media with high accuracy. This project uses a modern ensemble of Vision Transformers and EfficientNetV2, built with PyTorch and MediaPipe, to provide a robust and production-ready detection solution.

This system is built to be modular, efficient, and easy to use, from training on new datasets to deploying as a real-time API.

## ✨ Key Features

  * **State-of-the-Art Models:** Utilizes an ensemble of **Vision Transformer (ViT)** and **EfficientNetV2** for high-accuracy spatial analysis.
  * **No Compilation Headaches:** Uses Google's **MediaPipe** for face detection, completely avoiding the complex setup of dlib.
  * **High Performance:** Optimized training script with **Automatic Mixed Precision (AMP)** and advanced data loading for GPU-accelerated training.
  * **Ready-to-Deploy:** Includes a user-friendly **Streamlit** web app for demos and a high-performance **FastAPI** REST API for production use.
  * **Modular & Extensible:** Built with a clean project structure that is easy to understand, modify, and extend.

## 🚀 Our Approach (SOTA 2024/2025)

This project avoids legacy approaches and focuses on current state-of-the-art techniques:

1.  **Vision Transformers (ViT):** The primary detector, leveraging the attention mechanism to capture global features and subtle inconsistencies that older CNNs miss.
2.  **EfficientNetV2:** A fast and powerful secondary detector that provides a complementary, lightweight analysis.
3.  **Ensemble Method:** The final prediction is a weighted average of the ViT and EfficientNetV2 scores, creating a more robust system than any single model.
4.  **Modern Tooling:** Uses `timm` for models, `albumentations` for augmentations, and `OneCycleLR` for fast, "super-convergence" training.

## 📂 Project Structure

```
Mithya/
├── data/
│   ├── processed_faces/        # (Generated) Cropped faces
│   ├── ff_raw_videos/          # (Downloaded) Raw video files
│   └── ff_dataset.csv          # (Generated) Master CSV for training
├── models/
│   ├── vit_detector.py         # Vision Transformer model
│   ├── efficient_detector.py   # EfficientNetV2 model
│   ├── temporal_3dcnn.py     # (Future) 3D CNN model
│   └── ensemble.py             # Manages all models
├── utils/
│   ├── face_extractor.py       # MediaPipe face detection
│   └── video_processor.py    # Frame extraction
├── weights/
│   ├── .gitkeep              # (Generated) Trained .pth files go here
│   └── vit_best.pth            # (Generated)
├── .gitignore
├── app.py                      # Streamlit Web App
├── api.py                      # FastAPI REST API
├── config.py                   # Central configuration
├── download_ff.py              # FaceForensics++ downloader
├── preprocess_ff.py            # Video-to-face-crop processor
├── requirements.txt            # All Python dependencies
├── train.py                    # Optimized training script
└── README.md                   # You are here
```

## 🏁 Getting Started: Step-by-Step

Follow these instructions to get the Mithya system running from scratch.

### 1\. Setup

First, set up the Python environment and install all dependencies.

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/your-username/Mithya.git
cd Mithya

# Create a Python virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### 2\. GPU Setup (HIGHLY Recommended)

To train or run inference on your NVIDIA GPU, you must install the CUDA-enabled version of PyTorch.

```bash
# First, uninstall the default CPU-only version (if installed)
pip uninstall torch torchvision torchaudio

# Install the CUDA-enabled version (this example uses CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify the installation
python -c "import torch; print(f'GPU Ready: {torch.cuda.is_available()}')"
# You should see "GPU Ready: True"
```

### 3\. Data Preparation

This is a two-step process: download the raw videos, then process them into face crops.

**Step 3.A: Download the Data**
We will download a small "developer" sample (50 real, 150 fake) from the FaceForensics++ dataset.

```bash
# Run these commands one by one
python download_ff.py -d original -c c23 -n 50
python download_ff.py -d Deepfakes -c c23 -n 50
python download_ff.py -d Face2Face -c c23 -n 50
python download_ff.py -d FaceSwap -c c23 -n 50
```

This will download all videos into the `data/ff_raw_videos/` directory.

**Step 3.B: Pre-process the Data**
This script will extract frames, detect faces, and create the `data/ff_dataset.csv` file for training.

```bash
# This will use your CPU to process all videos in the base directory
# This may take 30-60 minutes.
python preprocess_ff.py
```

### 4\. Model Training

Now that you have a dataset, you can train your models. The trained weights will be saved in the `weights/` folder.

```bash
# 1. Train the Vision Transformer (Primary Model)
python train.py --model vit --data_csv data/ff_dataset.csv --use_amp --verbose

# 2. Train the EfficientNetV2 (Secondary Model)
python train.py --model efficientnet --data_csv data/ff_dataset.csv --use_amp --verbose
```

*(Note: `--use_amp` enables mixed-precision for a 2x speedup on GPUs. `--verbose` shows progress bars.)*

### 5\. Run the Application

With the models trained, you can now launch the web app or the API.

**Option 1: Run the Streamlit Web App (For Demos)**

```bash
streamlit run app.py
```

This will open the application in your web browser. You can now upload a video file from your `data/ff_raw_videos/` folder to test it.

**Option 2: Run the FastAPI (For Production)**

```bash
uvicorn api:app --reload
```

This starts the API server. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.