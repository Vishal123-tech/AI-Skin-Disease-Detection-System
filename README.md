---
title: AI Skin Disease Detection
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: gradio_app.py
app_port: 7860
short_description: AI skin lesion and normal skin image classification demo
---

# AI Skin Disease & Normal Skin Detection Prototype

[![View Repository](https://img.shields.io/badge/GitHub-View%20Repository-181717?logo=github)](https://github.com/Vishal123-tech/AI-Skin-Disease-Detection-System)
[![Launch Live Web App](https://img.shields.io/badge/Demo-Launch%20Live%20App-4CAF50)](https://ai-skin-disease-detection-system.onrender.com)

**Project links:** [View Repository](https://github.com/Vishal123-tech/AI-Skin-Disease-Detection-System) · [Launch Live Web App](https://ai-skin-disease-detection-system.onrender.com)

Educational prototype for capturing skin images, checking photo quality, running a LiteRT/TensorFlow Lite classifier for **skin disease lesions**, **normal healthy skin**, and **non-skin objects**, and generating a PDF report.

> **Disclaimer:** This is an educational research prototype and not a medical diagnostic device. Predictions must never replace a qualified dermatologist or medical practitioner.

---

## Features

- **Multi-Class Skin & Non-Skin Classification:** Distinguishes between skin disease lesions, **Normal Healthy Skin**, and **Non-Skin/Other Objects** to eliminate false positive disease diagnoses.
- **Out-of-Distribution Guardrails:** Low-confidence predictions (<45%) are automatically flagged as uncertain or non-skin inputs.
- **Image Quality Checks:** Automated checks for blur, brightness, and resolution.
- **LiteRT / TFLite Inference:** Lightweight, hardware-friendly local inference on PC and Raspberry Pi.
- **Demo Mode:** Fully functional workflow even when TensorFlow models are not installed.
- **PDF Report Generation:** Generates styled diagnostic reports with images, classification category, confidence, and quality metrics.
- **Dataset Setup & Augmentation:** Built-in `setup_dataset.py` helper to structure class folders and generate synthetic samples.

---

## Quick Start on Windows / Linux / macOS

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux / macOS / Raspberry Pi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Flask Web Application
python app.py
```

Open `http://127.0.0.1:5000` in your browser to analyze images.

---

## Share the Web App

### Local Wi-Fi Network
```bash
python app.py --host 0.0.0.0
```
Find your local IP address using `ipconfig` (Windows) or `ifconfig` (Linux) and share `http://YOUR_IP:5000` with devices on the same Wi-Fi.

### Gradio Interface & Public Hugging Face Space
Run the Gradio interface:
```bash
python gradio_app.py
```
Upload this repository to a Hugging Face Space using the **Gradio** SDK (`app_file: gradio_app.py`, `requirements-space.txt`) to generate a public HTTPS URL.

---

## Dataset & Training Guide

### 1. Set Up Dataset Folder Structure
Run `setup_dataset.py` to automatically create all 8 class directories (`Actinic Keratoses`, `Basal Cell Carcinoma`, `Benign Keratosis like Lesions`, `Dermatofibroma`, `Melanocytic Nevi`, `Vascular Lesions`, `Normal_Skin`, `Other_Non_Skin`):

```bash
python setup_dataset.py
```

To test the training pipeline immediately with synthetic sample images:
```bash
python setup_dataset.py --generate-samples
```

### 2. Organize your Data
Place your image files (`.jpg`, `.png`) under `data/skin/`:
```text
data/skin/
  train/
    Normal_Skin/*.jpg
    Other_Non_Skin/*.jpg
    Actinic Keratoses/*.jpg
    ...
  val/
    Normal_Skin/*.jpg
    Other_Non_Skin/*.jpg
    Actinic Keratoses/*.jpg
    ...
```

### 3. Train & Export the Model
```bash
python train.py --data data/skin --epochs 8 --fine-tune-epochs 4 --output models/skin_model.tflite --labels models/labels.txt
```

---

## Inference Runtimes

Local inference automatically selects the best available engine in order of preference:
1. `ai-edge-litert` (Google LiteRT)
2. `tflite-runtime`
3. `tensorflow`

If no runtime or model file is present, the app gracefully operates in **Demo Mode**.
