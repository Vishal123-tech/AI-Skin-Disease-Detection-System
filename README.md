# AI Skin Disease Detection Prototype

[![View Repository](https://img.shields.io/badge/GitHub-View%20Repository-181717?logo=github)](https://github.com/Vishal123-tech/AI-Skin-Disease-Detection-System)
[![Launch Local Demo](https://img.shields.io/badge/Demo-Launch%20Local%20App-1976D2)](http://127.0.0.1:5000/)

**Project links:** [View Repository](https://github.com/Vishal123-tech/AI-Skin-Disease-Detection-System) · [Launch Local Demo](http://127.0.0.1:5000/)

## Share the app with other people

The local demo link works only on the computer running Flask. To test from another device on the same Wi-Fi, start the app with:

```bash
python app.py --host 0.0.0.0
```

Then find your computer's local IP address with `ipconfig` and share `http://YOUR_IP:5000` with devices on the same network.

For a public internet link, deploy this repository to Render. The included `render.yaml` and `requirements-deploy.txt` configure the web service automatically. After deployment, Render will provide a public HTTPS URL that can be shared with anyone.

Educational prototype for capturing a skin image, running a TensorFlow Lite classifier, and generating a PDF report.

> This is not a medical diagnostic device. Predictions must not replace a qualified dermatologist.

## Features

- Raspberry Pi Camera capture, with USB/webcam fallback
- Image-quality checks for blur, brightness, and minimum resolution
- TensorFlow Lite image classification
- Demo mode when no model is installed
- PDF report with image, prediction, confidence, timestamp, and disclaimer
- Training script for transfer learning with an `ImageFolder` dataset

## Quick start on Windows/Linux

```bash
python -m venv .venv
 .venv\\Scripts\\activate       # Windows
source .venv/bin/activate        # Linux/Raspberry Pi
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` and upload a skin image. The report is saved in `reports/`.

## Raspberry Pi camera

Install Picamera2 through Raspberry Pi OS, then run:

```bash
python app.py --camera
```

The app will use the Pi camera when available and fall back to an uploaded image when it is not.

## Included model

The project includes `models/skin_model.tflite` and `models/labels.txt`, sourced from the open-source [Skin-Disease-Detection-App](https://github.com/ananmaysuri/Skin-Disease-Detection-App) repository. It is a cancer-lesion classifier with six classes. The included model is for educational experimentation only; it has not been clinically validated for this project.

To enable real inference, install TensorFlow separately:

```bash
pip install tensorflow
```

If TensorFlow is unavailable, the web app remains usable in clearly labelled demo mode.

## Replace the model

Place these files in `models/`:

- `skin_model.tflite`
- `labels.txt` — one class name per line, in model output order

Without these files, the app uses a clearly marked demo prediction so the complete workflow can be tested.

## Train a model

Arrange images like this:

```text
data/skin/
  train/acne/*.jpg
  train/eczema/*.jpg
  train/melanoma/*.jpg
  val/acne/*.jpg
  val/eczema/*.jpg
  val/melanoma/*.jpg
```

Then run:

```bash
python train.py --data data/skin --output models/skin_model.tflite --labels models/labels.txt
```

Use legally obtained, clinically appropriate data. Do not scrape or redistribute patient images without permission.
