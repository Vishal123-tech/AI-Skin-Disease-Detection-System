---
title: AI Skin Disease Detection
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: gradio_app.py
app_port: 7860
short_description: Educational skin image screening prototype
---

# AI Skin Disease Detection

Educational prototype for capturing a skin image, rejecting obvious non-skin images, running a lightweight image classifier, and generating a PDF report.

> This project is not a medical diagnostic device. It must not replace a dermatologist or other qualified healthcare professional.

## Architecture

The planned hardware flow is:

```text
Raspberry Pi Camera -> Wi-Fi -> Flask JSON API -> TFLite models -> mobile/browser result
```

The Raspberry Pi is only the camera and uploader in this design. Model inference runs on the server. TensorFlow Lite is the compact model format/runtime used by the server; it is not required on the Pi for cloud inference.

Detailed architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Project structure

```text
.
├── gradio_app.py       # Simple browser interface; current live-style UI
├── app.py              # Flask interface and JSON API at /api/predict
├── pi_capture.py       # Raspberry Pi camera capture and API upload client
├── predictor.py        # Gate + classifier inference and rejection rules
├── quality.py          # Blur, brightness, and image-quality checks
├── report.py           # PDF report generation
├── feedback.py         # Append-only feedback capture for supervised review
├── review_feedback.py  # Inspect/export feedback before retraining
├── drive_feedback.py   # Optional private Google Drive feedback storage
├── setup_google_drive_oauth.py # Create a local Drive OAuth token
├── train.py            # Train and export a replacement classifier
├── train_acne.py       # Train a focused Acne vs Not-Acne classifier
├── setup_dataset.py    # Create dataset folders/sample data for testing
├── models/             # TFLite models and output labels
├── data/               # Local training data; do not commit private images
├── docs/               # Architecture and project status
├── Dockerfile          # Container build
└── render.yaml         # Render deployment configuration
```

## Run locally

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate     # Linux/macOS/Raspberry Pi
pip install -r requirements.txt
```

### Browser interface

This is the simple Gradio layout matching the current live design:

```bash
pip install -r requirements-space.txt
python gradio_app.py
```

Open: `http://127.0.0.1:7860/`

### Flask API

Run this when the Raspberry Pi needs a direct JSON endpoint:

```bash
python app.py
```

Open the browser at `http://127.0.0.1:5000/`.

The API endpoint is:

```text
POST http://127.0.0.1:5000/api/predict
```

It accepts a multipart image field named `image` and returns JSON containing the category, label, confidence, quality result, and PDF report URL.

## Raspberry Pi

Install Raspberry Pi OS, Picamera2, Python, and `requests`. The Pi does not need TensorFlow Lite when the server performs inference.

Set the Flask API address before running the camera client:

```bash
export SERVER_URL="http://SERVER_IP:5000/api/predict"
python pi_capture.py
```

On Windows PowerShell, use:

```powershell
$env:SERVER_URL = "http://SERVER_IP:5000/api/predict"
python pi_capture.py
```

Do not set `SERVER_URL` to a normal browser page. The Pi needs a JSON API endpoint, not a Gradio HTML page. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the hosted deployment decision that remains.

## Models

- `models/skin_gate.tflite`: trained two-class `SKIN`/`NON_SKIN` gate.
- `models/skin_gate_labels.txt`: gate label order.
- `models/skin_model.tflite`: locally trained four-class SCIN baseline exported from Google Colab.
- `models/labels.txt`: `Acne Vulgaris`, `Eczema`, `Fungal Infection`, and `Psoriasis`.
- `models/skin_model_legacy.tflite`: previous seven-class HAM10000/dermoscopic model used as a local fallback.
- `models/labels_legacy.txt`: labels for the previous lesion model.
- `models/skin_model_metadata.json`: input preprocessing and model-scope metadata used by the predictor.
- `models/acne_model.tflite`: optional focused two-class acne model (created only after real acne training).
- `models/acne_labels.txt`: labels for the optional acne model.

The gate runs first and can reject objects, animals, documents, and rooms. The local predictor uses the SCIN model for the newer common-disease classes and falls back to the previous dermoscopic model when the SCIN model is weak. This is a two-model compatibility approach, not a single validated 11-class model.

### Feedback for future improvement

After an analysis, the Gradio interface lets a user mark the result as
**Correct**, **Wrong**, or **Not sure**. A wrong result can be given a corrected
condition label. Feedback is stored locally in the ignored `feedback/` folder
for supervised review; it never changes the model automatically. See
[`docs/FEEDBACK_LOOP.md`](docs/FEEDBACK_LOOP.md) before using any feedback for
retraining.

### Optional permanent Google Drive storage

For a small private prototype, feedback can also be copied to a private Google
Drive folder. Enable the Google Drive API in Google Cloud, create an OAuth
Desktop client, install `requirements-google-drive.txt`, and run:

```powershell
python setup_google_drive_oauth.py C:\path\to\client_secret.json
```

Set `GOOGLE_DRIVE_FOLDER_ID` to the ID of a private `skin-feedback` folder and
set `GOOGLE_DRIVE_TOKEN_JSON` to the contents of the generated token file in
Render/Hugging Face secrets. Never commit either credential. The app uploads
only after the user checks the consent box. If Drive is not configured, local
feedback still works.

### Acne-focused project mode

If acne is the main college-project target, train a separate binary model with
`train_acne.py` using real `Acne` and `Not_Acne` folders. After independently
testing it, the default `ACNE_PRIORITY_MODE=1` makes the local app check acne
first and then continue to the broader disease experts when acne is not the
best match. Set `ACNE_FOCUS_MODE=1` only if you want an acne-only demo.

The ISIC 2016 images supplied for this project do not have acne labels, so they
cannot train an acne detector by themselves. Use `prepare_acne_dataset.py`
after obtaining genuinely labelled acne and non-acne skin images.

See [`docs/ACNE_TRAINING.md`](docs/ACNE_TRAINING.md) for dataset requirements,
licensing, folder structure, and evaluation rules.

## Training a replacement model

Use a dataset with this structure:

```text
data/skin/
├── train/
│   ├── Acne/
│   ├── Eczema/
│   ├── Fungal_Infection/
│   ├── Normal_Skin/
│   └── ...
├── val/
└── test/
```

Train on Google Colab or a capable computer, not on the Raspberry Pi:

```bash
python train.py --data data/skin --epochs 8 --fine-tune-epochs 4 --output models/skin_model.tflite --labels models/labels.txt
```

The current local model was trained with MobileNetV2 transfer learning on the pure-label SCIN subset prepared by `COLAB_PREPARE_SCIN.ipynb` and trained by `COLAB_TRAIN_INDIAN_SKIN.ipynb`. The subset contains 393 images across four classes, so it is only a baseline. Do not use it for medical claims; evaluate each class separately and test with camera images before any live release.

## Deployment

The current Render/Hugging Face configuration starts `gradio_app.py` for the browser UI. The live site must remain unchanged until a replacement disease model and a direct Pi-to-server API path have passed local testing.

Current status and remaining work: [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)
