---
title: AI Skin Disease Detection
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: gradio_app.py
app_port: 7860
python_version: "3.12"
short_description: Educational skin image screening prototype
---

# AI Skin Disease Detection

Educational prototype for capturing a skin image, rejecting obvious non-skin images, running a lightweight image classifier, and generating a PDF report.

> This project is not a medical diagnostic device. It must not replace a dermatologist or other qualified healthcare professional.

## Live demo

**[Launch the Web Demo](https://ai-skin-disease-detection-system.onrender.com/)**

Open the demo on a phone or computer, upload an image, select **Analyze Image**, and download the report. The free Render instance can take a while to wake up after inactivity.

## Features

- Dark Gradio interface for image upload, analysis, and report download.
- Skin/non-skin checks using a trained gate and image heuristics.
- Acne-priority inference, a common-condition classifier, and a legacy lesion model.
- Blur and brightness measurements, plus uncertain-input handling.
- Professional, paginated PDF reports with the uploaded image, classification, confidence, quality metrics, analysis summary, next steps, and disclaimer.
- Consent-based feedback collection and experimental, manually triggered retraining.
- Separate Flask JSON API and Raspberry Pi capture/upload client.

## Architecture

The browser flow is:

```text
Image upload -> Gradio -> image checks + model inference -> result + PDF
```

The separate hardware/API flow is:

```text
Raspberry Pi Camera -> Wi-Fi -> Flask JSON API -> server models -> result + PDF URL
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
├── self_train.py       # Experimental manual feedback retraining
├── review_feedback.py  # Inspect/export feedback before retraining
├── drive_feedback.py   # Optional private Google Drive feedback storage
├── setup_google_drive_oauth.py # Create a local Drive OAuth token
├── train.py            # Train and export a replacement classifier
├── train_acne.py       # Train a focused Acne vs Not-Acne classifier
├── setup_dataset.py    # Create dataset folders/sample data for testing
├── models/             # PyTorch/TFLite models, labels, and metadata
├── data/               # Local training data; do not commit private images
├── docs/               # Architecture and project status
├── Dockerfile          # Container build
└── render.yaml         # Render deployment configuration
```

## Run locally

Use Python 3.12 on a supported Windows/Linux x86-64 computer. Clone the project and create a virtual environment:

```bash
git clone https://github.com/Vishal123-tech/AI-Skin-Disease-Detection-System.git
cd AI-Skin-Disease-Detection-System
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on Linux.

### Browser interface

This is the simple Gradio layout matching the current live design:

```bash
python -m pip install -r requirements-space.txt
python gradio_app.py
```

Open: `http://127.0.0.1:7860/`

This address works only on the computer running the app. Keep its process running. The local launcher also attempts to create a temporary Gradio share link and prints it in the terminal.

The browser requirements include CPU-only PyTorch, torchvision, Gradio, and LiteRT. Full TensorFlow is not needed for this inference setup. The pinned CPU wheels target Windows/Linux x86-64; the Raspberry Pi camera client does not need the server's model dependencies.

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

Install Raspberry Pi OS, Picamera2, OpenCV, Python, and `requests`. The Pi captures and uploads images; it does not need a display or TensorFlow Lite when the server performs inference.

For a local-network trial, start the Flask API on the inference computer:

```bash
python app.py --host 0.0.0.0 --port 5000
```

Allow port 5000 on the trusted network and use that computer's LAN address below. Hardware operation still needs verification on the actual Pi and camera.

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

The current public Render service starts Gradio and does not expose the Flask `/api/predict` route. The Pi client needs a separately hosted Flask API or an adaptation to the Gradio API before it can use the public service directly. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Models

- `models/skin_gate.tflite`: trained two-class `SKIN`/`NON_SKIN` gate.
- `models/skin_gate_labels.txt`: gate label order.
- `models/skin_model.pt`: PyTorch MobileNetV2 checkpoint, preferred by the local backend when available.
- `models/skin_model.tflite`: locally trained four-class SCIN baseline exported from Google Colab.
- `models/labels.txt`: `Acne Vulgaris`, `Eczema`, `Fungal Infection`, and `Psoriasis`.
- `models/skin_model_legacy.tflite`: previous seven-class HAM10000/dermoscopic model used as a local fallback.
- `models/labels_legacy.txt`: labels for the previous lesion model.
- `models/skin_model_metadata.json`: input preprocessing and model-scope metadata used by the predictor.
- `models/acne_model.tflite`: bundled focused two-class acne model.
- `models/acne_labels.txt`: labels for the optional acne model.

The prediction pipeline combines the trained gate, image heuristics, acne-priority inference, and disease experts. The main classifier covers Acne Vulgaris, Eczema, Fungal Infection, and Psoriasis. The legacy expert covers Actinic Keratoses, Basal Cell Carcinoma, Benign Keratosis like Lesions, Dermatofibroma, Melanocytic Nevi, Melanoma, and Vascular Lesions.

The predictor prefers the PyTorch checkpoint, uses LiteRT for TFLite models, and can select the legacy expert when the main expert is uncertain. This is not a single validated 11-class classifier. Gates can reject genuine skin images and can also admit unrelated objects. A Normal Skin feedback option does not establish a validated healthy-skin classifier.

### Feedback for future improvement

After an analysis, the Gradio interface lets a user mark the result as
**Correct**, **Wrong**, or **Not sure**. A wrong result can be given a corrected
condition label. Feedback is stored locally in the ignored `feedback/` folder
for later use. Saving feedback alone does not train the model.

The **Retrain Model on Feedback Now** button explicitly runs `self_train.py`. It uses eligible user-labelled feedback and available replay images, backs up model files, updates the PyTorch checkpoint, and reloads it. Corrections are not automatically verified by a dermatologist, and retraining does not guarantee better accuracy.

The current interface exposes this control without a separate administrator login. Access control, reviewed labels, and independent evaluation remain necessary for a public feedback-learning service. Substantial training should run locally or in Colab rather than on the small Render inference instance.

Feedback and model changes inside a cloud instance are not durable unless persistent storage is configured. They are not automatically pushed to GitHub. The [feedback workflow guide](docs/FEEDBACK_LOOP.md) describes the earlier supervised workflow; the current code additionally supports manual retraining.

### Optional permanent Google Drive storage

Google Drive helper scripts are included, but the current `record_feedback()`
function saves locally and does not call them. Automatic Drive backup still
needs integration and testing. For configuring the helper, enable the Google Drive API in Google Cloud, create an OAuth
Desktop client, install `requirements-google-drive.txt`, and run:

```powershell
python setup_google_drive_oauth.py C:\path\to\client_secret.json
```

Set `GOOGLE_DRIVE_FOLDER_ID` to the ID of a private `skin-feedback` folder and
set `GOOGLE_DRIVE_TOKEN_JSON` to the contents of the generated token file in
Render/Hugging Face secrets. Never commit either credential. Setting these
values alone does not connect Drive storage to the current feedback save path.

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
│   ├── Acne Vulgaris/
│   ├── Eczema/
│   ├── Fungal Infection/
│   └── Psoriasis/
├── val/                 # Same class folders
└── test/                # Same class folders; held out from training
```

Train on Google Colab or a capable computer, not on the Raspberry Pi:

```bash
python train.py --data data/skin --epochs 8 --output work/candidate/skin_model.tflite --labels work/candidate/labels.txt
```

`train.py` selects PyTorch first when installed and saves a `.pt` checkpoint. Otherwise it can use TensorFlow and export TFLite; `--fine-tune-epochs` applies to that TensorFlow path. Without either framework, its fallback only creates labels and does not train a model. The example writes a candidate outside the deployed model folder. Preserve matching labels and test the candidate before replacing bundled artifacts.

The SCIN notebooks provide dataset preparation and baseline training. SCIN is not Maharashtra-specific; regional performance needs consented, expert-labelled evaluation data representing the intended users and cameras.

**Current validation limitation:** the committed main-model metadata records a run with **one feedback image and zero replay images**. Its `validation_accuracy: 1.0` is not a reliable accuracy estimate: with one sample, the training script reuses that image for validation. No general or clinical accuracy claim can be made from that run.

Evaluate per-class precision/recall, a confusion matrix, non-skin rejection, healthy skin, varied skin tones, and real phone-camera images. Keep each person's/case's images in one split, and keep evaluation images out of replay training.

## Deployment

### Render

The verified live service uses the **Dockerfile** and starts `gradio_app.py`, binding to `0.0.0.0` and Render's `PORT`. It installs CPU-only PyTorch and LiteRT to reduce inference memory use.

On **18 September 2026**, commit `8958d79` deployed successfully. A live test rejected a non-skin slide and generated the new two-page PDF. This verifies deployment and report generation, not disease-classification accuracy. The existing service required a manual deployment; a GitHub push alone does not confirm a live update.

The checked-in `render.yaml` describes an older Python-runtime setup. It does not include the Dockerfile's CPU PyTorch installation step. Use the Docker runtime and Dockerfile to reproduce the verified deployment.

### Hugging Face Spaces

The README metadata selects Gradio, Python 3.12, and `gradio_app.py`. The additional inference dependencies are in `requirements-space.txt`. A Gradio Space installs its root `requirements.txt`, so merely uploading `requirements-space.txt` is insufficient.

Prepare a separate Space deployment folder: combine the base requirements with the extra entries in `requirements-space.txt` into its root `requirements.txt`, removing the `-r requirements.txt` include to avoid recursion. Include the app modules and matching model files. Exclude credentials, feedback images, training datasets, and generated reports. Authenticated deployment and build verification remain pending; no verified Hugging Face demo URL is available yet.

## Current status

| Area | Status |
|---|---|
| Gradio interface and bundled inference models | Implemented; live on Render |
| Professional PDF report | Implemented; live generation verified |
| Feedback and manual retraining | Experimental; needs reviewed data and independent evaluation |
| Raspberry Pi capture and Flask upload | Code included; hardware trial pending |
| Hugging Face deployment | Configuration prepared; authenticated deployment pending |
| Durable cloud feedback storage | Not connected to the current feedback save path |
| Disease accuracy and regional validation | More labelled data and independent testing required |

Supporting guides: [Architecture](docs/ARCHITECTURE.md), [Acne training](docs/ACNE_TRAINING.md), [Colab training](docs/COLAB_TRAINING.md), and [Project status](docs/PROJECT_STATUS.md). These guides include earlier milestones; use the source and status above for the current implementation.
