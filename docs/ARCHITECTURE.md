# Project Architecture

## Purpose

This is an educational skin-image screening prototype. It is not a medical diagnostic device.

The selected deployment design is cloud inference:

```text
Raspberry Pi camera
        |
        | captures JPEG and uploads it over Wi-Fi
        v
Web/API server
  - validates the image
  - runs the skin/non-skin gate
  - runs the disease model
  - creates the PDF report
        |
        v
Browser or mobile phone displays the result
```

## Responsibilities

| Component | Responsibility | Needs the disease model? |
|---|---|---:|
| `pi_capture.py` | Capture a photo with Picamera2/OpenCV and upload it | No |
| `app.py` | Flask web page and JSON endpoint `/api/predict` | Yes, on the server |
| `gradio_app.py` | Simple Gradio upload interface used by the current live UI | Yes, on the server |
| `predictor.py` | Load models, run the gate, classify, and apply rejection rules | Yes |
| `quality.py` | Check blur, brightness, and image quality | No |
| `report.py` | Generate the PDF report | No |
| `models/skin_gate.tflite` | Classify `SKIN` vs `NON_SKIN` | Yes |
| `models/skin_model.tflite` | Classify the supported lesion classes | Yes |
| `train.py` | Train/export a replacement classifier | Only during training |

## Local endpoints

### Gradio user interface

```text
python gradio_app.py
http://127.0.0.1:7860/
```

This is the simple upload interface that matches the current live design.

### Flask user interface and API

```text
python app.py
http://127.0.0.1:5000/
POST http://127.0.0.1:5000/api/predict
```

The Flask endpoint returns JSON and is the easiest endpoint for a Raspberry Pi client.

## Raspberry Pi connection

The Pi does not need TensorFlow Lite in cloud-inference mode. It needs only:

- Raspberry Pi OS
- Raspberry Pi Camera Module and Picamera2
- Python and `requests`
- Wi-Fi access to the server

The `SERVER_URL` in `pi_capture.py` must point to a reachable Flask `/api/predict` endpoint. A browser-only Gradio URL is not the same as a direct JSON API endpoint.

## Model flow

```text
Uploaded image
      |
      v
Skin/non-skin gate
      |
      +--> NON_SKIN: reject and request a skin photo
      |
      +--> SKIN: run the disease classifier
                    |
                    +--> supported lesion class or uncertain result
```

The current gate is trained. The current disease model is a temporary seven-class HAM10000 baseline converted from Hugging Face. It must be replaced with a properly trained and validated project model before claiming reliable disease-name classification.
