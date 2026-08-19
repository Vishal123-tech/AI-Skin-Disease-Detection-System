from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "skin_model.tflite"
LABELS_PATH = ROOT / "models" / "labels.txt"
UPLOAD_DIR = ROOT / "uploads"
REPORT_DIR = ROOT / "reports"
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_MB = 10

for folder in (MODEL_PATH.parent, UPLOAD_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
