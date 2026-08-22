import os
from pathlib import Path

# ── Load .env file if present (dev convenience) ───────────────────────────────
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "skin_model.tflite"
LABELS_PATH = ROOT / "models" / "labels.txt"
UPLOAD_DIR = ROOT / "uploads"
REPORT_DIR = ROOT / "reports"
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_MB = 10

# Gemini Vision API key — set in .env or as environment variable
# When set, enables 30+ disease detection and non-skin image rejection
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

for folder in (MODEL_PATH.parent, UPLOAD_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
