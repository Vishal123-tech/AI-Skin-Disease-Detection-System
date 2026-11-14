"""Local-only SkinScanix UI trial with the external DINOv2 research model."""

import argparse
import importlib.metadata as metadata
import os
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "work" / "candidates" / "external_dinov2_20261113"
if not (HERE / "model" / "model.safetensors").is_file():
    raise SystemExit(
        "External model weights are missing. See README: Experimental pretrained-model trial."
    )
sys.path.insert(0, str(ROOT))
os.environ["GEMINI_API_KEY"] = ""
os.environ["ACNE_FOCUS_MODE"] = "0"
os.environ["ACNE_PRIORITY_MODE"] = "0"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
temp_root = ROOT / "work" / "t"
temp_root.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(temp_root)
os.environ["TMP"] = str(temp_root)
os.environ["GRADIO_TEMP_DIR"] = str(temp_root / "gradio")

import numpy as np
import torch
from PIL import Image, ImageOps

# The workspace's bundled CPU torch lacks a usable package metadata version.
original_version = metadata.version
metadata.version = lambda name: torch.__version__ if name == "torch" else original_version(name)
from transformers import AutoImageProcessor, AutoModelForImageClassification

from skinscanix import branding, config, feedback, self_train
from skinscanix.predictor import _get_disease_info

torch.set_num_threads(2)
processor = AutoImageProcessor.from_pretrained(HERE / "model", local_files_only=True, use_fast=False)
external_model = AutoModelForImageClassification.from_pretrained(HERE / "model", local_files_only=True)
external_model.eval()

trial = HERE / "runtime_trial"
config.REPORT_DIR = trial / "reports"
config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
config.GEMINI_API_KEY = ""
feedback.FEEDBACK_DIR = trial / "feedback"
feedback.FEEDBACK_IMAGE_DIR = feedback.FEEDBACK_DIR / "images"
feedback.FEEDBACK_LOG = feedback.FEEDBACK_DIR / "feedback.jsonl"
self_train.FEEDBACK_DIR = feedback.FEEDBACK_DIR
self_train.FEEDBACK_LOG = feedback.FEEDBACK_LOG
self_train.FEEDBACK_IMG_DIR = feedback.FEEDBACK_IMAGE_DIR
self_train.LEARNED_TRACKER = feedback.FEEDBACK_DIR / "learned_ids.json"
feedback.trigger_self_training = lambda **_kwargs: {
    "success": False,
    "message": "Training is disabled in this external-model trial.",
}

original_header = branding.website_header
def trial_header():
    return original_header() + (
        '<div class="safety-banner" role="note"><strong>External model trial — local only.</strong> '
        'This model performed worse on the main held-out test. Results are experimental; '
        'the active website model is unchanged.</div>'
    )
branding.website_header = trial_header

from skinscanix.gradio_app import demo, UI_LAUNCH_KWARGS, predictor

predictor.labels = [external_model.config.id2label[i] for i in range(external_model.config.num_labels)]
predictor.legacy_interpreter = None
predictor.legacy_labels = []
predictor.acne_focus = False
predictor.acne_priority = False

def external_probs(self, image_path):
    with Image.open(image_path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt")
    with torch.inference_mode():
        logits = external_model(**inputs).logits[0]
    return torch.softmax(logits, dim=-1).numpy()

predictor._run_local_model = MethodType(external_probs, predictor)
original_broad = predictor._predict_broad_local
supported = {
    "Acne and Rosacea Photos": "Acne Vulgaris",
    "Atopic Dermatitis Photos": "Eczema",
    "Eczema Photos": "Eczema",
    "Nail Fungus and other Nail Disease": "Fungal Infection",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal Infection",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis",
}

def external_broad(self, image_path):
    result = original_broad(image_path)
    result["model_used"] = "External DINOv2 local trial"
    result["notes"] = result.get("notes", "").replace(
        "Disease model: SCIN", "Disease model: External DINOv2 local trial"
    )
    raw = result["raw_class"]
    if result["category"] in ("non_skin", "uncertain") or raw == "normal skin":
        return result
    canonical = supported.get(raw)
    if canonical is None:
        result.update({
            "label": "Unable to assess — this model suggested a category outside SkinScanix's supported conditions",
            "category": "uncertain",
            "disease_info": {"description": "The external model's top category is outside the supported SkinScanix labels. No condition is confirmed.", "severity": "No condition confirmed", "emoji": "❓", "category": "uncertain"},
            "notes": result.get("notes", "") + " | Unsupported external class: " + raw,
        })
        return result
    info = _get_disease_info(canonical)
    result.update({
        "raw_class": canonical,
        "label": f"{info.get('emoji', '🩺')} {canonical} (external model suggestion)",
        "category": info.get("category", "lesion"),
        "disease_info": info,
        "notes": result.get("notes", "") + " | External raw class: " + raw,
    })
    return result

predictor._predict_broad_local = MethodType(external_broad, predictor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"External DINOv2 trial: http://{args.host}:{args.port}/", flush=True)
    demo.launch(server_name=args.host, server_port=args.port, share=False, **UI_LAUNCH_KWARGS)
