"""
predictor.py — Two-Stage AI Skin Disease Predictor
────────────────────────────────────────────────────
Stage 1 : Skin Gate — checks whether the uploaded image actually contains skin
           (HSV color analysis + optional Gemini API gate).
Stage 2 : Disease Classifier — runs Gemini Vision (30+ diseases) or
           local TFLite/PyTorch model with entropy-based rejection.

Non-skin images (tables, fans, food, etc.) are rejected before classification.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ─── Comprehensive Disease Taxonomy (30+ conditions) ─────────────────────────

DISEASE_INFO: dict[str, dict] = {
    # ── HAM10000 / ISIC dermoscopy classes ──────────────────────────────────
    "Actinic Keratoses": {
        "description": (
            "Rough, scaly patch on the skin caused by years of sun exposure. "
            "Can develop into squamous cell carcinoma if left untreated."
        ),
        "severity": "Moderate — See a dermatologist",
        "emoji": "☀️",
        "category": "lesion",
        "also_known_as": ["Solar Keratosis", "AK"],
    },
    "Basal Cell Carcinoma": {
        "description": (
            "Most common type of skin cancer. Appears as a pearly or waxy bump on "
            "sun-exposed areas (face, neck, hands). Grows slowly but must be treated."
        ),
        "severity": "High — Consult a dermatologist urgently",
        "emoji": "⚠️",
        "category": "lesion",
        "also_known_as": ["BCC"],
    },
    "Benign Keratosis like Lesions": {
        "description": (
            "Non-cancerous skin growths including seborrheic keratoses, solar "
            "lentigines and lichen-planus-like keratoses. Generally harmless."
        ),
        "severity": "Low — Monitor for changes",
        "emoji": "🟡",
        "category": "lesion",
        "also_known_as": ["Seborrheic Keratosis", "Age Spots", "Liver Spots"],
    },
    "Dermatofibroma": {
        "description": (
            "Harmless, firm nodule (bump) in the skin, usually brownish-pink. "
            "Commonly found on legs. Benign and rarely requires treatment."
        ),
        "severity": "Low — No treatment usually required",
        "emoji": "🟤",
        "category": "lesion",
        "also_known_as": ["Fibrous Histiocytoma"],
    },
    "Melanocytic Nevi": {
        "description": (
            "Common moles — benign growths formed by clusters of melanocytes. "
            "Apply the ABCDE rule: Asymmetry, Border, Color, Diameter, Evolution."
        ),
        "severity": "Low — Monitor for ABCDE changes",
        "emoji": "🔵",
        "category": "lesion",
        "also_known_as": ["Common Mole", "Nevi", "Birthmark"],
    },
    "Melanoma": {
        "description": (
            "Most dangerous form of skin cancer, developing from melanocytes. "
            "Can spread rapidly to other organs. Early detection is life-saving."
        ),
        "severity": "🚨 CRITICAL — Seek immediate medical attention",
        "emoji": "🚨",
        "category": "lesion",
        "also_known_as": ["Malignant Melanoma", "Skin Cancer"],
    },
    "Vascular Lesions": {
        "description": (
            "Abnormalities of blood vessels in or near the skin, including "
            "port-wine stains, cherry angiomas and spider angiomas."
        ),
        "severity": "Low to Moderate — Consult if growing or bleeding",
        "emoji": "🔴",
        "category": "lesion",
        "also_known_as": ["Cherry Angioma", "Spider Angioma", "Hemangioma"],
    },
    "Squamous Cell Carcinoma": {
        "description": (
            "Common skin cancer developing in squamous cells of the outer skin "
            "layer. Often looks like a firm red nodule or flat lesion with scaly crust."
        ),
        "severity": "High — Consult a dermatologist urgently",
        "emoji": "⚠️",
        "category": "lesion",
        "also_known_as": ["SCC"],
    },
    # ── Common everyday skin conditions ──────────────────────────────────────
    "Acne Vulgaris": {
        "description": (
            "Very common skin condition causing pimples, blackheads and whiteheads. "
            "Results from clogged hair follicles, bacteria and hormones."
        ),
        "severity": "Low to Moderate — OTC treatment or dermatologist for severe cases",
        "emoji": "🫧",
        "category": "lesion",
        "also_known_as": ["Pimples", "Zits", "Acne", "Breakouts", "Blackheads", "Whiteheads"],
    },
    "Eczema": {
        "description": (
            "Chronic inflammatory skin condition causing itchy, red, dry and cracked "
            "skin. Often triggered by allergens, stress or dry weather."
        ),
        "severity": "Moderate — Dermatologist recommended for management",
        "emoji": "🌡️",
        "category": "lesion",
        "also_known_as": ["Atopic Dermatitis", "Atopic Eczema"],
    },
    "Psoriasis": {
        "description": (
            "Autoimmune condition causing rapid skin cell buildup, resulting in "
            "scaly red patches that can be itchy and sometimes painful."
        ),
        "severity": "Moderate — Requires dermatologist management",
        "emoji": "🩹",
        "category": "lesion",
        "also_known_as": ["Plaque Psoriasis", "Psoriatic Lesion"],
    },
    "Rosacea": {
        "description": (
            "Chronic skin condition causing redness, visible blood vessels and "
            "acne-like breakouts mainly on the face."
        ),
        "severity": "Moderate — Dermatologist recommended",
        "emoji": "🌹",
        "category": "lesion",
        "also_known_as": ["Adult Acne", "Facial Redness"],
    },
    "Ringworm": {
        "description": (
            "A fungal infection (not an actual worm) that causes a ring-shaped, "
            "scaly, itchy rash on the skin. Highly contagious."
        ),
        "severity": "Moderate — Antifungal cream/medication required",
        "emoji": "⭕",
        "category": "lesion",
        "also_known_as": ["Tinea Corporis", "Fungal Infection"],
    },
    "Athlete's Foot": {
        "description": (
            "Fungal infection usually starting between the toes, causing itching, "
            "burning, stinging and scaling skin."
        ),
        "severity": "Low — OTC antifungal treatment usually effective",
        "emoji": "🦶",
        "category": "lesion",
        "also_known_as": ["Tinea Pedis"],
    },
    "Contact Dermatitis": {
        "description": (
            "Skin irritation or allergic reaction caused by direct contact with "
            "a substance — detergents, metals, plants or cosmetics."
        ),
        "severity": "Moderate — Identify and avoid trigger; topical steroids may help",
        "emoji": "☣️",
        "category": "lesion",
        "also_known_as": ["Allergic Rash", "Skin Allergy", "Allergic Contact Dermatitis"],
    },
    "Urticaria": {
        "description": (
            "Raised, itchy welts (hives) that appear suddenly, triggered by "
            "allergic reactions, stress, infections or medications."
        ),
        "severity": "Moderate — Antihistamines help; see doctor if severe/chronic",
        "emoji": "🐝",
        "category": "lesion",
        "also_known_as": ["Hives", "Nettle Rash"],
    },
    "Chickenpox": {
        "description": (
            "Highly contagious viral infection causing an itchy blister-like rash "
            "all over the body. Caused by the varicella-zoster virus."
        ),
        "severity": "Moderate — Rest, antihistamines; antiviral for severe cases",
        "emoji": "🔵",
        "category": "lesion",
        "also_known_as": ["Varicella"],
    },
    "Shingles": {
        "description": (
            "Painful rash caused by reactivation of the chickenpox virus. Often "
            "appears as a stripe of blisters wrapping around one side of the body."
        ),
        "severity": "High — Antiviral treatment needed urgently (within 72 hours)",
        "emoji": "⚡",
        "category": "lesion",
        "also_known_as": ["Herpes Zoster"],
    },
    "Impetigo": {
        "description": (
            "Highly contagious bacterial skin infection causing red sores that "
            "rupture and form honey-colored crusts. Common in children."
        ),
        "severity": "Moderate — Antibiotic ointment or oral antibiotics required",
        "emoji": "🧫",
        "category": "lesion",
        "also_known_as": ["School Sores"],
    },
    "Cellulitis": {
        "description": (
            "Serious bacterial infection of the deeper layers of skin and "
            "underlying tissue. Appears as swollen, red, warm and tender skin."
        ),
        "severity": "High — Seek medical care immediately; may need antibiotics/IV treatment",
        "emoji": "🆘",
        "category": "lesion",
        "also_known_as": ["Skin Infection", "Deep Skin Infection"],
    },
    "Seborrheic Dermatitis": {
        "description": (
            "Chronic condition causing scaly patches, red skin and dandruff. "
            "Mainly affects oily areas — scalp, face, sides of nose, eyebrows."
        ),
        "severity": "Low — Medicated shampoos and antifungal creams",
        "emoji": "❄️",
        "category": "lesion",
        "also_known_as": ["Dandruff", "Seborrhea", "Cradle Cap"],
    },
    "Vitiligo": {
        "description": (
            "Skin condition where patches of skin lose their pigment, resulting "
            "in white or light-colored patches. Not contagious or harmful."
        ),
        "severity": "Low — Cosmetic; consult dermatologist for treatment options",
        "emoji": "⬜",
        "category": "lesion",
        "also_known_as": ["Skin Depigmentation", "Leucoderma"],
    },
    "Folliculitis": {
        "description": (
            "Inflammation of hair follicles caused by bacterial or fungal "
            "infection. Looks like small red bumps or whiteheads around follicles."
        ),
        "severity": "Low to Moderate — Antibiotic creams; see doctor if spreading",
        "emoji": "🔴",
        "category": "lesion",
        "also_known_as": ["Hair Follicle Infection", "Razor Bumps"],
    },
    "Warts": {
        "description": (
            "Small, rough growths caused by the human papillomavirus (HPV). "
            "Usually harmless but can be contagious through direct contact."
        ),
        "severity": "Low — OTC wart remover, freezing (cryotherapy) or dermatologist",
        "emoji": "🟢",
        "category": "lesion",
        "also_known_as": ["Verruca", "HPV Wart", "Common Wart", "Plantar Wart"],
    },
    "Molluscum Contagiosum": {
        "description": (
            "Viral infection causing small, firm, pearl-like bumps with a central "
            "dimple. Common in children. Spreads by touch."
        ),
        "severity": "Low — Usually resolves on its own in 6–12 months",
        "emoji": "💧",
        "category": "lesion",
        "also_known_as": ["Water Warts"],
    },
    "Sunburn": {
        "description": (
            "Skin inflammation from overexposure to UV radiation. Results in red, "
            "painful, warm skin that may peel. Repeated sunburn raises cancer risk."
        ),
        "severity": "Low to Moderate — Cool water, aloe vera, hydration; avoid sun",
        "emoji": "☀️",
        "category": "lesion",
        "also_known_as": ["UV Burn", "Sun Damage", "Erythema Solare"],
    },
    "Scabies": {
        "description": (
            "Contagious infestation by tiny mites causing intense itching (especially "
            "at night) and a pimple-like rash, often between fingers and on wrists."
        ),
        "severity": "Moderate — Prescription scabicide (permethrin) required",
        "emoji": "🔬",
        "category": "lesion",
        "also_known_as": ["Mite Infestation", "Seven-Year Itch"],
    },
    "Cold Sores": {
        "description": (
            "Small fluid-filled blisters around the lips caused by herpes simplex "
            "virus type 1 (HSV-1). Recur in the same spot; can be triggered by stress."
        ),
        "severity": "Low — Antiviral creams (acyclovir); prescription antivirals for frequent recurrence",
        "emoji": "🌡️",
        "category": "lesion",
        "also_known_as": ["Herpes Labialis", "Fever Blisters", "Oral Herpes"],
    },
    "Tinea Versicolor": {
        "description": (
            "Fungal infection causing small discolored patches on the skin "
            "(lighter or darker than surrounding skin). Common in hot, humid weather."
        ),
        "severity": "Low — Antifungal shampoo or cream",
        "emoji": "🍂",
        "category": "lesion",
        "also_known_as": ["Pityriasis Versicolor"],
    },
    "Lupus Rash": {
        "description": (
            "A butterfly-shaped rash across the cheeks and nose is a hallmark of "
            "systemic lupus erythematosus (SLE), an autoimmune disease."
        ),
        "severity": "High — Rheumatologist referral needed immediately",
        "emoji": "🦋",
        "category": "lesion",
        "also_known_as": ["Butterfly Rash", "Malar Rash", "SLE Rash"],
    },
    # ── Normal / Non-skin ────────────────────────────────────────────────────
    "Normal_Skin": {
        "description": (
            "No significant skin lesion detected. The skin appears healthy with "
            "no visible abnormalities. Maintain a good skincare routine."
        ),
        "severity": "None — Keep up good skin care",
        "emoji": "✅",
        "category": "normal",
        "also_known_as": ["Healthy Skin", "Clear Skin"],
    },
    "Other_Non_Skin": {
        "description": (
            "The uploaded image does not appear to contain human skin. "
            "Please upload a clear, close-up photo of the skin area you want analyzed."
        ),
        "severity": "N/A — Please upload a clear photo of skin",
        "emoji": "❌",
        "category": "non_skin",
        "also_known_as": [],
    },
}


# ─── Helper Utilities ─────────────────────────────────────────────────────────

def _get_disease_info(raw_label: str) -> dict:
    """Fuzzy-match a raw model label to the disease taxonomy."""
    norm = raw_label.strip().lower().replace("_", " ")
    # Exact match first
    for k, v in DISEASE_INFO.items():
        if k.lower().replace("_", " ") == norm:
            return dict(v)
    # Partial match
    for k, v in DISEASE_INFO.items():
        kn = k.lower().replace("_", " ")
        if norm in kn or kn in norm:
            return dict(v)
    # Alias match
    for k, v in DISEASE_INFO.items():
        aliases = [a.lower() for a in v.get("also_known_as", [])]
        if norm in aliases or any(norm in a for a in aliases):
            return dict(v)
    return {
        "description": "Skin condition identified by AI. Consult a dermatologist for diagnosis.",
        "severity": "Consult a dermatologist",
        "emoji": "🩺",
        "category": "lesion",
    }


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution (in bits)."""
    p = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(p * np.log2(p)))


def skin_gate_hsv(image_path: str, min_skin_ratio: float = 0.10) -> tuple[bool, float]:
    """
    Check whether the image likely contains human skin using HSV color analysis.

    Returns (is_skin: bool, skin_ratio: float).
    Covers a wide range of skin tones (light → dark).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, 0.0

    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Skin-tone hue ranges (0-25° and 340-360° in OpenCV 0-180 scale → 0-12 and 170-180)
    # Saturation 20-170, Value 50-255 covers light to dark tones
    lower1 = np.array([0, 20, 50], dtype=np.uint8)
    upper1 = np.array([25, 170, 255], dtype=np.uint8)
    lower2 = np.array([170, 20, 50], dtype=np.uint8)
    upper2 = np.array([180, 170, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    # Remove isolated pixels. A few skin-coloured pixels in a table, wall, or
    # piece of furniture should not be enough to make the image look like skin.
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total = img.shape[0] * img.shape[1]
    skin_px = int(cv2.countNonZero(mask))
    ratio = skin_px / total

    # Require at least one reasonably sized connected region. This is an
    # out-of-distribution safeguard, not a medical segmentation algorithm.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    largest_ratio = 0.0
    if num_labels > 1:
        largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        largest_ratio = largest_area / total

    # The ratio threshold remains the primary test; the component test avoids
    # accepting scattered colour noise from non-skin images.
    is_skin = ratio >= min_skin_ratio and largest_ratio >= 0.02
    return is_skin, ratio


def looks_like_document(image_path: str) -> bool:
    """Reject common page/screenshot inputs before disease classification.

    This is deliberately conservative: it targets bright page-like images with
    dense writing/diagram edges, not a general-purpose object detector.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bright_ratio = float(np.mean(gray > 205))
    edge_ratio = float(np.mean(cv2.Canny(gray, 60, 140) > 0))
    low_saturation_ratio = float(np.mean(hsv[:, :, 1] < 65))

    # A photographed/scan-like page normally has a large bright, low-
    # saturation area plus many high-contrast writing or diagram edges.
    return bright_ratio > 0.45 and low_saturation_ratio > 0.45 and edge_ratio > 0.045


# ─── Main Predictor ───────────────────────────────────────────────────────────

class SkinPredictor:
    """
    Two-stage skin disease predictor.

    Priority order:
      1. Gemini Vision API   (30+ diseases, rejects non-skin images)
      2. Local PyTorch .pt   (original classes + entropy gate)
      3. Local TFLite model  (original classes + entropy gate)
      4. Demo mode           (no backend available)
    """

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        image_size: tuple[int, int] = (224, 224),
        min_confidence: float = 0.55,
        gemini_api_key: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.interpreter = None
        self.pt_model = None
        self.transform = None
        self.device = None
        self.labels: list[str] = []
        self.load_error: Optional[str] = None
        self.backend: Optional[str] = None
        self.gemini_model = None
        self.ollama_model = ollama_model or os.environ.get("OLLAMA_MODEL", "")
        self.ollama_url = (ollama_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")

        # Load labels
        if self.labels_path.exists():
            self.labels = [
                x.strip()
                for x in self.labels_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]

        # ── 1. Try Gemini Vision API ──────────────────────────────────────────
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self.backend = "gemini"
                print("Gemini Vision API ready - 30+ disease detection enabled")
            except Exception as exc:
                print(f"Gemini API failed to load: {exc}")

        # ── 2. Load local model (fallback) ────────────────────────────────────
        if self.gemini_model is None:
            self._load_local_model()

    # ─────────────────────────────────────────────────────────────────────────
    def _load_local_model(self) -> None:
        """Try PyTorch → TFLite → give up."""
        pt_path = self.model_path.with_suffix(".pt")

        # PyTorch
        if pt_path.exists():
            try:
                import torch
                import torch.nn as nn
                from torchvision import models, transforms  # type: ignore

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = models.mobilenet_v2()
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Sequential(
                    nn.Dropout(0.3), nn.Linear(num_features, len(self.labels))
                )
                model.load_state_dict(torch.load(pt_path, map_location=self.device))
                model.eval()
                model = model.to(self.device)
                self.pt_model = model
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                self.backend = "pytorch"
                print(f"PyTorch model loaded ({len(self.labels)} classes, device={self.device})")
                return
            except Exception as exc:
                self.load_error = str(exc)
                print(f"PyTorch load failed: {exc}")

        # TFLite
        if self.model_path.exists():
            try:
                try:
                    import ai_edge_litert.interpreter as tflite  # type: ignore
                    self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
                    self.backend = "litert"
                except ImportError:
                    try:
                        import tflite_runtime.interpreter as tflite  # type: ignore
                        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
                        self.backend = "tflite_runtime"
                    except ImportError:
                        import tensorflow as tf  # type: ignore
                        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                        self.backend = "tensorflow"

                self.interpreter.allocate_tensors()
                shape = self.interpreter.get_input_details()[0]["shape"]
                if len(shape) == 4:
                    self.image_size = (int(shape[2]), int(shape[1]))
                print(f"TFLite model loaded ({len(self.labels)} classes, backend={self.backend})")
            except Exception as exc:
                self.interpreter = None
                self.load_error = str(exc)
                print(f"TFLite load failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def demo_mode(self) -> bool:
        return (self.interpreter is None and self.pt_model is None
                and self.gemini_model is None and not self.ollama_model)

    # ─────────────────────────────────────────────────────────────────────────
    def _predict_gemini(self, image_path: str) -> dict:
        """Full Gemini Vision two-stage prediction."""
        # Encode image
        with open(image_path, "rb") as fh:
            img_bytes = fh.read()
        b64 = base64.b64encode(img_bytes).decode()
        ext = Path(image_path).suffix.lstrip(".").lower()
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "bmp": "image/bmp",
            "webp": "image/webp",
        }.get(ext, "image/jpeg")

        img_part = {"mime_type": mime, "data": b64}

        # ── Stage 1: Is this a skin image? ───────────────────────────────────
        gate_prompt = (
            "You are a medical image triage system.\n"
            "Look at this image and determine if it shows human skin "
            "(any body part, skin condition, lesion, rash, mole, or healthy skin).\n"
            "Respond ONLY with valid JSON, no markdown:\n"
            '{"is_skin": true_or_false, "reason": "one short sentence"}'
        )
        try:
            gate_resp = self.gemini_model.generate_content([img_part, gate_prompt])
            raw = gate_resp.text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            gate_data: dict = json.loads(m.group()) if m else {"is_skin": True}
        except Exception:
            gate_data = {"is_skin": True}

        if not gate_data.get("is_skin", True):
            info = DISEASE_INFO["Other_Non_Skin"]
            return {
                "label": f"{info['emoji']} Not a Skin Image — Please upload a photo of skin",
                "raw_class": "Other_Non_Skin",
                "confidence": 0.0,
                "category": "non_skin",
                "status": "gemini",
                "probabilities": {},
                "disease_info": info,
                "top3": [],
                "notes": gate_data.get("reason", ""),
            }

        # ── Stage 2: Classify the skin condition ─────────────────────────────
        disease_list = "\n".join(
            f"- {name}" for name in DISEASE_INFO if name != "Other_Non_Skin"
        )
        classify_prompt = f"""You are an expert AI dermatology assistant for educational screening.

Analyze the skin in this image and identify the most likely skin condition.
Return ONLY valid JSON (no markdown code block, no extra text):
{{
  "primary_condition": "exact name from list below",
  "confidence": 0.00,
  "top3": [
    {{"condition": "name", "confidence": 0.00}},
    {{"condition": "name", "confidence": 0.00}},
    {{"condition": "name", "confidence": 0.00}}
  ],
  "notes": "1-2 sentence clinical observation about visible features"
}}

VALID CONDITION NAMES (use EXACT spelling, do NOT invent new names):
{disease_list}

RULES:
- confidence must be between 0.0 and 1.0
- top3 confidences should roughly sum to 1.0
- Use "Normal_Skin" if no lesion or condition is visible
- Prefer specific diagnoses over "Melanocytic Nevi" unless clearly a mole
- Always return exactly 3 items in top3
"""
        try:
            cl_resp = self.gemini_model.generate_content([img_part, classify_prompt])
            raw = cl_resp.text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data: dict = json.loads(m.group()) if m else {}
        except Exception:
            data = {}

        primary: str = data.get("primary_condition", "Melanocytic Nevi")
        confidence: float = float(data.get("confidence", 0.6))
        top3_raw: list = data.get("top3", [])
        notes: str = data.get("notes", "")

        # Validate primary against taxonomy
        if primary not in DISEASE_INFO:
            info = _get_disease_info(primary)
            # Remap to closest key
            for k in DISEASE_INFO:
                if k.lower().replace("_", " ") in primary.lower():
                    primary = k
                    break
        info = _get_disease_info(primary)

        category = info.get("category", "lesion")
        emoji = info.get("emoji", "🩺")
        clean_name = primary.replace("_", " ").strip()

        top3 = [
            (item.get("condition", ""), float(item.get("confidence", 0.0)))
            for item in top3_raw[:3]
            if item.get("condition")
        ]
        probs = {cond: conf for cond, conf in top3}

        return {
            "label": f"{emoji} {clean_name}",
            "raw_class": primary,
            "confidence": confidence,
            "category": category,
            "status": "gemini",
            "probabilities": probs,
            "disease_info": info,
            "top3": top3,
            "notes": notes,
        }

    def _predict_ollama(self, image_path: str) -> dict:
        """Use an optional local Ollama vision model as a conservative gate."""
        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")

        prompt = (
            "Classify this image for an educational skin-screening prototype. "
            "Return ONLY JSON with keys is_skin, category, condition, confidence, notes. "
            "category must be exactly one of skin_disease, healthy_skin, document, random_object. "
            "If the image is a page, handwriting, diagram, room, animal, table, fan, or other object, "
            "set is_skin false and category document or random_object. Never guess a disease for a non-skin image."
        )
        payload = json.dumps({
            "model": self.ollama_model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            outer = json.loads(response.read().decode("utf-8"))
        text = outer.get("message", {}).get("content", "{}")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group()) if match else {}

        is_skin = bool(data.get("is_skin", False))
        category = str(data.get("category", "random_object"))
        if not is_skin or category in {"document", "random_object"}:
            info = DISEASE_INFO["Other_Non_Skin"]
            return {"label": f"{info['emoji']} Not a Skin Image — Please upload a photo of skin",
                    "raw_class": "Other_Non_Skin", "confidence": 0.0,
                    "category": "non_skin", "status": "ollama", "probabilities": {},
                    "disease_info": info, "top3": [],
                    "notes": str(data.get("notes", "Ollama rejected this as non-skin."))}

        raw_label = str(data.get("condition", "Normal_Skin"))
        info = _get_disease_info(raw_label)
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        clean = raw_label.replace("_", " ")
        return {"label": f"{info.get('emoji', '🩺')} {clean}", "raw_class": raw_label,
                "confidence": confidence, "category": info.get("category", "lesion"),
                "status": "ollama", "probabilities": {}, "disease_info": info,
                "top3": [], "notes": str(data.get("notes", ""))}

    # ─────────────────────────────────────────────────────────────────────────
    def _run_local_model(self, image_path: str) -> np.ndarray:
        """Run the local model and return a probability array."""
        image = Image.open(image_path).convert("RGB")

        if self.backend == "pytorch" and self.pt_model is not None:
            import torch

            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.pt_model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        else:
            resized = image.resize(self.image_size)
            data = np.asarray(resized, dtype=np.float32)[None, ...]
            det = self.interpreter.get_input_details()[0]
            if det["dtype"] == np.uint8:
                sc, zp = det["quantization"]
                data = (data / sc + zp).astype(np.uint8) if sc else data.astype(np.uint8)
            else:
                data = (data / 127.5) - 1.0
            self.interpreter.set_tensor(det["index"], data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(
                self.interpreter.get_output_details()[0]["index"]
            )[0]
            if np.max(output) > 1.0 or np.min(output) < 0.0:
                exp = np.exp(output - np.max(output))
                probs = exp / np.sum(exp)
            else:
                probs = output

        return probs

    def _predict_local(self, image_path: str) -> dict:
        """Local model prediction with skin gate + entropy-based rejection."""

        # ── Gate 1: HSV skin color check ─────────────────────────────────────
        is_skin_color, skin_ratio = skin_gate_hsv(image_path)
        document_like = looks_like_document(image_path)

        # ── Run model ─────────────────────────────────────────────────────────
        probs = self._run_local_model(image_path)
        index = int(np.argmax(probs))
        confidence = float(probs[index])
        raw_label = self.labels[index] if index < len(self.labels) else f"Class {index}"

        # Top-3 predictions
        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            (self.labels[i] if i < len(self.labels) else f"Class {i}", float(probs[i]))
            for i in top_idx
        ]

        probabilities = {
            (self.labels[i] if i < len(self.labels) else f"Class {i}"): float(probs[i])
            for i in range(len(probs))
        }

        # ── Gate 2: Entropy check ─────────────────────────────────────────────
        n = len(probs)
        max_ent = np.log2(n) if n > 1 else 1.0
        entropy_ratio = _entropy(probs) / max_ent  # 0 = certain, 1 = maximally confused

        norm = raw_label.lower().replace("_", " ")

        # ── Decision logic ────────────────────────────────────────────────────
        # Strong non-skin rejection: if almost no skin-tone pixels are present,
        # never allow the classifier to force a disease label onto an object.
        # The HSV gate is intentionally conservative so darker skin tones are
        # not rejected solely by this heuristic.
        if document_like:
            category = "non_skin"
            info = DISEASE_INFO["Other_Non_Skin"]
            label = f"{info['emoji']} Not a Skin Image — Please upload a photo of skin"

        elif skin_ratio < 0.05:
            category = "non_skin"
            info = DISEASE_INFO["Other_Non_Skin"]
            label = f"{info['emoji']} Not a Skin Image — Please upload a photo of skin"

        # Non-skin: HSV check failed AND model is uncertain
        elif not is_skin_color and confidence < 0.65:
            category = "non_skin"
            info = DISEASE_INFO["Other_Non_Skin"]
            label = f"{info['emoji']} Not a Skin Image — Please upload a photo of skin"

        # Entropy too high: model is guessing
        elif entropy_ratio > 0.88 or confidence < self.min_confidence:
            category = "uncertain"
            info = {
                "description": (
                    "The model could not confidently identify a skin condition in this image. "
                    "Please try a closer, clearer, better-lit photo."
                ),
                "severity": "Try a better-quality skin photo",
                "emoji": "❓",
                "category": "uncertain",
            }
            label = f"❓ Uncertain — Low Confidence ({confidence:.1%}) — Please try a clearer photo"

        elif "non skin" in norm or "other" in norm or "background" in norm:
            category = "non_skin"
            info = DISEASE_INFO["Other_Non_Skin"]
            label = f"{info['emoji']} Not a Skin Image — Please upload a photo of skin"

        elif "normal" in norm or "healthy" in norm:
            category = "normal"
            info = DISEASE_INFO.get("Normal_Skin", {})
            label = f"✅ Normal Healthy Skin — No Lesion Detected"

        else:
            info = _get_disease_info(raw_label)
            category = info.get("category", "lesion")
            emoji = info.get("emoji", "🩺")
            clean = raw_label.replace("_", " ").strip()
            label = f"{emoji} {clean}"

        return {
            "label": label,
            "raw_class": raw_label,
            "confidence": confidence,
            "category": category,
            "status": "model",
            "probabilities": probabilities,
            "disease_info": info,
            "top3": top3,
            "notes": f"Skin color ratio: {skin_ratio:.1%} | Entropy ratio: {entropy_ratio:.2f}",
        }

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> dict:
        """Main predict entry point. Auto-selects Gemini → local → demo."""
        if self.demo_mode:
            has_model = (
                self.model_path.exists() or self.model_path.with_suffix(".pt").exists()
            )
            label = (
                "Demo mode — ML runtime unavailable"
                if has_model
                else "Demo mode — model not installed"
            )
            return {
                "label": label,
                "raw_class": "demo",
                "confidence": 0.0,
                "category": "demo",
                "status": "demo",
                "probabilities": {},
                "disease_info": {},
                "top3": [],
                "notes": "",
            }

        if self.gemini_model is not None:
            return self._predict_gemini(image_path)
        if self.ollama_model:
            try:
                return self._predict_ollama(image_path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
                self.load_error = f"Ollama unavailable: {exc}"
        return self._predict_local(image_path)
