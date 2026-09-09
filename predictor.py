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
from pathlib import Path
from typing import Any, Optional

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
        acne_model_path: Optional[Path] = None,
        acne_labels_path: Optional[Path] = None,
        gate_model_path: Optional[Path] = None,
        gate_labels_path: Optional[Path] = None,
        image_size: tuple[int, int] = (224, 224),
        min_confidence: float = 0.55,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.acne_model_path = Path(acne_model_path) if acne_model_path else self.model_path.with_name("acne_model.tflite")
        self.acne_labels_path = Path(acne_labels_path) if acne_labels_path else self.model_path.with_name("acne_labels.txt")
        self.acne_interpreter = None
        self.acne_labels: list[str] = []
        self.acne_image_size = image_size
        self.acne_preprocess_mode = "minus_one_to_one"
        # Opt in only after an acne model has been trained and independently
        # tested. Keeping this false by default preserves the existing model
        # behavior for users who have not supplied the focused model yet.
        self.acne_focus = str(os.environ.get("ACNE_FOCUS_MODE", "0")).lower() in {
            "1", "true", "yes", "on"
        }
        self.acne_priority = str(os.environ.get("ACNE_PRIORITY_MODE", "1")).lower() in {
            "1", "true", "yes", "on"
        }
        # Keep the former dermoscopic model as a second local expert. The
        # current SCIN model covers common inflammatory/infectious conditions;
        # the legacy model covers the older lesion taxonomy.
        self.legacy_model_path = self.model_path.with_name("skin_model_legacy.tflite")
        self.legacy_labels_path = self.labels_path.with_name("labels_legacy.txt")
        self.gate_model_path = Path(gate_model_path) if gate_model_path else self.model_path.with_name("skin_gate.tflite")
        self.gate_labels_path = Path(gate_labels_path) if gate_labels_path else self.model_path.with_name("skin_gate_labels.txt")
        self.image_size = image_size
        self.legacy_image_size = image_size
        self.min_confidence = min_confidence
        self.interpreter = None
        self.legacy_interpreter = None
        self.gate_interpreter = None
        self.pt_model = None
        self.transform = None
        self.device = None
        self.labels: list[str] = []
        self.legacy_labels: list[str] = []
        self.gate_labels: list[str] = []
        self.load_error: Optional[str] = None
        self.backend: Optional[str] = None
        self.legacy_backend: Optional[str] = None
        self.gemini_model = None
        # The original bundled model expected pixels in [-1, 1].  The current
        # HF baseline has an embedded 1/255 rescaling layer, so it expects the
        # raw 0..255 image range at the TFLite input.  Keep this explicit so a
        # future model swap cannot silently change predictions.
        self.preprocess_mode = "minus_one_to_one"
        metadata_path = self.model_path.with_name("skin_model_metadata.json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.preprocess_mode = str(
                    metadata.get("input_preprocessing", self.preprocess_mode)
                )
            except (OSError, ValueError, TypeError) as exc:
                print(f"Model metadata could not be read: {exc}")

        # Load labels
        if self.labels_path.exists():
            self.labels = [
                x.strip()
                for x in self.labels_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]

        if self.acne_labels_path.exists():
            self.acne_labels = [
                x.strip()
                for x in self.acne_labels_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            acne_metadata_path = self.acne_model_path.with_name("acne_model_metadata.json")
            if acne_metadata_path.exists():
                try:
                    acne_metadata = json.loads(
                        acne_metadata_path.read_text(encoding="utf-8")
                    )
                    self.acne_preprocess_mode = str(
                        acne_metadata.get(
                            "input_preprocessing", self.acne_preprocess_mode
                        )
                    )
                except (OSError, ValueError, TypeError) as exc:
                    print(f"Acne model metadata could not be read: {exc}")

        if self.legacy_labels_path.exists():
            self.legacy_labels = [
                x.strip()
                for x in self.legacy_labels_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]

        if self.gate_labels_path.exists():
            self.gate_labels = [
                x.strip()
                for x in self.gate_labels_path.read_text(encoding="utf-8").splitlines()
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
            self._load_legacy_model()
            self._load_gate_model()
            self._load_acne_model()

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
                output_shape = self.interpreter.get_output_details()[0]["shape"]
                output_count = int(output_shape[-1]) if len(output_shape) else 0
                if output_count and len(self.labels) != output_count:
                    print(
                        "Model/label mismatch: model returns "
                        f"{output_count} classes but labels.txt contains {len(self.labels)}."
                    )
                    if len(self.labels) > output_count:
                        self.labels = self.labels[:output_count]
                    else:
                        self.labels.extend(
                            f"Class {i}" for i in range(len(self.labels), output_count)
                        )
                print(f"TFLite model loaded ({len(self.labels)} classes, backend={self.backend})")
            except Exception as exc:
                self.interpreter = None
                self.load_error = str(exc)
                print(f"TFLite load failed: {exc}")

    def _load_gate_model(self) -> None:
        """Load the optional two-class skin/non-skin TFLite gate."""
        if not self.gate_model_path.exists():
            return

        try:
            try:
                import ai_edge_litert.interpreter as tflite  # type: ignore
                self.gate_interpreter = tflite.Interpreter(model_path=str(self.gate_model_path))
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite  # type: ignore
                    self.gate_interpreter = tflite.Interpreter(model_path=str(self.gate_model_path))
                except ImportError:
                    import tensorflow as tf  # type: ignore
                    self.gate_interpreter = tf.lite.Interpreter(model_path=str(self.gate_model_path))

            self.gate_interpreter.allocate_tensors()
            print(f"Skin gate loaded ({len(self.gate_labels)} classes): {self.gate_model_path.name}")
        except Exception as exc:
            self.gate_interpreter = None
            print(f"Skin gate load failed: {exc}")

    def _load_acne_model(self) -> None:
        """Load the optional focused two-class acne model."""
        if not self.acne_model_path.exists() or len(self.acne_labels) != 2:
            return

        try:
            try:
                import ai_edge_litert.interpreter as tflite  # type: ignore
                self.acne_interpreter = tflite.Interpreter(
                    model_path=str(self.acne_model_path)
                )
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite  # type: ignore
                    self.acne_interpreter = tflite.Interpreter(
                        model_path=str(self.acne_model_path)
                    )
                except ImportError:
                    import tensorflow as tf  # type: ignore
                    self.acne_interpreter = tf.lite.Interpreter(
                        model_path=str(self.acne_model_path)
                    )

            self.acne_interpreter.allocate_tensors()
            input_shape = self.acne_interpreter.get_input_details()[0]["shape"]
            if len(input_shape) == 4:
                self.acne_image_size = (int(input_shape[2]), int(input_shape[1]))
            output_shape = self.acne_interpreter.get_output_details()[0]["shape"]
            output_count = int(output_shape[-1]) if len(output_shape) else 0
            if output_count != 2:
                self.acne_interpreter = None
                print(
                    "Acne model ignored: expected two output classes, "
                    f"found {output_count}."
                )
                return
            print(f"Acne model loaded (2 classes): {self.acne_model_path.name}")
        except Exception as exc:
            self.acne_interpreter = None
            print(f"Acne model load failed: {exc}")

    def _load_legacy_model(self) -> None:
        """Load the former HAM10000 lesion model when it is bundled locally."""
        if not self.legacy_model_path.exists() or not self.legacy_labels:
            return

        try:
            try:
                import ai_edge_litert.interpreter as tflite  # type: ignore
                self.legacy_interpreter = tflite.Interpreter(
                    model_path=str(self.legacy_model_path)
                )
                self.legacy_backend = "litert"
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite  # type: ignore
                    self.legacy_interpreter = tflite.Interpreter(
                        model_path=str(self.legacy_model_path)
                    )
                    self.legacy_backend = "tflite_runtime"
                except ImportError:
                    import tensorflow as tf  # type: ignore
                    self.legacy_interpreter = tf.lite.Interpreter(
                        model_path=str(self.legacy_model_path)
                    )
                    self.legacy_backend = "tensorflow"

            self.legacy_interpreter.allocate_tensors()
            legacy_shape = self.legacy_interpreter.get_input_details()[0]["shape"]
            if len(legacy_shape) == 4:
                self.legacy_image_size = (int(legacy_shape[2]), int(legacy_shape[1]))
            output_shape = self.legacy_interpreter.get_output_details()[0]["shape"]
            output_count = int(output_shape[-1]) if len(output_shape) else 0
            if output_count and len(self.legacy_labels) != output_count:
                print(
                    "Legacy model/label mismatch: model returns "
                    f"{output_count} classes but labels_legacy.txt contains "
                    f"{len(self.legacy_labels)}."
                )
                self.legacy_labels = self.legacy_labels[:output_count]
            print(
                f"Legacy TFLite model loaded ({len(self.legacy_labels)} classes, "
                f"backend={self.legacy_backend})"
            )
        except Exception as exc:
            self.legacy_interpreter = None
            self.legacy_backend = None
            print(f"Legacy TFLite load failed: {exc}")

    def reload_models(self) -> None:
        """Hot-reload local models from disk after self-training/fine-tuning."""
        if self.labels_path.exists():
            self.labels = [
                x.strip()
                for x in self.labels_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
        self.pt_model = None
        self.interpreter = None
        self._load_local_model()
        self._load_legacy_model()
        self._load_gate_model()
        self._load_acne_model()
        print(f"SkinPredictor hot-reloaded: backend={self.backend}, classes={len(self.labels)}")

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def demo_mode(self) -> bool:
        return (
            self.interpreter is None
            and self.legacy_interpreter is None
            and self.acne_interpreter is None
            and self.pt_model is None
            and self.gemini_model is None
        )

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

    # ─────────────────────────────────────────────────────────────────────────
    def _run_local_model(self, image_path: str) -> np.ndarray:
        """Run the local model and return a probability array."""
        if self.backend == "pytorch" and self.pt_model is not None:
            image = Image.open(image_path).convert("RGB")
            import torch

            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.pt_model(tensor)
                return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return self._run_tflite_model(
            image_path, self.interpreter, self.image_size, self.preprocess_mode
        )

    def _run_legacy_model(self, image_path: str) -> np.ndarray:
        """Run the optional legacy dermoscopic model."""
        return self._run_tflite_model(
            image_path,
            self.legacy_interpreter,
            self.legacy_image_size,
            "raw_0_255",
        )

    def _run_acne_model(self, image_path: str) -> np.ndarray:
        """Run the optional focused acne model."""
        return self._run_tflite_model(
            image_path,
            self.acne_interpreter,
            self.acne_image_size,
            self.acne_preprocess_mode,
        )

    @staticmethod
    def _run_tflite_model(
        image_path: str,
        interpreter: Any,
        image_size: tuple[int, int],
        preprocess_mode: str,
    ) -> np.ndarray:
        """Run one float/quantized TFLite model with explicit preprocessing."""
        if interpreter is None:
            raise RuntimeError("TFLite interpreter is not available")

        image = Image.open(image_path).convert("RGB")
        resized = image.resize(image_size)
        data = np.asarray(resized, dtype=np.float32)[None, ...]
        det = interpreter.get_input_details()[0]
        if det["dtype"] == np.uint8:
            sc, zp = det["quantization"]
            data = (data / sc + zp).astype(np.uint8) if sc else data.astype(np.uint8)
        elif preprocess_mode == "zero_one":
            data = data / 255.0
        elif preprocess_mode == "raw_0_255":
            # The SCIN and legacy exports contain their own preprocessing.
            pass
        else:
            data = (data / 127.5) - 1.0
        interpreter.set_tensor(det["index"], data)
        interpreter.invoke()
        output = interpreter.get_tensor(
            interpreter.get_output_details()[0]["index"]
        )[0]
        if np.max(output) > 1.0 or np.min(output) < 0.0:
            exp = np.exp(output - np.max(output))
            return exp / np.sum(exp)
        return output

    def _run_skin_gate(self, image_path: str) -> Optional[tuple[bool, float]]:
        """Return (is_skin, confidence) from the optional trained gate."""
        if self.gate_interpreter is None:
            return None

        image = Image.open(image_path).convert("RGB")
        details = self.gate_interpreter.get_input_details()[0]
        shape = details["shape"]
        gate_size = (int(shape[2]), int(shape[1])) if len(shape) == 4 else (224, 224)
        data = np.asarray(image.resize(gate_size), dtype=np.float32)[None, ...]
        if details["dtype"] == np.uint8:
            scale, zero_point = details["quantization"]
            data = (data / scale + zero_point).astype(np.uint8) if scale else data.astype(np.uint8)
        # The trained gate contains its own Keras Rescaling layer, so float
        # inputs must remain in the original 0..255 image range. Normalizing
        # here would apply the transform twice and invert gate predictions.

        self.gate_interpreter.set_tensor(details["index"], data)
        self.gate_interpreter.invoke()
        output = self.gate_interpreter.get_tensor(
            self.gate_interpreter.get_output_details()[0]["index"]
        )[0]
        if np.max(output) > 1.0 or np.min(output) < 0.0:
            exp = np.exp(output - np.max(output))
            output = exp / np.sum(exp)

        labels = [label.lower().replace("_", " ") for label in self.gate_labels]
        skin_index = next((i for i, label in enumerate(labels) if "skin" in label and "non" not in label), None)
        non_skin_index = next((i for i, label in enumerate(labels) if "non" in label), None)
        if skin_index is None or non_skin_index is None:
            return None

        predicted_index = int(np.argmax(output))
        confidence = float(output[predicted_index])
        return predicted_index == skin_index, confidence

    def _predict_acne_local(self, image_path: str) -> dict:
        """Focused acne screening after the existing skin/non-skin gate."""
        trained_gate = self._run_skin_gate(image_path)
        is_skin_color, skin_ratio = skin_gate_hsv(image_path)
        document_like = looks_like_document(image_path)
        probs = np.asarray(self._run_acne_model(image_path), dtype=np.float32).reshape(-1)
        probs = probs / max(float(np.sum(probs)), 1e-8)
        acne_index = next(
            (i for i, name in enumerate(self.acne_labels)
             if "acne" in name.lower() and "not" not in name.lower()),
            None,
        )
        if acne_index is None or len(probs) != 2:
            raise RuntimeError("Focused acne model labels must contain Acne and Not_Acne")

        index = int(np.argmax(probs))
        confidence = float(probs[index])
        acne_confidence = float(probs[acne_index])
        predicted_acne = index == acne_index
        top_idx = np.argsort(probs)[::-1]
        top3 = [
            (self.acne_labels[i], float(probs[i]))
            for i in top_idx[:2]
        ]
        probabilities = {
            self.acne_labels[i]: float(probs[i]) for i in range(len(probs))
        }

        strong_hsv_skin = is_skin_color and skin_ratio >= 0.50
        gate_rejects = trained_gate is not None and (
            (not trained_gate[0] and trained_gate[1] >= 0.65 and not strong_hsv_skin)
            or (trained_gate[0] and trained_gate[1] < 0.65 and not strong_hsv_skin)
        )

        if gate_rejects or document_like or skin_ratio < 0.05:
            info = DISEASE_INFO["Other_Non_Skin"]
            category = "non_skin"
            label = f"{info['emoji']} Not a Skin Image — Please upload a photo of skin"
        elif not is_skin_color and confidence < 0.65:
            info = DISEASE_INFO["Other_Non_Skin"]
            category = "non_skin"
            label = f"{info['emoji']} Not a Skin Image — Please upload a photo of skin"
        elif confidence < self.min_confidence:
            info = {
                "description": "The focused acne model is not confident enough for a screening result.",
                "severity": "Try a closer, clearer, better-lit photo",
                "emoji": "❓",
                "category": "uncertain",
            }
            category = "uncertain"
            label = f"❓ Uncertain — Low Confidence ({confidence:.1%}) — Please try a clearer photo"
        elif predicted_acne:
            info = DISEASE_INFO["Acne Vulgaris"]
            category = "lesion"
            label = f"{info['emoji']} Possible Acne Vulgaris — Screening Result Only"
        else:
            info = {
                "description": "The focused model did not find a strong acne pattern in this image.",
                "severity": "No acne pattern detected; other conditions are not ruled out",
                "emoji": "✅",
                "category": "non_acne",
            }
            category = "non_acne"
            label = "✅ No Strong Acne Pattern Detected — Other Conditions Not Ruled Out"

        return {
            "label": label,
            "raw_class": self.acne_labels[index],
            "confidence": confidence,
            "acne_confidence": acne_confidence,
            "category": category,
            "status": "model",
            "probabilities": probabilities,
            "disease_info": info,
            "top3": top3,
            "model_used": "focused acne model",
            "notes": (
                f"Skin color ratio: {skin_ratio:.1%} | Acne probability: {acne_confidence:.1%}"
                + (
                    f" | Trained gate: {'skin' if trained_gate[0] else 'non-skin'} "
                    f"({trained_gate[1]:.1%})"
                    if trained_gate is not None else ""
                )
            ),
        }

    def _predict_local(self, image_path: str) -> dict:
        """Local model prediction with skin gate + entropy-based rejection."""

        if self.acne_interpreter is not None and (self.acne_focus or self.acne_priority):
            acne_result = self._predict_acne_local(image_path)
            # Acne gets first priority. If the focused model says that the
            # image is not acne, continue to the broader disease experts.
            if self.acne_focus or acne_result["category"] in {"lesion", "non_skin"}:
                return acne_result

        # ── Gate 1: trained skin/non-skin model ──────────────────────────────
        trained_gate = self._run_skin_gate(image_path)

        # ── Gate 2: HSV skin color and document checks ───────────────────────
        is_skin_color, skin_ratio = skin_gate_hsv(image_path)
        document_like = looks_like_document(image_path)

        # ── Run model ─────────────────────────────────────────────────────────
        probs = self._run_local_model(image_path)
        active_labels = self.labels
        selected_model = "SCIN"
        index = int(np.argmax(probs))
        confidence = float(probs[index])

        # Use the SCIN expert for its newer common-disease classes when it has
        # a usable signal. Fall back to the older dermoscopic expert only when
        # the SCIN expert is weak and the legacy expert is clearly confident.
        # This keeps both taxonomies available without pretending their scores
        # are directly comparable.
        if self.legacy_interpreter is not None and self.legacy_labels:
            legacy_probs = self._run_legacy_model(image_path)
            legacy_confidence = float(np.max(legacy_probs))
            if confidence < 0.65 and legacy_confidence >= 0.75:
                probs = legacy_probs
                active_labels = self.legacy_labels
                selected_model = "legacy HAM10000"

        index = int(np.argmax(probs))
        confidence = float(probs[index])
        raw_label = active_labels[index] if index < len(active_labels) else f"Class {index}"

        # Top-3 predictions
        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            (active_labels[i] if i < len(active_labels) else f"Class {i}", float(probs[i]))
            for i in top_idx
        ]

        probabilities = {
            (active_labels[i] if i < len(active_labels) else f"Class {i}"): float(probs[i])
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
        # Treat the trained gate as a strong signal, but allow a clear
        # close-up skin region to recover from a gate false-negative.  This is
        # needed for photos such as a bald scalp/lesion where hair, glasses,
        # shadows, and the surrounding background can dominate the gate.  A
        # high HSV skin ratio (50%+) is deliberately required so a small
        # skin-coloured object in a room does not bypass the gate.
        strong_hsv_skin = is_skin_color and skin_ratio >= 0.50
        gate_rejects = trained_gate is not None and (
            (
                not trained_gate[0]
                and trained_gate[1] >= 0.65
                and not strong_hsv_skin
            )
            or (
                trained_gate[0]
                and trained_gate[1] < 0.65
                and not strong_hsv_skin
            )
        )

        if gate_rejects:
            category = "non_skin"
            info = DISEASE_INFO["Other_Non_Skin"]
            label = f"{info['emoji']} Not a Confirmed Skin Image — Please upload a clear photo of skin"

        elif document_like:
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
            "model_used": selected_model,
            "notes": (
                f"Skin color ratio: {skin_ratio:.1%} | Entropy ratio: {entropy_ratio:.2f}"
                + (
                    f" | Trained gate: {'skin' if trained_gate[0] else 'non-skin'} "
                    f"({trained_gate[1]:.1%})"
                    if trained_gate is not None
                    else ""
                )
                + f" | Disease model: {selected_model}"
            ),
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
        return self._predict_local(image_path)
