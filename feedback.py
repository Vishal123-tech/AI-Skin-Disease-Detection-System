"""Safe feedback capture for supervised model improvement.

Feedback is stored locally for review. It never changes a deployed model at
prediction time, and it intentionally stores no patient name or identity.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEEDBACK_DIR = ROOT / "feedback"
FEEDBACK_IMAGE_DIR = FEEDBACK_DIR / "images"
FEEDBACK_LOG = FEEDBACK_DIR / "feedback.jsonl"


def available_feedback_labels() -> list[str]:
    """Return human-readable labels from the bundled models."""

    labels: list[str] = []
    for filename in (
        "labels.txt",
        "labels_legacy.txt",
        "acne_labels.txt",
        "skin_gate_labels.txt",
    ):
        path = ROOT / "models" / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            label = line.strip().replace("_", " ")
            if label and label.upper() not in {"SKIN", "NON SKIN"} and label not in labels:
                labels.append(label)

    for extra in ("Normal Skin", "Not Skin / Object", "Other / Not Sure"):
        if extra not in labels:
            labels.append(extra)
    return labels


def _safe_text(value: object, max_length: int) -> str:
    return " ".join(str(value or "").split())[:max_length]


def _copy_feedback_image(image_path: Path, feedback_id: str) -> tuple[str, str, Path]:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError("The analyzed image is no longer available.")

    suffix = image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    FEEDBACK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    target = FEEDBACK_IMAGE_DIR / f"{feedback_id}{suffix}"
    shutil.copy2(image_path, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return str(target.relative_to(ROOT)), digest, target


def record_feedback(
    image_path: str | Path,
    result: dict,
    rating: str,
    correct_label: str = "",
    comment: str = "",
    consent: bool = False,
) -> str:
    """Record one feedback item for later human review.

    ``rating`` must be ``Correct``, ``Wrong``, or ``Not sure``. A wrong
    prediction requires a user-selected correction label. This is intentionally
    append-only and does not retrain or replace any model.
    """

    rating = _safe_text(rating, 20)
    if rating not in {"Correct", "Wrong", "Not sure"}:
        raise ValueError("Choose Correct, Wrong, or Not sure.")
    correct_label = _safe_text(correct_label, 100)
    if rating == "Wrong" and not correct_label:
        raise ValueError("Choose the correct condition before submitting.")
    if not consent:
        raise ValueError("Confirm consent before storing the image for model improvement.")

    feedback_id = uuid.uuid4().hex
    image_file, image_sha256, stored_image_path = _copy_feedback_image(Path(image_path), feedback_id)
    now = datetime.now(timezone.utc).isoformat()
    confidence = result.get("confidence")
    record = {
        "id": feedback_id,
        "created_at": now,
        "rating": rating,
        "predicted_label": _safe_text(result.get("raw_class") or result.get("label"), 120),
        "predicted_category": _safe_text(result.get("category"), 40),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
        "model_used": _safe_text(result.get("model_used"), 100),
        "correct_label": correct_label if rating == "Wrong" else "",
        "comment": _safe_text(comment, 500),
        "image_file": image_file,
        "image_sha256": image_sha256,
    }
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return (
        "✅ Feedback saved locally for continuous learning. "
        f"Feedback ID: `{feedback_id[:12]}`"
    )


def trigger_self_training(epochs: int = 5, min_samples: int = 1) -> dict:
    """Trigger the continuous self-training pipeline on saved feedback."""
    from self_train import train_on_feedback
    return train_on_feedback(epochs=epochs, min_samples=min_samples)


def get_feedback_summary() -> dict:
    """Return summary of feedback collection and training status."""
    from self_train import get_feedback_stats
    return get_feedback_stats()


if __name__ == "__main__":
    print(f"Feedback log: {FEEDBACK_LOG}")
    print(f"Available labels: {', '.join(available_feedback_labels())}")
    print(f"Summary: {get_feedback_summary()}")
