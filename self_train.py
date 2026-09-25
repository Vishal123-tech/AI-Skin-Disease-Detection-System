"""Feedback statistics and a fail-closed guard against unsafe model updates.

One-click training is suspended after the one-image training regression.
Use the supervised, separate-candidate workflow in docs/FEEDBACK_LOOP.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
BACKUP_DIR = MODELS_DIR / "backups"
FEEDBACK_DIR = ROOT / "feedback"
FEEDBACK_LOG = FEEDBACK_DIR / "feedback.jsonl"
FEEDBACK_IMG_DIR = FEEDBACK_DIR / "images"
LEARNED_TRACKER = FEEDBACK_DIR / "learned_ids.json"
LABELS_PATH = MODELS_DIR / "labels.txt"
MODEL_PT_PATH = MODELS_DIR / "skin_model.pt"
METADATA_PATH = MODELS_DIR / "skin_model_metadata.json"

# Canonical disease taxonomy
CANONICAL_LABELS = [
    "Acne Vulgaris",
    "Eczema",
    "Fungal Infection",
    "Psoriasis",
]


def _normalize_label(raw: str) -> str:
    """Normalize user-entered or model labels to canonical names."""
    clean = raw.strip().replace("_", " ").lower()
    for canonical in CANONICAL_LABELS:
        if canonical.lower() == clean or clean in canonical.lower() or canonical.lower() in clean:
            return canonical
    if "acne" in clean:
        return "Acne Vulgaris"
    if "eczema" in clean or "atopic" in clean:
        return "Eczema"
    if "fungal" in clean or "tinea" in clean or "ringworm" in clean:
        return "Fungal Infection"
    if "psoriasis" in clean:
        return "Psoriasis"
    # Fallback to title-cased clean label
    return " ".join(word.capitalize() for word in clean.split())


def get_learned_ids() -> set[str]:
    """Return IDs of feedback samples already used in past training runs."""
    if not LEARNED_TRACKER.exists():
        return set()
    try:
        data = json.loads(LEARNED_TRACKER.read_text(encoding="utf-8"))
        return set(data.get("learned_ids", []))
    except Exception:
        return set()


def mark_learned_ids(new_ids: list[str]) -> None:
    """Record newly trained feedback IDs."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    all_ids = list(get_learned_ids().union(new_ids))
    LEARNED_TRACKER.write_text(
        json.dumps({"learned_ids": all_ids, "last_updated": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def load_usable_feedback(include_already_learned: bool = True) -> list[dict]:
    """Parse feedback log and return verified (image_path, target_label) records."""
    if not FEEDBACK_LOG.exists():
        return []

    learned_ids = get_learned_ids() if not include_already_learned else set()
    records: list[dict] = []

    for line in FEEDBACK_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        item_id = item.get("id", "")
        if item_id in learned_ids:
            continue

        rating = item.get("rating", "")
        if rating not in {"Correct", "Wrong"}:
            continue

        # Target condition determination
        if rating == "Correct":
            target = item.get("predicted_label") or item.get("label") or ""
        else:
            target = item.get("correct_label") or ""

        target = _normalize_label(target)
        if not target or "not skin" in target.lower():
            continue

        img_rel = item.get("image_file", "")
        img_path = ROOT / img_rel if img_rel else None
        if not img_path or not img_path.exists():
            continue

        records.append({
            "id": item_id,
            "image_path": str(img_path),
            "label": target,
            "rating": rating,
            "created_at": item.get("created_at", ""),
        })

    return records


def collect_replay_samples(target_labels: list[str], max_per_class: int = 15) -> list[tuple[str, str]]:
    """Gather existing baseline images to prevent catastrophic forgetting."""
    replay: list[tuple[str, str]] = []

    # Check data/skin directory
    skin_dir = ROOT / "data" / "skin"
    if skin_dir.exists():
        for split in ("train", "val", "test"):
            split_dir = skin_dir / split
            if not split_dir.exists():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                c_name = _normalize_label(class_dir.name)
                if c_name not in target_labels:
                    continue
                count = 0
                for img_file in class_dir.glob("*"):
                    if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        replay.append((str(img_file), c_name))
                        count += 1
                        if count >= max_per_class:
                            break

    # Check uploads directory for previous historical samples
    uploads_dir = ROOT / "uploads"
    if uploads_dir.exists() and len(replay) < len(target_labels) * 5:
        for img_file in uploads_dir.glob("*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lower = img_file.name.lower()
                for c_name in target_labels:
                    key = c_name.lower().split()[0]
                    if key in lower:
                        replay.append((str(img_file), c_name))
                        break

    return replay


def train_on_feedback(
    epochs: int = 5,
    min_samples: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
) -> dict:
    """Keep feedback collection separate from active-model deployment.

    The previous one-click trainer could initialize a new classifier on one
    user-labelled example, validate on that same example, and overwrite the
    active checkpoint. Do not reactivate it by merely increasing min_samples.
    Use reviewed labels, original training data, patient-disjoint validation
    and a separately evaluated candidate via the offline training workflow.
    """
    return {
        "success": False,
        "requires_review": True,
        "feedback_count": len(load_usable_feedback()),
        "message": (
            "Automatic feedback training and activation are paused after a model "
            "regression. Feedback is still saved. Review the labels, train a "
            "separate candidate with the original dataset, and evaluate every "
            "class on independent patients before replacing the active model. "
            "See docs/FEEDBACK_LOOP.md."
        ),
    }


def get_feedback_stats() -> dict:
    """Return summary statistics of collected feedback."""
    records = load_usable_feedback(include_already_learned=True)
    learned = get_learned_ids()
    correct_count = sum(1 for r in records if r["rating"] == "Correct")
    wrong_count = sum(1 for r in records if r["rating"] == "Wrong")
    unlearned_count = sum(1 for r in records if r["id"] not in learned)

    last_trained = "Never"
    if METADATA_PATH.exists():
        try:
            m = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            last_trained = m.get("last_trained", "Never")
        except Exception:
            pass

    return {
        "total_feedback": len(records),
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "unlearned_count": unlearned_count,
        "last_trained": last_trained,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Training Model Engine from Feedback")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum feedback samples required")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--stats", action="store_true", help="Print feedback statistics and exit")
    args = parser.parse_args()

    if args.stats:
        stats = get_feedback_stats()
        print(json.dumps(stats, indent=2))
        return

    result = train_on_feedback(epochs=args.epochs, min_samples=args.min_samples, batch_size=args.batch_size)
    print("\nResult:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
