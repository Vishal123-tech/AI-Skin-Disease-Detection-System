"""self_train.py — Continuous Self-Training Engine from User Feedback
───────────────────────────────────────────────────────────────────────
Ingests confirmed feedback from `feedback/feedback.jsonl`, pairs it with
baseline reference data (experience replay) to prevent catastrophic forgetting,
fine-tunes the MobileNetV2 classifier head with PyTorch, backs up the previous
model, and outputs an updated model artifact.
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
    """Execute continuous learning fine-tuning."""
    usable = load_usable_feedback(include_already_learned=True)
    if len(usable) < min_samples:
        return {
            "success": False,
            "message": f"Insufficient confirmed feedback. Need at least {min_samples} samples, found {len(usable)}.",
            "feedback_count": len(usable),
        }

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms
    from torch.utils.data import Dataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine class set
    existing_labels = []
    if LABELS_PATH.exists():
        existing_labels = [
            l.strip() for l in LABELS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
    active_labels = list(dict.fromkeys(existing_labels + CANONICAL_LABELS))

    # Add any new labels from feedback
    for item in usable:
        if item["label"] not in active_labels:
            active_labels.append(item["label"])

    label_to_idx = {name: idx for idx, name in enumerate(active_labels)}

    # Build dataset pairs: (path, class_idx)
    data_pairs: list[tuple[str, int]] = []
    feedback_ids_trained: list[str] = []

    for item in usable:
        data_pairs.append((item["image_path"], label_to_idx[item["label"]]))
        feedback_ids_trained.append(item["id"])

    # Blend experience replay samples
    replay_samples = collect_replay_samples(active_labels, max_per_class=10)
    for path, c_name in replay_samples:
        if c_name in label_to_idx and Path(path).exists():
            data_pairs.append((path, label_to_idx[c_name]))

    if not data_pairs:
        return {"success": False, "message": "No valid training images found on disk."}

    print(f"Self-training with {len(usable)} feedback images + {len(replay_samples)} replay images.")
    print(f"Classes ({len(active_labels)}): {active_labels}")

    # Dataset definition
    class SkinDataset(Dataset):
        def __init__(self, pairs: list[tuple[str, int]], transform=None):
            self.pairs = pairs
            self.transform = transform

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            path, label_idx = self.pairs[idx]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, label_idx

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Train / Val Split (80 / 20)
    np.random.seed(42)
    indices = np.random.permutation(len(data_pairs))
    split_idx = max(1, int(len(data_pairs) * 0.8))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:] if split_idx < len(data_pairs) else indices[:1]

    train_set = SkinDataset([data_pairs[i] for i in train_indices], transform=train_transform)
    val_set = SkinDataset([data_pairs[i] for i in val_indices], transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=min(batch_size, len(train_set)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=min(batch_size, len(val_set)), shuffle=False)

    # Initialize MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze earlier layers, fine-tune head
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, len(active_labels)),
    )

    # If existing PyTorch model exists, load compatible weights
    if MODEL_PT_PATH.exists():
        try:
            chk = torch.load(MODEL_PT_PATH, map_location=device)
            # Only load matching classifier shape if classes count matches
            if chk.get("classifier.1.1.weight") is not None and chk["classifier.1.1.weight"].shape[0] == len(active_labels):
                model.load_state_dict(chk, strict=False)
                print("Loaded warm-start weights from existing skin_model.pt")
        except Exception as exc:
            print(f"Could not load previous checkpoint: {exc}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier[1].parameters(), lr=learning_rate, weight_decay=1e-3)

    print(f"Starting self-training for {epochs} epochs on device: {device}...")
    best_loss = float("inf")
    final_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == targets).item()
            total += inputs.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = corrects / max(total, 1)

        # Validation
        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for v_in, v_tgt in val_loader:
                v_in = v_in.to(device)
                v_tgt = v_tgt.to(device)
                v_out = model(v_in)
                _, v_pred = torch.max(v_out, 1)
                val_corrects += torch.sum(v_pred == v_tgt).item()
                val_total += v_in.size(0)

        val_acc = val_corrects / max(val_total, 1)
        final_acc = val_acc
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.1%}, Val Acc: {val_acc:.1%}")

    # Backup existing models
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = None
    if MODEL_PT_PATH.exists():
        backup_file = BACKUP_DIR / f"skin_model_{stamp}.pt"
        shutil.copy2(MODEL_PT_PATH, backup_file)
    tflite_path = MODELS_DIR / "skin_model.tflite"
    if tflite_path.exists():
        shutil.copy2(tflite_path, BACKUP_DIR / f"skin_model_{stamp}.tflite")

    # Save new PyTorch model
    torch.save(model.state_dict(), MODEL_PT_PATH)
    LABELS_PATH.write_text("\n".join(active_labels), encoding="utf-8")

    # Update metadata
    meta = {
        "source": "Continuous Self-Training from User Feedback",
        "architecture": "MobileNetV2 fine-tuned",
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "classes": active_labels,
        "feedback_samples_used": len(usable),
        "replay_samples_used": len(replay_samples),
        "validation_accuracy": round(final_acc, 4),
        "backup_path": str(backup_file) if backup_file else None,
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Record learned sample IDs
    mark_learned_ids(feedback_ids_trained)

    print(f"Self-training complete! Saved to {MODEL_PT_PATH}")
    return {
        "success": True,
        "message": f"Successfully fine-tuned model on {len(usable)} feedback images + {len(replay_samples)} replay samples.",
        "classes": active_labels,
        "feedback_count": len(usable),
        "replay_count": len(replay_samples),
        "val_accuracy": final_acc,
        "backup_path": str(backup_file) if backup_file else None,
        "timestamp": meta["last_trained"],
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
