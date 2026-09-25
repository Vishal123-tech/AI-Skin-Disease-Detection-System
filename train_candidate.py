"""Offline, case-separated SCIN candidate training. Never edits deployed models.

Uses cached ImageNet MobileNetV2 features, class/case-balanced sampling and
internal validation for checkpoint selection. Existing val/ is a final audit,
not a new independent test set. See output metadata for all limitations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
LABELS = ["Acne Vulgaris", "Eczema", "Fungal Infection", "Psoriasis"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(root, split):
    rows = []
    for label in LABELS:
        for path in sorted((root / split / label).glob("*.jpg")):
            match = re.fullmatch(r"(-?\d+)_image_[123]_path", path.stem)
            if not match:
                raise ValueError(f"Missing SCIN case identifier: {path.name}")
            with Image.open(path) as image:
                image.verify()
            rows.append({"path": str(path.resolve()), "case": match[1],
                         "label": label, "target": LABELS.index(label), "sha256": digest(path)})
    if set(r["label"] for r in rows) != set(LABELS):
        raise ValueError(f"Every class must be represented in {split}")
    return rows


def split_training(rows, seed=42):
    case_labels = defaultdict(set)
    for row in rows:
        case_labels[row["case"]].add(row["label"])
    if any(len(labels) != 1 for labels in case_labels.values()):
        raise ValueError("Conflicting case labels require review")
    rng = random.Random(seed)
    validation_cases = set()
    for label in LABELS:
        cases = sorted(case for case, labels in case_labels.items() if label in labels)
        if len(cases) < 5:
            raise ValueError(f"Insufficient independent cases for {label}")
        rng.shuffle(cases)
        validation_cases.update(cases[:max(2, round(len(cases) * 0.2))])
    return ([r for r in rows if r["case"] not in validation_cases],
            [r for r in rows if r["case"] in validation_cases])


def check_separation(*splits):
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            for key in ("case", "sha256"):
                if {r[key] for r in left} & {r[key] for r in right}:
                    raise ValueError(f"Cross-split leakage detected by {key}")


def case_weights(rows):
    cases = defaultdict(set)
    images_per_case = Counter(r["case"] for r in rows)
    for row in rows:
        cases[row["label"]].add(row["case"])
    return torch.tensor([1 / (len(cases[r["label"]]) * images_per_case[r["case"]])
                         for r in rows], dtype=torch.double)


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            return self.transform(image.convert("RGB")), row["target"]


def metrics(targets, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for actual, predicted in zip(targets, predictions):
        cm[int(actual), int(predicted)] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        precision = float(cm[i, i] / max(cm[:, i].sum(), 1))
        recall = float(cm[i, i] / max(cm[i].sum(), 1))
        per_class[label] = {"images": int(cm[i].sum()), "precision": precision,
                            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    return {"images": len(targets), "accuracy": float(np.mean(predictions == targets)),
            "macro_f1": float(np.mean([r["f1"] for r in per_class.values()])),
            "macro_recall": float(np.mean([r["recall"] for r in per_class.values()])),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    targets, probabilities = [], []
    for inputs, labels in loader:
        probabilities.extend(model(inputs).softmax(dim=1).numpy())
        targets.extend(labels.tolist())
    return metrics(np.array(targets), np.array(probabilities))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True, help="New folder name under work/candidates")
    parser.add_argument("--head-epochs", type=int, default=25)
    parser.add_argument("--fine-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise ValueError("Use a simple unique candidate name")
    if args.head_epochs < 1 or args.fine_epochs < 0:
        raise ValueError("Invalid epoch count")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    protected = {str(p): digest(p) for p in (ROOT / "models").glob("*") if p.is_file()}
    train, validation = split_training(scan(args.data, "train"), args.seed)
    audit = scan(args.data, "val")
    check_separation(train, validation, audit)
    pretrained = Path(torch.hub.get_dir()) / "checkpoints/mobilenet_v2-7ebf99e0.pth"
    if not pretrained.exists() or not digest(pretrained).startswith("7ebf99e0"):
        raise ValueError("Official cached ImageNet weights missing or checksum mismatch; no automatic download")
    out = ROOT / "work/candidates" / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / "split_manifest.json").write_text(json.dumps({"train": train, "validation": validation,
                                                      "historical_audit": audit}, indent=2))
    counts = {name: dict(Counter(r["label"] for r in rows)) for name, rows in
              (("train", train), ("validation", validation), ("historical_audit", audit))}
    print(json.dumps(counts, indent=2), flush=True)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    evaluation = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                      transforms.ToTensor(), normalize])
    augmentation = transforms.Compose([transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), normalize])
    model = models.mobilenet_v2(weights=None)
    model.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
    model.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(LABELS)))
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    validation_loader = DataLoader(Images(validation, evaluation), batch_size=16)
    audit_loader = DataLoader(Images(audit, evaluation), batch_size=16)
    # Cache three training-only augmented views; validation never enters sampling.
    model.eval()
    features, targets = [], []
    print("Extracting cached ImageNet features on CPU...", flush=True)
    with torch.inference_mode():
        for view in range(3):
            for inputs, labels in DataLoader(Images(train, evaluation if view == 0 else augmentation), batch_size=16):
                features.append(model.features(inputs).mean((2, 3)))
                targets.append(labels)
    # Ordinary tensors (not inference tensors) are required for head gradients.
    feature_data = TensorDataset(torch.cat(features).clone(), torch.cat(targets).clone())
    weights = case_weights(train)
    loader = DataLoader(feature_data, batch_size=32,
                        sampler=WeightedRandomSampler(weights.repeat(3), len(feature_data), replacement=True))
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_score, best_epoch = -1, None
    history = []
    started = time.monotonic()

    def checkpoint(stage, epoch, loss):
        nonlocal best_score, best_epoch
        result = evaluate(model, validation_loader)
        row = {"stage": stage, "epoch": epoch, "loss": loss, "validation": result}
        history.append(row)
        if result["macro_f1"] > best_score:
            best_score, best_epoch = result["macro_f1"], f"{stage}-{epoch}"
            torch.save(model.state_dict(), out / "skin_model.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"{stage} {epoch}: loss={loss:.4f}, validation macro-F1={result['macro_f1']:.3f}, recall={result['macro_recall']:.3f}", flush=True)

    for epoch in range(1, args.head_epochs + 1):
        model.classifier.train()
        total = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model.classifier(inputs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        checkpoint("head", epoch, total / len(feature_data))

    model.load_state_dict(torch.load(out / "skin_model.pt", weights_only=True))
    for parameter in model.features[-3:].parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.features[-3:].parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}], weight_decay=1e-3)
    loader = DataLoader(Images(train, augmentation), batch_size=16,
                        sampler=WeightedRandomSampler(weights, len(train), replacement=True))
    for epoch in range(1, args.fine_epochs + 1):
        # Keep pretrained BatchNorm statistics fixed even in the trainable blocks.
        model.eval()
        model.classifier.train()
        total = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        checkpoint("fine", epoch, total / len(train))

    model.load_state_dict(torch.load(out / "skin_model.pt", weights_only=True))
    final_validation = evaluate(model, validation_loader)
    audit_result = evaluate(model, audit_loader)
    # Same historical examples and label order as the original TFLite baseline.
    from ai_edge_litert.interpreter import Interpreter
    from predictor import SkinPredictor
    baseline = Interpreter(model_path=str(ROOT / "models/skin_model.tflite"), num_threads=2)
    baseline.allocate_tensors()
    baseline_probs = [SkinPredictor._run_tflite_model(r["path"], baseline, (224, 224), "raw_0_255") for r in audit]
    baseline_result = metrics(np.array([r["target"] for r in audit]), np.array(baseline_probs))
    unchanged = protected == {str(p): digest(p) for p in (ROOT / "models").glob("*") if p.is_file()}
    report = {"selected_epoch": best_epoch, "seed": args.seed, "counts": counts,
              "internal_validation": final_validation, "historical_audit": audit_result,
              "baseline_historical_audit": baseline_result, "active_files_unchanged": unchanged,
              "elapsed_seconds": round(time.monotonic() - started), "promoted": False,
              "limitations": ["Small, imbalanced four-class SCIN experiment; not clinically validated",
                  "Historical audit was previously used in development; not a blind external test",
                  "SCIN case IDs are used for separation; cross-case patient identity is unavailable",
                  "No new normal/non-skin or Maharashtra-specific training data",
                  "No automatic deployment, calibration claim, or promise of accuracy improvement"]}
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    (out / "labels.txt").write_text("\n".join(LABELS), encoding="utf-8")
    (out / "skin_model_metadata.json").write_text(json.dumps({
        "source": "SCIN case-separated balanced training candidate", "architecture": "PyTorch MobileNetV2",
        "class_names": LABELS, "input_preprocessing": "Resize(256), CenterCrop(224), ToTensor, ImageNet normalization",
        "training_device": "cpu", "sha256": digest(out / "skin_model.pt"),
        "selection": "Best internal-validation macro-F1", "approved_for_deployment": False,
        "evaluation_file": "evaluation.json"}, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"Candidate saved to {out}. Active model NOT changed.", flush=True)
    if not unchanged:
        raise RuntimeError("Active model files changed during run; investigate concurrent writes")


if __name__ == "__main__":
    main()
