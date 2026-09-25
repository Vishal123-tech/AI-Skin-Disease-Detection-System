"""Local, read-only model comparison. Never trains or promotes checkpoints."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import SkinPredictor
from ai_edge_litert.interpreter import Interpreter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", action="store_true", help="Also evaluate gating and routing")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    os.environ["GEMINI_API_KEY"] = ""  # Audit only local models; never upload dataset images.
    predictor = SkinPredictor(ROOT / "models/skin_model.tflite", ROOT / "models/labels.txt")
    baseline = Interpreter(model_path=str(ROOT / "models/skin_model.tflite"), num_threads=2)
    baseline.allocate_tensors()
    labels = predictor.labels
    records = []
    train = list((args.dataset / "train").rglob("*.jpg"))
    train_cases = {p.stem.split("_image_")[0] for p in train}
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in train}
    excluded = 0
    for split in ("val", "test"):
        for path in sorted((args.dataset / split).rglob("*.jpg")):
            if path.parent.name not in labels:
                continue
            if (path.stem.split("_image_")[0] in train_cases or
                    hashlib.sha256(path.read_bytes()).hexdigest() in train_hashes):
                excluded += 1
                continue
            scores = {
                "active_model": predictor._run_local_model(str(path)),
                "original_tflite_raw": predictor._run_tflite_model(str(path), baseline, (224, 224), "raw_0_255"),
                "acne_specialist": predictor._run_acne_model(str(path)),
            }
            row = {"split": split, "file": str(path.relative_to(args.dataset)), "expected": path.parent.name}
            for name, probs in scores.items():
                names = predictor.acne_labels if name == "acne_specialist" else labels
                row[name] = {"label": names[int(np.argmax(probs))], "score": float(np.max(probs)),
                             "probabilities": dict(zip(names, map(float, probs)))}
            if args.pipeline:
                result = predictor.predict(str(path))
                row["pipeline"] = {key: result.get(key) for key in
                                   ("category", "raw_class", "confidence", "model_used", "uncertainty_reason")}
            records.append(row)
    summaries = {}
    for split in ("val", "test"):
        rows = [r for r in records if r["split"] == split]
        for name in ("active_model", "original_tflite_raw", "acne_specialist"):
            per_class = {}
            for label in labels:
                group = [r for r in rows if r["expected"] == label]
                target = ("Acne" if label == "Acne Vulgaris" else "Not_Acne") if name == "acne_specialist" else label
                correct = sum(r[name]["label"] == target for r in group)
                per_class[label] = {"count": len(group), "correct": correct,
                                    "predictions": dict(Counter(r[name]["label"] for r in group))}
            total_correct = sum(g["correct"] for g in per_class.values())
            summaries[f"{split}/{name}"] = {"images": len(rows), "correct": total_correct,
                "accuracy": total_correct / len(rows) if rows else None, "per_class": per_class}
    pipeline_summary = {}
    if args.pipeline:
        for label in labels:
            group = [r for r in records if r["expected"] == label]
            accepted = [r for r in group if r["pipeline"]["category"] == "lesion"]
            pipeline_summary[label] = {"images": len(group), "accepted": len(accepted),
                "correct_accepted": sum(r["pipeline"]["raw_class"] == label for r in accepted),
                "categories": dict(Counter(r["pipeline"]["category"] for r in group))}
    result = {"active_backend": predictor.backend, "excluded_train_overlap": excluded,
              "preprocessing": predictor.preprocess_mode, "acne_crosscheck": predictor.acne_priority,
              "pipeline": pipeline_summary,
              "limitations": "Small existing local splits; not an independent clinical validation. Binary and multiclass accuracies are not comparable.",
              "summaries": summaries, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
