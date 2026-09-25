"""Compare candidate routing locally without changing deployed files or thresholds.

The historical SCIN set only evaluates the four common conditions. It cannot
establish accuracy for the seven dermoscopic classes, normal skin or new patients.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GEMINI_API_KEY"] = ""
os.environ["ACNE_FOCUS_MODE"] = "0"
os.environ["ACNE_PRIORITY_MODE"] = "0"
import torch
from predictor import SkinPredictor


def file_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "models").iterdir() if p.is_file()}


def summarize(rows):
    groups = {}
    for condition in sorted({r["expected"] for r in rows}):
        examples = [r for r in rows if r["expected"] == condition]
        accepted = [r for r in examples if r["category"] == "lesion"]
        groups[condition] = {
            "images": len(examples), "condition_outputs": len(accepted),
            "correct_condition_outputs": sum(r["raw_class"] == condition for r in accepted),
            "incorrect_condition_outputs": sum(r["raw_class"] != condition for r in accepted),
            "categories": dict(Counter(r["category"] for r in examples)),
            "selected_models": dict(Counter(r["model_used"] for r in examples)),
        }
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    protected = file_hashes()
    candidate = args.candidate.resolve()
    meta = json.loads((candidate / "skin_model_metadata.json").read_text())
    if hashlib.sha256((candidate / "skin_model.pt").read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Candidate checkpoint checksum mismatch")
    manifest = json.loads((candidate / "split_manifest.json").read_text())
    audit = manifest["historical_audit"]
    baseline = SkinPredictor(ROOT / "models/skin_model.tflite", ROOT / "models/labels.txt")
    trial = SkinPredictor(candidate / "skin_model.tflite", candidate / "labels.txt",
                         gate_model_path=ROOT / "models/skin_gate.tflite",
                         gate_labels_path=ROOT / "models/skin_gate_labels.txt")
    # Reuse the identical legacy artifact for this serial comparison.
    trial.legacy_interpreter = baseline.legacy_interpreter
    trial.legacy_labels = baseline.legacy_labels
    trial.legacy_image_size = baseline.legacy_image_size
    for p in (baseline, trial):
        p._run_local_model = lru_cache(maxsize=256)(p._run_local_model)
        p._run_legacy_model = lru_cache(maxsize=256)(p._run_legacy_model)
        p._run_skin_gate = lru_cache(maxsize=256)(p._run_skin_gate)
    records = {}
    for name, predictor, use_legacy in (("baseline_auto", baseline, True),
                                      ("candidate_auto", trial, True),
                                      ("candidate_common_only", trial, False)):
        predictor.legacy_interpreter = baseline.legacy_interpreter if use_legacy else None
        rows = []
        for row in audit:
            result = predictor.predict(row["path"])
            rows.append({"image": Path(row["path"]).name, "case": row["case"], "expected": row["label"],
                         **{key: result.get(key) for key in
                            ("category", "raw_class", "confidence", "model_used")}})
        records[name] = rows
        print(name, json.dumps(summarize(rows)), flush=True)
    smoke = []
    # A few labelled object examples only: a smoke check, not a rejection benchmark.
    objects = sorted((ROOT / "data/skin_gate/val/NON_SKIN").glob("*.jpg"))[:12]
    for path in objects:
        r = trial.predict(str(path))
        smoke.append({"image": path.name, "category": r["category"]})
    result = {"summaries": {key: summarize(rows) for key, rows in records.items()},
              "records": records, "object_smoke": smoke,
              "active_files_unchanged": protected == file_hashes(),
              "limitations": ["Historical validation, not a blind test",
                  "Seven-class dermoscopic accuracy NOT evaluated: no verified labelled evaluation set located",
                  "Common-only mode must not be used as an 11-class classifier",
                  "Object examples may have been used in prior development; no clinical validation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Object smoke:", dict(Counter(r["category"] for r in smoke)), flush=True)
    print("Active model files unchanged:", result["active_files_unchanged"], flush=True)
    if not result["active_files_unchanged"]:
        raise RuntimeError("Model files changed concurrently; investigate")


if __name__ == "__main__":
    main()
