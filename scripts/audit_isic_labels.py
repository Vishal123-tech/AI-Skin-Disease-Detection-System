"""Audit exact ISIC filename matches; never infer diagnoses from an image.

Writes a provenance manifest, not an approved training split. Exact identity
matches do not establish patient independence or permission to redistribute.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

LABELS = {
    "AK": "Actinic Keratoses", "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis-like Lesions", "DF": "Dermatofibroma",
    "NV": "Melanocytic Nevi", "MEL": "Melanoma", "VASC": "Vascular Lesions",
    "SCC": "Squamous Cell Carcinoma", "UNK": "Unknown",
}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["image"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {path}")
    return {row["image"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.images.is_dir():
        raise ValueError("Image directory does not exist")
    diagnoses, metadata = read_rows(args.labels), read_rows(args.metadata)
    matched, unmatched, ambiguous = [], [], []
    images = sorted(p for p in args.images.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in images:
        row = diagnoses.get(path.stem)
        if row is None:
            unmatched.append(str(path))
            continue
        values = {key: float(row[key]) for key in LABELS}
        active = [key for key, value in values.items() if value == 1.0]
        if len(active) != 1 or any(v not in (0.0, 1.0) for v in values.values()):
            ambiguous.append(str(path))
            continue
        code = active[0]
        matched.append({
            "image_id": path.stem, "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "label_code": code, "condition": LABELS[code],
            "lesion_id": metadata.get(path.stem, {}).get("lesion_id") or None,
        })
    counts = Counter(item["label_code"] for item in matched)
    result = {
        "approved_for_training": False,
        "reason": "Incomplete class coverage; patient/lesion grouping requires review.",
        "source": "https://challenge.isic-archive.com/data/",
        "label_file_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "metadata_file_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "total_local_images": len(images), "matched_images": len(matched),
        "class_counts": {LABELS[key]: counts[key] for key in LABELS},
        "matched_with_lesion_id": sum(bool(item["lesion_id"]) for item in matched),
        "duplicate_matched_image_hashes": len(matched) - len({r["sha256"] for r in matched}),
        "matched": matched, "unmatched": unmatched, "ambiguous": ambiguous,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"matched", "unmatched", "ambiguous"}}, indent=2))


if __name__ == "__main__":
    main()
