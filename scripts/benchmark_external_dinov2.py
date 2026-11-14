"""Local-only comparison of a third-party classifier on existing held-out photos."""

import importlib.metadata as metadata
import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps

# This workspace's bundled CPU torch reports its runtime version but has an
# incomplete distribution metadata version. Transformers checks that metadata.
_original_version = metadata.version
metadata.version = lambda name: torch.__version__ if name == "torch" else _original_version(name)

from transformers import AutoImageProcessor, AutoModelForImageClassification


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "work" / "candidates" / "external_dinov2_20261113"
MODEL = HERE / "model"
MANIFEST = ROOT / "work/candidates/multisource_20261023/split_manifest.json"
SPLITS = ("internal_test", "background_test", "healthy_source_test", "dermacon_test")
NORMAL = 4
MODEL_NORMAL = 22
MAPPING = {0: 0, 2: 1, 5: 1, 13: 2, 18: 2, 14: 3, 22: 4}


def score(rows):
    confusion = [[0] * 6 for _ in range(5)]
    for row in rows:
        confusion[row["target"]][row["mapped_prediction"]] += 1
    normal = [r for r in rows if r["target"] == NORMAL]
    disease = [r for r in rows if r["target"] != NORMAL]
    return {
        "images": len(rows),
        "normal_images": len(normal),
        "normal_correct": sum(r["prediction_index"] == MODEL_NORMAL for r in normal),
        "normal_to_eczema": sum(r["mapped_prediction"] == 1 for r in normal),
        "normal_to_disease": sum(r["prediction_index"] != MODEL_NORMAL for r in normal),
        "disease_images": len(disease),
        "disease_to_normal": sum(r["prediction_index"] == MODEL_NORMAL for r in disease),
        "unsupported_class_outputs": sum(r["mapped_prediction"] == 5 for r in rows),
        "supported_disease_name_correct": sum(r["target"] == r["mapped_prediction"] for r in disease),
        "confusion_matrix_5_plus_unsupported": confusion,
        "predictions_by_original_label": dict(Counter(r["prediction_label"] for r in rows)),
    }


def main():
    torch.set_num_threads(2)
    start = time.monotonic()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print("Loading processor", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    print("Loading model", flush=True)
    model = AutoModelForImageClassification.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    print(f"Model ready after {time.monotonic() - start:.1f}s", flush=True)
    results = {}
    for split in SPLITS:
        source = manifest[split]
        rows = []
        print(f"Starting {split}: {len(source)} images", flush=True)
        for index in range(0, len(source), 8):
            batch_rows = source[index:index + 8]
            images = []
            for row in batch_rows:
                with Image.open(row["path"]) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits
            predicted = logits.argmax(dim=-1).tolist()
            for row, index_pred in zip(batch_rows, predicted):
                rows.append({
                    "row_id": row["row_id"],
                    "target": row["target"],
                    "source": row["source"],
                    "prediction_index": index_pred,
                    "prediction_label": model.config.id2label[index_pred],
                    "mapped_prediction": MAPPING.get(index_pred, 5),
                })
            if (index // 8 + 1) % 2 == 0:
                print(f"{split}: {min(index + 8, len(source))}/{len(source)} ({time.monotonic() - start:.1f}s)", flush=True)
        results[split] = score(rows)
        print(f"{split}: {json.dumps({k: v for k, v in results[split].items() if k not in ('confusion_matrix_5_plus_unsupported', 'predictions_by_original_label')})}", flush=True)
    output = {
        "model": "shindevaishnavi304/dinov2-finetuned-skin-disease",
        "revision": "2d631c6f61021a58f540ae15fb4f35240a3b6c4b",
        "model_labels": model.config.id2label,
        "mapping": MAPPING,
        "source": "Existing multisource split manifest; labels not independently clinically verified",
        "evaluation": "Raw classifier only; no SkinScanix quality gate, fallback or PDF pipeline",
        "runtime_seconds": round(time.monotonic() - start, 1),
        "splits": results,
    }
    (HERE / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
