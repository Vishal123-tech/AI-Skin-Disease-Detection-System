"""Review/export feedback records before any future retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from feedback import FEEDBACK_LOG


def read_feedback(path: Path = FEEDBACK_LOG) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def export_csv(output: Path, records: list[dict]) -> None:
    fields = [
        "id", "created_at", "rating", "predicted_label", "predicted_category",
        "confidence", "model_used", "correct_label", "comment", "image_file",
        "image_sha256",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review/export local model feedback")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    args = parser.parse_args()
    records = read_feedback()
    print(f"Feedback records: {len(records)}")
    print(f"Wrong predictions needing review: {sum(r.get('rating') == 'Wrong' for r in records)}")
    if args.output:
        export_csv(args.output, records)
        print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
