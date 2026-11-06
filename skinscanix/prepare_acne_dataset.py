"""Build a leakage-resistant acne dataset from already labelled folders.

This copies images; it does not download, scrape, or invent labels. Use it
only when the source folders are genuinely labelled ``Acne`` and
``Not_Acne``. Keep a separate test set from different people whenever that
information is available.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def unique(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            result.append(path)
    return result


def copy_split(paths: list[Path], destination: Path, prefix: str, rng: random.Random) -> None:
    rng.shuffle(paths)
    n = len(paths)
    test_count = round(n * 0.15)
    val_count = round(n * 0.15)
    split_paths = {
        "test": paths[:test_count],
        "val": paths[test_count:test_count + val_count],
        "train": paths[test_count + val_count:],
    }
    for split, selected in split_paths.items():
        out_dir = destination / split / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(selected):
            shutil.copy2(path, out_dir / f"{prefix.lower()}_{index:05d}{path.suffix.lower()}")
        print(f"{prefix:10s} {split:5s}: {len(selected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare labelled acne data")
    parser.add_argument("--acne", required=True, help="Folder containing acne-positive images")
    parser.add_argument("--not-acne", required=True, help="Folder containing non-acne skin images")
    parser.add_argument("--output", default="data/acne")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    acne = unique(files(Path(args.acne)))
    not_acne = unique(files(Path(args.not_acne)))
    if not acne or not not_acne:
        raise SystemExit("Both source folders must contain real labelled images.")

    print(f"Unique acne images: {len(acne)}")
    print(f"Unique not-acne images: {len(not_acne)}")
    output = Path(args.output)
    copy_split(acne, output, "Acne", random.Random(args.seed))
    copy_split(not_acne, output, "Not_Acne", random.Random(args.seed + 1))
    print(f"Prepared dataset at: {output.resolve()}")


if __name__ == "__main__":
    main()
