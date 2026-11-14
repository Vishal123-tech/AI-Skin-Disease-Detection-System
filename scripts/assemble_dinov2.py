"""Assemble and verify the experimental DINOv2 ONNX artifact from Git parts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "dinov2"
TARGET = MODEL_DIR / "skin_model.onnx"
PARTS = (MODEL_DIR / "skin_model.part0", MODEL_DIR / "skin_model.part1")
EXPECTED = json.loads((MODEL_DIR / "skin_model.runtime.json").read_text())["sha256"]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    if TARGET.exists() and digest(TARGET) == EXPECTED:
        print("DINOv2 model already assembled and verified", flush=True)
        return
    temporary = MODEL_DIR / "skin_model.onnx.partial"
    with temporary.open("wb") as output:
        for part in PARTS:
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    if digest(temporary) != EXPECTED:
        temporary.unlink(missing_ok=True)
        raise ValueError("DINOv2 model parts do not match the pinned checksum")
    temporary.replace(TARGET)
    print("DINOv2 model assembled and verified", flush=True)


if __name__ == "__main__":
    main()
