"""Publish only explicitly listed application assets, never user images or secrets.

Uses the existing Hugging Face login. Run with --repo OWNER/SPACE.
Creates a free CPU Gradio Space if missing; does not change existing hardware.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "README.md", "gradio_app.py", "branding.py", "ui_presentation.py", "config.py",
    "predictor.py", "quality.py", "report.py", "feedback.py", "self_train.py",
    "static/skinscanix-logo.png", "static/dermaai.css", "static/theme-tokens.css",
    "models/skin_model.tflite", "models/labels.txt", "models/skin_model.runtime.json",
    "models/skin_gate.tflite", "models/skin_gate_labels.txt",
    "models/skin_model_legacy.tflite", "models/labels_legacy.txt",
    "models/skin_model_legacy_metadata.json", "models/skin_model_metadata.json",
    "models/acne_model.tflite", "models/acne_labels.txt", "models/acne_model_metadata.json",
    "docs/ARCHITECTURE.md", "docs/ACNE_TRAINING.md", "docs/COLAB_TRAINING.md",
    "docs/FEEDBACK_LOOP.md", "docs/PROJECT_STATUS.md", "docs/MODEL_REGRESSION_AUDIT.md",
    "docs/CANDIDATE_TRAINING_20260920.md", "docs/ELEVEN_CLASS_DATA_AUDIT.md",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    # Gradio Spaces install root requirements.txt, not requirements-space.txt.
    base = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    extra = (ROOT / "requirements-space.txt").read_text(encoding="utf-8")
    requirements = base + "\n" + "\n".join(
        line for line in extra.splitlines() if line.strip() != "-r requirements.txt"
    ) + "\n"
    print(json.dumps({"repo": args.repo, "files": FILES + ["requirements.txt"],
                      "private_data_included": False}, indent=2))
    if args.dry_run:
        return
    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=ROOT / name) for name in FILES]
    operations.append(CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj=requirements.encode()))
    info = api.create_commit(args.repo, repo_type="space", operations=operations,
                             commit_message="Deploy SkinScanix branding, compact reports and verified local baseline")
    print("Space:", f"https://huggingface.co/spaces/{args.repo}")
    print("Commit:", info.oid)


if __name__ == "__main__":
    main()
