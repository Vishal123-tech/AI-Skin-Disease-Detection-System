# Balanced SCIN candidate — 20 September 2026

Training completed locally on CPU. The candidate was **not deployed** and the
active model files were verified unchanged.

## Training

- Architecture: ImageNet-initialized MobileNetV2 with a new four-class head.
- Data: the existing labelled SCIN dataset only; no unreviewed feedback images.
- 261 training images, 53 internal validation images; SCIN case IDs and exact
  image hashes checked for cross-split overlap.
- Equal sampling weight for each class, and equal case weight within each class.
- Three cached training-only image views; 25 head-training epochs followed by
  four fine-tuning epochs. Pretrained BatchNorm running statistics stayed frozen.
- Checkpoint selected by internal-validation macro-F1: **head epoch 22**.
  Later fine-tuning did not outperform it.
- Model selection did not use the 79-image historical comparison set.

## Historical comparison (raw classifier predictions)

| Condition | Images | Original SCIN baseline correct | Candidate correct |
|---|---:|---:|---:|
| Acne Vulgaris | 5 | 3 | 4 |
| Eczema | 56 | 43 | 46 |
| Fungal Infection | 8 | 6 | 6 |
| Psoriasis | 10 | 1 | 5 |
| Total | 79 | 53 | 61 |

Accuracy: 67.1% → 77.2%. Macro-F1: 0.520 → 0.684.
Macro recall (equal weighting per condition): 55.4% → 71.8%.

These are raw top-1 classifier results, not results for the complete website
with its skin gate, legacy model routing and confidence rejection. The audit
set was used in prior development; this is **not** a blind or clinical test.
Acne has only three cases in this comparison. Scoring 4/5 is not evidence of
80% accuracy on arbitrary acne photos. Cross-case patient identity is unknown.
The candidate still missed half the psoriasis examples and confused fungal
infection with eczema in two of eight examples.

## Saved artifacts

All artifacts are under the ignored local directory
`work/candidates/scin_balanced_20260920/`:

- `skin_model.pt`: selected PyTorch weights.
- `labels.txt` and `skin_model_metadata.json`: label order and preprocessing.
- `split_manifest.json`: exact case-separated subsets and image hashes.
- `history.json`: epoch-by-epoch validation results.
- `evaluation.json`: confusion matrices, per-class scores and baseline comparison.

Reproduce in a new output directory:

```powershell
python train_candidate.py --data work/scin_training_dataset_test/scin_training_dataset --name scin_balanced_repeat --head-epochs 25 --fine-epochs 4 --seed 42
```

## Before activation

Run a separate local candidate trial with the existing gate and interface.
Evaluate consented, labelled images from new patients, including acne hard
negatives, eczema, fungal infection, psoriasis, normal skin and non-skin
objects. Evaluate different cameras and skin tones; obtain dermatologist
review. Do not activate through the old one-image feedback trainer. This run
does not add new disease classes or establish Maharashtra-specific performance.
