# Local model regression audit — September 2026

This is an engineering regression check, not a clinical validation or a new
training run. No live deployment was updated as part of this rollback.

## Evidence

The feedback-trained `skin_model.pt` was automatically preferred over the older
SCIN export. Its metadata records one feedback sample, zero replay samples and
100% validation accuracy. The previous trainer reused the training image for
validation when only one sample existed. That score did not measure generalization.

The local SCIN dataset has 314 training images and 79 validation images. The
audit excluded matching training case IDs and exact image hashes; none of the
79 validation images matched. There is no separate test split here, and this
split was already used during development. It is not independent of all past
training, particularly the acne specialist's data preparation.

Raw classifier top-1 counts (before skin gates, uncertainty rejection or legacy routing):

| Label | Validation images | Feedback PT correct | Original SCIN TFLite correct |
|---|---:|---:|---:|
| Acne Vulgaris | 5 | 0 | 3 |
| Eczema | 56 | 56 | 43 |
| Fungal Infection | 8 | 0 | 6 |
| Psoriasis | 10 | 0 | 1 |

The PT model predicted Eczema on **all 79 images**. Its 70.9% overall accuracy
merely reflects class imbalance; macro recall is 25%. The original baseline
has 67.1% overall accuracy and 55.4% macro recall. Restoring it recovers some
class discrimination, not uniformly better accuracy. Psoriasis remains weak.

The restored **end-to-end** pipeline (including skin gate, confidence rejection
and legacy fallback) produced 50 condition outputs: 39 matched the validation
labels and 11 did not. Another 18 images were uncertain and 11 genuine skin
images were rejected as non-skin. Correct accepted outputs were 3/5 acne,
33/56 eczema, 3/8 fungal, and 0/10 psoriasis. These results show substantial
remaining model, gate and routing limitations; fewer disagreement messages
must not be presented as accurate disease detection.

The binary acne specialist labelled 64 of 74 non-acne examples as Acne.
Forcing agreement with that specialist therefore rejected many broader-model
results; letting it override the broader model caused acne false positives.

## Local correction

- Retain all model binaries and backups; do not delete the PT checkpoint.
- Pin the original TFLite artifact by SHA-256 and class order.
- Use raw 0–255 pixels, matching its embedded rescaling and original export metadata.
- Disable the acne cross-check in normal inference; retain the skin gate and legacy model.
- Pause the unsafe feedback trainer. Feedback remains available for human review.
- Keep the existing Gradio layout and confidence thresholds. Low certainty is
  not automatically evidence of poor image quality.

## Reproduction

From the project directory, using its Python environment:

```powershell
python scripts/audit_skin_models.py work/scin_training_dataset_test/scin_training_dataset --pipeline --output outputs/model_audit_after.json
python -m unittest discover -s tests -v
```

The audit never uploads images or activates checkpoints. Detailed results stay
in ignored `outputs/`; the original pre-rollback comparison is saved locally as
`outputs/model_audit_before.json`. Re-running uses the **currently selected**
model, so it does not recreate the previous PT run after the rollback.

## Still needed

Review consent and labels, assemble more balanced acne/fungal/eczema/psoriasis
training data, split by patient, train a separate candidate, and evaluate
per-class sensitivity and false positives on untouched patients, normal skin,
objects, diverse skin tones and phone-camera images. Neither model is proven
for diagnosing disease, Maharashtra-specific use, or all skin conditions.
