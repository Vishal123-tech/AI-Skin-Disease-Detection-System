# Eleven-condition improvement audit — 21 September 2026

No production model, website interface, or live deployment was changed by this
audit. The separate four-condition candidate is not an approved replacement.

## Complete prediction pipeline, not just classifier scores

Local comparison on the same 79 historical SCIN images:

| Configuration | Correct condition | Wrong condition | Uncertain | Rejected as non-skin |
|---|---:|---:|---:|---:|
| Active baseline + existing gate/routing | 39 | 11 | 18 | 11 |
| Candidate + existing gate/routing | 46 | 11 | 11 | 11 |
| Candidate + gate, common-condition model only | 46 | 9 | 13 | 11 |

The third configuration is an in-memory diagnostic experiment, **not** an
eleven-condition system. Routing clinical photographs to the dermoscopic
classifier introduced two wrong labels in this comparison. The skin gate
rejected 11 genuine SCIN skin images in all three configurations. Neither
problem is fixed merely by training the disease classifier longer.

Twelve object smoke examples were rejected by the common-only candidate.
That small check does not establish general non-skin rejection performance.
The historical set has already been used during development; these are not
blind-test or clinical accuracy claims. Raw-classifier results are documented
separately in CANDIDATE_TRAINING_20260920.md.

Reproduce with:

```powershell
python scripts/evaluate_candidate_pipeline.py --candidate work/candidates/scin_balanced_20260920 --output outputs/eleven_class_pipeline_audit.json
```

## Supplied folder

`C:\Users\vy355\OneDrive\Desktop\New folder` contains 900 ISIC 2016 training
JPGs and 5,000 COCO val2017 JPGs. No diagnosis-label file was found there.
The COCO images are object examples, not labelled skin diseases; they require
review before treating all of them as non-skin negatives.

Official ISIC 2019 diagnosis and metadata CSVs were downloaded into ignored
`work/isic_label_audit/`. An exact filename-ID join recovered these labels:

| Condition | Matched local images |
|---|---:|
| Actinic Keratoses | 0 |
| Basal Cell Carcinoma | 0 |
| Benign Keratosis-like Lesions | 1 |
| Dermatofibroma | 0 |
| Melanocytic Nevi | 447 |
| Melanoma | 120 |
| Vascular Lesions | 0 |

568 matched; 332 remain unlabelled by this source. No matched image has a
lesion-group ID in that metadata. No exact duplicate hashes were found among
the matched files. Images were neither copied into training nor assigned
guessed labels. Filename matches alone do not verify patient independence.

`scripts/audit_isic_labels.py` records image hashes, source CSV hashes, exact
labels and unmatched paths in `work/isic_label_audit/local_image_labels.json`.
That manifest is explicitly not approved for training. Existing synthetic
`data/skin/sample_*.jpg` examples must not be used for disease training.

## Data required next

Obtain the complete ISIC 2018 Task 3 training images (about 2.6 GB), diagnosis
CSV, and supplemental lesion-group metadata from the
[official ISIC download page](https://challenge.isic-archive.com/data/).
Keep all images of one lesion in the same split and check duplicates across
sources. Review the CC-BY-NC licence and attribution requirements before use.
ISIC dermoscopy does not establish performance on phone-camera photographs,
nor does it establish India/Maharashtra-specific performance.

Retain the four-condition SCIN candidate separately. More verified clinical
photos, healthy skin, and reviewed non-skin examples are needed to assess the
gate and cross-domain routing. Independent patient-labelled evaluation for
all eleven conditions is required before promoting a combined replacement.
