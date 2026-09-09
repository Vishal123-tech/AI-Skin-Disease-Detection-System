# Acne-focused training plan

The project can be focused on acne, but it must use a real image dataset. The
placeholder images under `data/skin/` are for interface smoke tests and must
not be used to claim acne accuracy.

## Recommended first dataset

Use the [ACNE04 dataset on Hugging Face](https://huggingface.co/datasets/timsep/acne04)
or the [ACNE04 research code](https://github.com/xpwu95/LDL). ACNE04 is an
academic research dataset and its underlying license is for academic use.
Read and follow the dataset terms before redistributing images or deploying a
public service. The newer [ACNE04-v2 annotations](https://github.com/AIpourlapeau/acne04v2)
are useful for acne-lesion annotations and severity experiments.

ACNE04 images are acne-positive examples. For a binary screening model, add a
separate, correctly labelled `Not_Acne` class containing healthy skin and
non-acne skin conditions captured in a similar way. The ISIC 2016 folder you
previously supplied is a lesion-image set without acne labels, so it cannot be
treated as a verified acne dataset by itself. Do not use tables, rooms,
animals, or documents as `Not_Acne`; those belong to the existing
`skin_gate.tflite` model. Do not use web images with unclear permission or
private patient images without written consent.

## Folder structure

```text
data/acne/
├── train/
│   ├── Acne/
│   └── Not_Acne/
├── val/
│   ├── Acne/
│   └── Not_Acne/
└── test/
    ├── Acne/
    └── Not_Acne/
```

You can use `prepare_acne_dataset.py` to copy already labelled source folders
into this structure:

```bash
python prepare_acne_dataset.py --acne path/to/Acne --not-acne path/to/Not_Acne
```

Keep images from the same person in only one split. Otherwise the score can be
artificially high because the model sees nearly identical faces in training
and testing. Before training, remove duplicates, watermarks, extremely blurry
images, and images where the skin area is too small.

For a credible college prototype, aim for at least 1,000 acne-positive and
1,000 non-acne images, with a held-out test set and a separate phone-camera
set. More images are helpful, but label quality and patient-level separation
matter more than a large unverified number.

## Train in Colab or locally

Install TensorFlow (use `requirements-train.txt` on a training machine), place
the folders above in the workspace, then run:

```bash
python train_acne.py --data data/acne --head-epochs 12 --fine-tune-epochs 20
```

The script uses MobileNetV2 transfer learning, class weights, augmentation,
early stopping, fine-tuning, and TFLite export. It creates:

- `models/acne_model.tflite`
- `models/acne_labels.txt`
- `models/acne_model_metadata.json`
- `reports/acne_training_report.json`

Only copy the model into a deployment after checking the independent test
metrics and false-positive rate. The existing skin/non-skin gate remains a
separate first stage.

## What this model can and cannot say

The model is a binary acne screening model. A result such as “Acne pattern
possible” is not proof of acne and “Not acne” does not rule out a disease. The
current project is for education and demonstration, not diagnosis or medical
treatment decisions.

When `models/acne_model.tflite` is present, the app checks this model first by
default (`ACNE_PRIORITY_MODE=1`). A strong acne result is returned; otherwise
the broader SCIN/legacy disease classifiers continue. Use
`ACNE_FOCUS_MODE=1` only for an acne-only demonstration.
