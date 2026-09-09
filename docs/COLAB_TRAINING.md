# Google Colab training

## Prepare SCIN data first

Open [`COLAB_PREPARE_SCIN.ipynb`](../COLAB_PREPARE_SCIN.ipynb) in Google Colab. It uses the official SCIN Google Cloud Storage bucket, asks you to authenticate with Google, prints the exact dermatologist labels, and downloads only the labels you explicitly map. Do not guess a label name.

The SCIN project is documented at <https://github.com/google-research-datasets/scin>. It provides `scin_cases.csv`, `scin_labels.csv`, and image paths in `dataset/images/`; the GitHub ZIP alone does not contain the full image dataset.

The preparation notebook creates `scin_training_dataset.zip` with `train/` and `val/` folders. Upload that ZIP to the training notebook below.

SCIN does not supply the project's `Normal Skin` and `Other Non Skin` classes. Add healthy, consented skin images and verified object/document/room/animal images separately. SCIN is collected from US contributors, so Maharashtra-specific training also needs consented Indian images reviewed by a qualified dermatologist.

Open [`COLAB_TRAIN_INDIAN_SKIN.ipynb`](../COLAB_TRAIN_INDIAN_SKIN.ipynb) in Google Colab. Upload a ZIP containing `train/` and `val/` folders, with one correctly labeled folder per class.

For the first Maharashtra-focused prototype, use a small, honest class list such as:

- Acne Vulgaris
- Eczema
- Fungal Infection
- Psoriasis
- Scabies
- Vitiligo
- Impetigo
- Warts
- Normal Skin
- Other Non Skin

Only include a class when you have enough correctly labeled images. The current ISIC 2016 images plus segmentation masks do not provide disease names, so they cannot be used by themselves for this training task. Do not scrape or upload private patient images without consent and permission.

The notebook exports three files:

- `skin_model.tflite`
- `labels.txt`
- `skin_model_metadata.json`

Copy those files into this project’s `models/` folder only after checking the classification report and testing Indian phone-camera images. Keep the existing `skin_gate.tflite`; the disease model and the skin/non-skin gate solve different problems.
