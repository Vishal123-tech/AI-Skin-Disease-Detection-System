# Project Status

Last reviewed: 28 August 2026

## Completed

- Simple Gradio upload interface restored locally to match the current live design.
- Flask web interface and JSON API are available locally.
- Image upload, quality checks, prediction response, and PDF report generation work locally.
- Raspberry Pi capture code supports Picamera2 and OpenCV fallback.
- Non-skin gate model trained and included as `models/skin_gate.tflite`.
- Gate labels included as `models/skin_gate_labels.txt`.
- Gate dataset contains 6,100 images: 4,720 training images and 1,380 validation images.
- Random-object/document rejection has been tested locally.
- A conservative gate-recovery rule now allows a close-up image with at least 50% skin-colour coverage to continue when the trained gate makes a false `NON_SKIN` prediction.
- SCIN metadata was prepared in Colab using the official public bucket and pure labels only.
- A MobileNetV2 transfer-learning classifier was trained locally in Colab on 393 SCIN images: 314 training images and 79 validation images.
- The trained classifier currently has four classes: Acne Vulgaris, Eczema, Fungal Infection, and Psoriasis.
- Validation results for this small baseline were 73% accuracy and 0.60 macro-F1; the exported TFLite model and labels are installed locally.
- The previous local model is backed up under `work/pre_scin_model_20260828`.
- The previous seven-class model is also bundled as `models/skin_model_legacy.tflite`; the predictor uses it as a conservative fallback when the four-class SCIN model is weak.
- Ollama was removed from the prediction path.
- Live Render/Hugging Face deployment has not been changed during this local work.
- An acne-priority training pipeline was added in `train_acne.py`; it keeps the
  existing gate and broader disease experts instead of replacing them.
- A local provisional acne dataset was prepared from 390 ACNE04 images and 899
  supplied ISIC lesion images. It is for an academic baseline only and is not
  Maharashtra-specific.

## Current limitations

- `models/skin_model.tflite` is now the four-class SCIN baseline trained in Colab and converted to TFLite.
- The training subset is very small and imbalanced: 15 Acne, 127 Eczema, 24 Fungal Infection, and 18 Psoriasis cases before the image split.
- The model is not Maharashtra-specific and should not be presented as reliable diagnosis; the validation macro-F1 of 0.60 is an early prototype result.
- The two-model fallback is a compatibility layer, not a properly trained unified 11-class classifier; model confidence values from the two models are not directly calibrated against each other.
- Testing on the repository's tiny synthetic validation folders still produces mostly Vascular Lesions and misclassifies the synthetic Normal_Skin folder. This confirms that the baseline is not a final classifier for this project.
- Testing on 900 supplied ISIC images produced mostly Vascular Lesions (775/900), so its confidence must not be treated as project accuracy.
- The repository's `data/skin` folders contain only small synthetic samples and are not suitable for final disease training.
- The 900 ISIC 2016 images previously supplied have lesion masks but no disease-name labels; masks alone cannot train the requested disease classifier.
- The gate detects skin versus non-skin. It does not identify acne, eczema, psoriasis, fungal infection, or other disease names.
- A focused two-class acne model was trained and exported locally as
  `models/acne_model.tflite`; `ACNE_PRIORITY_MODE=1` checks it before the
  broader disease experts, while `ACNE_FOCUS_MODE=1` is available for an
  acne-only demo.
- The provisional acne test split reported 100% accuracy and AUC 1.0, but this
  must not be treated as real-world accuracy because the source mix is small,
  not Maharashtra-specific, and not a patient-separated phone-camera study.
- A Gradio browser page and the Flask JSON API are separate interfaces. `pi_capture.py` currently targets Flask `/api/predict`, not a browser URL.
- The project does not yet have a tested production API endpoint connecting a Raspberry Pi directly to the hosted Gradio deployment.

## Remaining work before a proper live release

1. Choose a fixed, honest class list. Do not promise detection of every skin disease. The intended college prototype is a broader classifier with acne checked first.
2. Obtain real, legally usable, disease-labeled images for each selected class.
3. Include normal skin, non-skin images, and an unknown/out-of-scope class or rejection rule.
4. Split data by patient or lesion identity, not only by filename, to avoid data leakage.
5. Train a new classifier in Google Colab or on a capable computer using class balancing and augmentation.
6. Measure per-class precision, recall, macro-F1, confusion matrix, and rejection performance.
7. Test on Indian/Maharashtra-style camera images separately from dermoscopic images.
8. Collect substantially more consented, dermatologist-reviewed Indian/Maharashtra images for one fixed combined class list, train a single unified classifier, and compare it against the current two-model baseline.
9. Test the complete flow: camera -> API -> gate -> classifier -> report -> mobile browser.
10. Decide whether Render should run Flask API, or whether the Pi should use a documented Gradio API client.
11. Update deployment files only after local tests pass.
12. Deploy and smoke-test the live service without uploading private patient images.

## Release rule

The live site should not be advertised as proper disease detection until the replacement classifier passes the class-by-class evaluation and the end-to-end Raspberry Pi test.
