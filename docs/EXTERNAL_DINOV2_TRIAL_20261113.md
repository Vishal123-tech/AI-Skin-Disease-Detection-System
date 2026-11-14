# Local pretrained-model comparison — 13 November 2026

## Decision

The owner requested an **experimental live trial** despite worse main held-out results for both normal-to-disease and disease-to-normal errors. The DINOv2 build deployed as commit `f05382b`, but the first public analysis pushed Render Free over its **512 MB memory limit** and restarted the instance. Recovery commit `4aceb29` restored the five-class model; a public synthetic-image analysis completed and returned a PDF. This experiment is **not** an accuracy improvement or clinical validation.

## Candidate and method

- Candidate: `shindevaishnavi304/dinov2-finetuned-skin-disease`, Hugging Face revision `2d631c6f61021a58f540ae15fb4f35240a3b6c4b`.
- Local weights: 346,489,044 bytes, 23 output labels. `normal skin` is class 22; Eczema/Atopic Dermatitis outputs were grouped as Eczema for this comparison.
- Used the model's saved image processor and original test photo paths in `work/candidates/multisource_20261023/split_manifest.json`. No photos were uploaded to the model provider.
- Compared raw classifier results with the active model's previously saved raw-classifier results in `work/candidates/normal_phone_focus_20261113_preserve_eczema/evaluation.json`. This is not an end-to-end UI or clinical accuracy test.
- Existing labels are source-described/provisional and have not been independently clinically verified. The 19-photo healthy-source split has only one source group.

## Results

| Held-out split | Measure | Current model | External DINOv2 |
| --- | --- | ---: | ---: |
| Internal test (343) | Normal correct / 75 | 65 | 57 |
| Internal test (343) | Normal called Eczema / 75 | 2 | 6 |
| Internal test (343) | Normal called any disease / 75 | 10 | 18 |
| Internal test (343) | Disease called Normal / 268 | 13 | 21 |
| Background normal (68) | Normal correct / 68 | 61 | 57 |
| Background normal (68) | Normal called Eczema / 68 | 6 | 0 |
| Healthy-source normal (19) | Normal correct / 19 | 8 | 17 |
| Healthy-source normal (19) | Normal called Eczema / 19 | 2 | 0 |

On the internal test, the external model emitted 61 labels outside SkinScanix's four supported disease categories plus Normal. Its disease-name outputs therefore cannot be treated as interchangeable with the current five-class model. It matched 178 of the 268 disease labels after mapping the directly corresponding categories.

The original local run took about 10 minutes for the 430 photos across the first three splits, including Python/model startup. A further 405-photo disease-only split was stopped after the main replacement checks had already failed. A compact ONNX export subsequently reproduced the three completed split counts and was smoke-tested through a local Gradio upload and PDF report. The public Render analysis **failed due to memory exhaustion** despite a successful build and startup.

## Experimental hosted artifact

- The source model is [Apache-2.0 licensed](https://huggingface.co/shindevaishnavi304/dinov2-finetuned-skin-disease). Its model card does not document the training dataset or intended-use limitations in detail.
- The selected export is FP16 ONNX with FP32 input/output, 173,547,022 bytes, SHA-256 `23735a7ac2dec5eee8204539914ec358d7da9908118d9e1f647bd7fa647aff24`. The original model is about 346 MB. The checked-in two-part artifact is assembled and hash-verified by `scripts/assemble_dinov2.py` during the build.
- The saved image processor's bicubic 256-pixel resize, 224-pixel center crop, and ImageNet normalization were matched locally. On 60 balanced held-out images, the FP16 export and original model agreed on the top class for all 60; their maximum per-class probability difference was about 0.0028. The full 430-image comparison reproduced the original error counts above.
- Dynamic INT8 variants were rejected because they caused severe prediction regressions. The hosted build uses **FP16**, not INT8.
- The DINOv2 selection disables the old acne and legacy specialists. Unmapped source categories are shown as uncertain rather than silently assigned a supported disease name. The interface identifies the active model as experimental.
- With ONNX memory caching disabled, a local Python/Gradio process used about 411 MB resident memory after an analysis on Windows. Render's Linux instance exceeded 512 MB on a public analysis request, showing that the local measurement did not predict hosted peak memory.

## Reproduction

[`scripts/benchmark_external_dinov2.py`](../scripts/benchmark_external_dinov2.py) records the label mapping, preprocessing, and split selection. The downloaded model files are local under `work/candidates/external_dinov2_20261113/model/` and are ignored by Git. This comparison ended early after the decisive main split and two additional normal-photo splits, so `evaluation.json` was not written by the interrupted script. The console summary above was transcribed directly from the completed split outputs.

The separate [local trial launcher](../scripts/launch_external_dinov2_trial.py) preserves the SkinScanix interface and uses the original downloaded weights. An initial local trial displayed Demo mode for every image because `SkinPredictor.demo_mode` did not count the ONNX backend and the trial had intentionally disabled the legacy fallback. Adding the ONNX session to that check resolved the issue. After the fix, one normal-labelled and one eczema-labelled held-out photo each produced a non-demo result and a PDF through `/analyze`. The compact ONNX deployment path passed the same workflow smoke test. These are software checks, not clinical validation.

Next useful experiment: a smaller classifier trained on representative, reliably labelled normal and disease phone photos, with subject separation and the same error checks on each source. The favorable 19-photo result alone does not justify an accuracy claim.
