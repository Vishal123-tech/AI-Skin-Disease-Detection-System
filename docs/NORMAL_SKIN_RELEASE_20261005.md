# Experimental Normal Skin release — 5 October 2026

This owner-authorized educational release retains the existing Gradio interface,
branding and PDF design. The primary runtime is the five-class candidate tested
locally: Acne Vulgaris, Eczema, Fungal Infection, Psoriasis, Normal Skin.
The skin gate and seven-class legacy lesion model are unchanged.

## Runtime and verification

MobileNetV2 float32 ONNX uses CPU inference with EXIF orientation correction,
RGB, bilinear shorter-side resize to 256, center crop 224 and ImageNet
normalization. Softmax is included in the export. The runtime manifest verifies
the checksum and label order and disables Gemini and acne-specific overrides
to preserve trial behavior. Original PT/TFLite files remain for rollback.

- Source checkpoint SHA-256: `5639a917d10f1df1e23c44d301aa5f3389b7855e1c654a9a3fd86cfe9f5c39fc`.
- ONNX SHA-256: `4beae0218b285a1c89898faf60526908e0fa025a4c0dec1ad84b0e9fb7d4089b`.
- Export and source predictions matched on all 347 internal test images.
- Maximum absolute probability difference: 0.000003994.
- Regression suite: 30 passed, 2 skipped (32 total).

Export parity is not evidence of clinical accuracy.

## Known limitations

| Internal pipeline measure | Result |
| --- | --- |
| Overall correct | 248/347 (71.47%) |
| Acne | 102/135 |
| Eczema | 94/124 |
| Fungal Infection | 0/7 |
| Psoriasis | 0/6 |
| Normal Skin | 52/75 |
| Normal images assigned disease | 9/75 |
| Disease images assigned normal | 5/272 |

These provisional-label, development-source samples are small and imbalanced,
not an independent clinical evaluation. Fungal and psoriasis performance is
inadequate. False normal predictions remain. Do not use this demo to diagnose,
confirm healthy skin or rule out disease, or advertise these figures as clinical
or 11-condition accuracy. A subsequent normal-focused training run was not
selected; this release retains the original local trial checkpoint.

The owner confirmed data permission on 3 October 2026; it was not independently
verified. Metadata `approved_for_deployment: false` means formal validation
approval has not been achieved. `deployment_authorization` separately records
the owner's request for a public experimental demo. No training images, private
reports, feedback records or credentials are published with this release.

## Deployment and rollback

The existing Render service starts `gradio_app.py`. Its startup log should report
the pinned ONNX SHA above and five classes. For rollback, redeploy the previous
commit, `f79c7317`, or restore its runtime manifest, labels and metadata together.
The retained original TFLite model SHA is
`3d27f2e4d0fc7f123a455d432a5d8087d6d7028616a35174b4d052f8b4f51962`.
Never mix a model with another checkpoint's labels or preprocessing.
