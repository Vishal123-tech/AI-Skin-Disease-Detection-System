# Render memory fix (2026-10-27)

Render Events showed repeated failures exceeding the free instance's 512 MB
limit. The deployed commit was 2ad52fa; c489980 had not been deployed.
Disabling the queue alone is not a solution to an out-of-memory restart.

Changes:
- Bypass Gradio's full-image preprocessing for the analysis callback.
- Inspect dimensions before decoding; use JPEG decoder downsampling and cap
  the working image at 1280 pixels on its longest side.
- Reject images over 10 MB, JPEGs over 50 MP, and other formats over 4 MP.
- Serialize analyses with a three-item queue and a defensive nonblocking lock.
- Do not load the acne specialist when both of its optional modes are disabled.
- Skip legacy inference when the broad model already meets its existing 0.65
  fallback threshold. Keep the legacy model available for low-confidence cases.
- Use unique report names and log stage timings without image contents.

Validation: 14 upload/runtime tests passed. Three simultaneous synthetic 48 MP
JPEG uploads via the Gradio client returned PDFs in 7.11 seconds total locally.
Windows server peak working set was 379539456 bytes (about 362 MiB).
This is not a Linux memory guarantee or clinical accuracy evaluation. Large-image
resizing can change quality scores and predictions; small RGB PNG pixels are
preserved, and model weights/decision thresholds are unchanged.

Run the HTTP smoke check with:
`python scripts/check_hosted_analysis.py http://127.0.0.1:7862/`

After deployment, repeat the check against Render and inspect Events for new OOM
failures. Free-tier cold starts remain. If memory failures persist, profile further
or use a larger instance; do not silently remove disease models or alter thresholds.
