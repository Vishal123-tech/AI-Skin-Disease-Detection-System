"""
gradio_app.py — AI Skin Disease Detection (Gradio UI)
──────────────────────────────────────────────────────
Supports:
  • Gemini Vision API backend  (30+ diseases, non-skin rejection)
  • Local TFLite / PyTorch      (8 original classes, entropy gate)
"""

import os
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import GEMINI_API_KEY, LABELS_PATH, MODEL_PATH, OLLAMA_MODEL, OLLAMA_URL, REPORT_DIR, UPLOAD_DIR
from predictor import DISEASE_INFO, SkinPredictor
from quality import check_image
from report import create_report

predictor = SkinPredictor(MODEL_PATH, LABELS_PATH, gemini_api_key=GEMINI_API_KEY,
                          ollama_model=OLLAMA_MODEL, ollama_url=OLLAMA_URL)
PORT = int(os.environ.get("PORT", 7860))

# ── Category colour mapping for display ───────────────────────────────────────
CATEGORY_STYLES = {
    "lesion":   ("🔴", "Potential Skin Condition Detected"),
    "normal":   ("🟢", "Normal Healthy Skin"),
    "non_skin": ("🟠", "Not a Skin Image"),
    "uncertain":("🟡", "Uncertain — Please Try Again"),
    "demo":     ("⚪", "Demo Mode"),
}


def _format_top3(top3: list) -> str:
    """Format the top-3 predictions as a markdown table."""
    if not top3:
        return ""
    rows = ["| # | Condition | Confidence |", "|---|-----------|-----------|"]
    for i, (cond, conf) in enumerate(top3[:3], 1):
        info = DISEASE_INFO.get(cond, {})
        emoji = info.get("emoji", "🩺")
        bar_filled = round(conf * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        rows.append(f"| {i} | {emoji} {cond.replace('_',' ')} | `{bar}` {conf:.1%} |")
    return "\n".join(rows)


def analyze(image_path):
    if not image_path:
        return (
            "### ⬆️ Please upload a skin image to begin analysis.",
            None,
        )

    image_path = Path(image_path)
    quality = check_image(str(image_path))
    result = predictor.predict(str(image_path))

    # Save a copy to uploads
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = UPLOAD_DIR / f"{stamp}_{image_path.name}"
    try:
        import shutil
        shutil.copy2(image_path, saved)
    except Exception:
        saved = image_path

    # Generate PDF report
    report_path = REPORT_DIR / f"{stamp}_report.pdf"
    create_report(saved, result, quality, report_path)

    # ── Build output markdown ─────────────────────────────────────────────────
    category = result.get("category", "lesion")
    icon, header_text = CATEGORY_STYLES.get(category, ("🩺", "Analysis Result"))

    confidence = result["confidence"]
    conf_str = f"{confidence:.1%}" if result["status"] != "demo" else "Unavailable (demo mode)"

    disease_info = result.get("disease_info", {})
    description = disease_info.get("description", "")
    severity = disease_info.get("severity", "")

    top3_md = _format_top3(result.get("top3", []))
    notes = result.get("notes", "")
    quality_notice = "" if quality.ok else f"\n> ⚠️ **Quality Warning:** {quality.message}\n"

    backend_badge = f"`{result['status'].upper()}`"

    # ── Non-skin rejection ────────────────────────────────────────────────────
    if category == "non_skin":
        text = f"""## {icon} {header_text}

**This image does not appear to contain human skin.**

The system detected no skin-tone pixels or the AI identified the image as a non-medical photo (e.g. object, animal, food, scenery).

### What to do:
- 📸 Take a clear, close-up photo of the **skin area** you want analyzed
- 💡 Ensure good lighting (no harsh shadows)
- 🎯 Make sure the skin fills most of the frame
{quality_notice}
> **Disclaimer:** Educational prototype only. Not a medical diagnostic device."""
        return text, str(report_path)

    # ── Uncertain ─────────────────────────────────────────────────────────────
    if category == "uncertain":
        text = f"""## {icon} {header_text}

**Confidence is too low to make a reliable prediction ({conf_str}).**

{description}

### Tips for better results:
- 📸 Move the camera closer to the skin area
- 💡 Use bright, even, indirect lighting
- 🙆 Keep the camera steady (avoid blur)
- 🔍 Ensure the skin lesion/area is clearly visible and centered
{quality_notice}
> **Disclaimer:** Educational prototype only. Not a medical diagnostic device."""
        return text, str(report_path)

    # ── Normal skin ───────────────────────────────────────────────────────────
    if category == "normal":
        text = f"""## {icon} {header_text}

**Result:** {result["label"]}  
**Confidence:** {conf_str} | **Backend:** {backend_badge}

{description}

---
### 📊 Image Quality
| Metric | Value |
|--------|-------|
| Quality | {quality.message} |
| Blur Score | {quality.blur_score:.1f} |
| Brightness | {quality.brightness:.1f} |
{quality_notice}
> **Disclaimer:** Educational prototype only. Not a medical diagnostic device. Consult a dermatologist for any skin concerns."""
        return text, str(report_path)

    # ── Skin lesion / disease ─────────────────────────────────────────────────
    severity_icon = "🚨" if "CRITICAL" in severity or "High" in severity else ("⚠️" if "Moderate" in severity else "ℹ️")

    text = f"""## {icon} {header_text}

**Detected:** {result["label"]}  
**Confidence:** {conf_str} | **Backend:** {backend_badge}

---
### 📋 About This Condition
{description}

{severity_icon} **Severity / Action:** {severity}

"""

    if top3_md:
        text += f"""---
### 📊 Top Predictions
{top3_md}

"""

    if notes:
        text += f"""---
### 🔬 AI Observation
> {notes}

"""

    text += f"""---
### 📸 Image Quality
| Metric | Value |
|--------|-------|
| Quality | {quality.message} |
| Blur Score | {quality.blur_score:.1f} |
| Brightness | {quality.brightness:.1f} |
{quality_notice}
---
> **⚕️ Disclaimer:** This is an **educational AI screening prototype** — not a medical diagnostic device. Always consult a qualified dermatologist or healthcare professional for diagnosis and treatment."""

    return text, str(report_path)


# ─── Gradio Interface ─────────────────────────────────────────────────────────

backend_label = (
    "🤖 Backend: Gemini Vision API (30+ diseases)"
    if predictor.backend == "gemini"
    else f"🤖 Backend: Local Model ({len(predictor.labels)} classes)"
    if predictor.backend
    else "⚠️ Demo Mode (no model loaded)"
)

with gr.Blocks(
    title="AI Skin Disease Detection",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
    .output-markdown { font-size: 15px; line-height: 1.7; }
    .status-bar { background: #1e293b; color: #94a3b8; padding: 8px 16px; border-radius: 8px; font-size: 13px; margin-bottom: 12px; }
    """,
) as demo:
    gr.Markdown("""# 🩺 AI Skin Disease Detection
    Upload a photo of **skin** to get an AI-powered screening result with disease information, severity assessment, and a downloadable PDF report.""")

    gr.HTML(f'<div class="status-bar">{backend_label}</div>')

    gr.Markdown("> ⚕️ **Important:** This is an educational prototype — **not a medical diagnostic device.** Always consult a qualified dermatologist.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="📷 Upload Skin Image",
                height=320,
            )
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

            gr.Markdown("""**📌 Tips for best results:**
- Photo should show **skin clearly** (not objects, food, animals)
- Good lighting — avoid harsh shadows or glare
- Close-up of the affected area
- Steady camera (avoid blur)""")

        with gr.Column(scale=2):
            result_md = gr.Markdown(
                "### ⬆️ Upload a skin image and click **Analyze Image** to begin.",
                elem_classes=["output-markdown"],
            )
            report_file = gr.File(label="📄 Download PDF Report")

    analyze_btn.click(
        fn=analyze,
        inputs=image_input,
        outputs=[result_md, report_file],
    )

    gr.Markdown("""---
### 🔬 Detectable Conditions
The system can identify **30+ skin conditions** including:

| Category | Conditions |
|----------|-----------|
| **Skin Cancer** | Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma, Actinic Keratoses |
| **Common Infections** | Ringworm, Impetigo, Cellulitis, Chickenpox, Shingles, Warts, Scabies |
| **Inflammatory** | Eczema, Psoriasis, Contact Dermatitis, Rosacea, Urticaria (Hives) |
| **Acne & Follicular** | Acne Vulgaris (Pimples), Folliculitis, Seborrheic Dermatitis |
| **Pigmentation** | Vitiligo, Melanocytic Nevi (Moles), Tinea Versicolor |
| **Other** | Dermatofibroma, Vascular Lesions, Cold Sores, Sunburn, Molluscum Contagiosum |
| **Non-Skin** | Random images, objects, animals — **automatically rejected** ✅ |""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
