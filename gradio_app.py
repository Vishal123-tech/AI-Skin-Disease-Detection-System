"""Simple Gradio UI for the AI Skin Disease Detection prototype."""

import os
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import (
    ACNE_LABELS_PATH,
    ACNE_MODEL_PATH,
    GATE_LABELS_PATH,
    GATE_MODEL_PATH,
    GEMINI_API_KEY,
    LABELS_PATH,
    MODEL_PATH,
    REPORT_DIR,
)
from predictor import SkinPredictor
from quality import check_image
from report import create_report
from feedback import (
    available_feedback_labels,
    get_feedback_summary,
    record_feedback,
    trigger_self_training,
)


predictor = SkinPredictor(
    MODEL_PATH,
    LABELS_PATH,
    acne_model_path=ACNE_MODEL_PATH,
    acne_labels_path=ACNE_LABELS_PATH,
    gate_model_path=GATE_MODEL_PATH,
    gate_labels_path=GATE_LABELS_PATH,
    gemini_api_key=GEMINI_API_KEY,
)
PORT = int(os.environ.get("PORT", 7860))


def analyze(image_path):
    if not image_path:
        return "Please upload a skin image.", None, None, gr.update(visible=False)

    image_path = Path(image_path)
    quality = check_image(str(image_path))
    result = predictor.predict(str(image_path))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"{stamp}_report.pdf"
    create_report(image_path, result, quality, report_path)

    confidence = (
        f"{result['confidence']:.1%}"
        if result["status"] != "demo"
        else "Unavailable"
    )
    category = result.get("category", "lesion")
    top_suggestion = ""
    if category == "uncertain" and result.get("top3"):
        top_condition, top_probability = result["top3"][0]
        top_suggestion = (
            f"\n**Top possible condition (not confirmed):** "
            f"{top_condition} ({top_probability:.1%})"
        )

    if category == "normal":
        category_header = "🟢 **Classification:** Normal Healthy Skin"
    elif category == "non_skin":
        category_header = "🟠 **Classification:** Not a Skin Image"
    elif category == "uncertain":
        category_header = "🟠 **Classification:** Uncertain Input"
    elif category == "non_acne":
        category_header = "🟢 **Classification:** No Strong Acne Pattern"
    elif category == "lesion":
        category_header = "🔴 **Classification:** Potential Skin Lesion Detected"
    else:
        category_header = "ℹ️ **Classification:** Demo Mode"

    quality_notice = "" if quality.ok else f"\n\n> ⚠️ **Quality Warning:** {quality.message}"

    text = f"""### {category_header}

**Result:** {result['label']}
**Confidence:** {confidence}{top_suggestion}

---
**Image Quality Metrics:**
- **Quality Status:** {quality.message}
- **Blur Score:** {quality.blur_score:.1f}
- **Brightness:** {quality.brightness:.1f}{quality_notice}

> **Disclaimer:** Educational screening prototype only. This result is not a medical diagnosis. Consult a qualified healthcare professional or dermatologist."""
    feedback_context = {"image_path": str(image_path), "result": result}
    return text, str(report_path), feedback_context, gr.update(visible=True)


def submit_feedback(feedback_context, rating, correct_label, comment, consent):
    if not feedback_context:
        return "Please analyze an image before sending feedback."
    try:
        return record_feedback(
            feedback_context["image_path"],
            feedback_context["result"],
            rating,
            correct_label,
            comment,
            consent,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        return f"⚠️ Feedback was not saved: {exc}"


def refresh_stats_ui() -> str:
    stats = get_feedback_summary()
    return (
        f"**Feedback Collected:** {stats['total_feedback']} items "
        f"({stats['correct_count']} verified, {stats['wrong_count']} corrected) | "
        f"**Pending Retraining:** {stats['unlearned_count']} | "
        f"**Last Model Update:** `{stats['last_trained']}`"
    )


def run_self_training_ui() -> str:
    try:
        res = trigger_self_training(epochs=5, min_samples=1)
        if res.get("success"):
            predictor.reload_models()
            classes_str = ", ".join(res.get("classes", []))
            return (
                f"### 🚀 Self-Training Completed Successfully!\n\n"
                f"- **Summary:** {res.get('message')}\n"
                f"- **Supported Classes ({len(res.get('classes', []))}):** {classes_str}\n"
                f"- **Validation Accuracy:** {res.get('val_accuracy', 0):.1%}\n"
                f"- **Active Model Engine:** `{predictor.backend}`\n"
                f"- **Backup Saved:** `{res.get('backup_path') or 'Created in models/backups/'}`\n\n"
                f"> **Live Update:** The updated model weights were hot-reloaded into memory."
            )
        else:
            return f"⚠️ **Could not train:** {res.get('message')}"
    except Exception as exc:
        return f"❌ **Error during self-training:** {exc}"


with gr.Blocks(title="AI Skin Disease Detection") as demo:
    gr.Markdown(
        "# 🩺 AI Skin Disease Detection\n"
        "Upload a skin image to run the classifier (detecting normal skin, "
        "non-skin images, or skin lesions) and generate a PDF report."
    )
    gr.Markdown(
        "> **Important:** This is an educational prototype, not a medical diagnostic device."
    )
    with gr.Row():
        image = gr.Image(type="filepath", label="Upload skin image")
        result = gr.Markdown("Your result will appear here.")
    analyze_button = gr.Button("Analyze Image", variant="primary")
    report_file = gr.File(label="Download PDF report")
    feedback_context = gr.State(None)
    with gr.Group(visible=False) as feedback_group:
        gr.Markdown(
            "### Help improve this prototype\n"
            "Was the result correct? Your feedback is saved for continuous learning; "
            "the model adapts safely using experience replay to prevent forgetting."
        )
        feedback_rating = gr.Radio(
            ["Correct", "Wrong", "Not sure"],
            label="Prediction feedback",
            value="Correct",
        )
        correction_label = gr.Dropdown(
            choices=available_feedback_labels(),
            label="What is the correct condition?",
            visible=False,
        )
        feedback_comment = gr.Textbox(
            label="Optional note",
            placeholder="Example: dermatologist-confirmed acne; leave blank if unsure.",
            lines=2,
        )
        feedback_consent = gr.Checkbox(
            label="I consent to storing this image for supervised model improvement.",
            value=False,
        )
        feedback_button = gr.Button("Save Feedback")
        feedback_status = gr.Markdown()

    with gr.Accordion("🧠 Continuous Self-Training (Learn from Feedback)", open=True):
        gr.Markdown(
            "Fine-tune and adapt the AI model on confirmed user feedback. "
            "The engine combines feedback images with baseline references (experience replay) "
            "to prevent catastrophic forgetting, trains with PyTorch, and hot-reloads into memory."
        )
        stats_md = gr.Markdown(value=refresh_stats_ui)
        with gr.Row():
            train_btn = gr.Button("⚡ Retrain Model on Feedback Now", variant="primary")
            refresh_btn = gr.Button("🔄 Refresh Feedback Stats")
        train_status = gr.Markdown()

    analyze_button.click(
        analyze,
        inputs=image,
        outputs=[result, report_file, feedback_context, feedback_group],
    )
    feedback_rating.change(
        lambda rating: gr.update(visible=rating == "Wrong"),
        inputs=feedback_rating,
        outputs=correction_label,
    )
    feedback_button.click(
        submit_feedback,
        inputs=[feedback_context, feedback_rating, correction_label, feedback_comment, feedback_consent],
        outputs=feedback_status,
    ).then(refresh_stats_ui, outputs=stats_md)

    train_btn.click(
        run_self_training_ui,
        outputs=train_status,
    ).then(refresh_stats_ui, outputs=stats_md)
    refresh_btn.click(refresh_stats_ui, outputs=stats_md)


if __name__ == "__main__":
    # On cloud platforms (Render, HF Spaces), the platform provides the public URL.
    # Only enable share=True locally.
    is_cloud = bool(os.environ.get("RENDER") or os.environ.get("SPACE_ID"))
    _, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=(not is_cloud),
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_SHARE_URL={share_url}", flush=True)
    if share_url:
        Path("public_url.txt").write_text(share_url, encoding="utf-8")
