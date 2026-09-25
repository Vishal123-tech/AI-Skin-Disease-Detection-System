"""Simple Gradio UI for the AI Skin Disease Detection prototype."""

import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from branding import BRAND_NAME, website_header
from ui_presentation import CSS, EMPTY_RESULT, THEME_HEAD, render_result

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
from predictor import  SkinPredictor
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

    text = render_result(result, quality)
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
        f"({stats['correct_count']} marked correct, {stats['wrong_count']} corrections) | "
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


UI_LAUNCH_KWARGS = {"css": CSS, "head": THEME_HEAD, "theme": gr.themes.Base(
    primary_hue="orange", neutral_hue="slate", font=["system-ui", "sans-serif"]),
    "footer_links": ["api", "settings"]}


def reset_analysis():
    """An edited upload must never retain another photo's report or feedback."""
    return EMPTY_RESULT, None, None, gr.update(visible=False)


with gr.Blocks(title=f"{BRAND_NAME} | AI Skin Disease Detection", analytics_enabled=False, fill_width=True) as demo:
    gr.HTML(website_header(), elem_id="brand-header")
    gr.Markdown(
        "Upload a clear photo of the affected skin area for AI-assisted analysis "
        "and a downloadable PDF report.", elem_id="intro"
    )
    gr.HTML('<div class="workflow" aria-label="How it works"><span><b>1</b> Upload a photo</span>'
            '<span><b>2</b> Review the analysis</span><span><b>3</b> Save your report</span></div>')
    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="section-heading"><h2>Skin image</h2><span>Upload or use your camera</span></div>')
            image = gr.Image(type="filepath", label="Upload skin image", height=350, elem_id="skin-upload")
            gr.Markdown("**For a better photo** · Use even lighting, keep the camera steady, "
                        "and frame the affected area closely. Avoid identifiable details where possible.", elem_id="photo-tips")
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="section-heading"><h2>Analysis overview</h2><span>AI-assisted · Not diagnostic</span></div>')
            result = gr.HTML(EMPTY_RESULT, elem_id="analysis-result")
    analyze_button = gr.Button("Analyze image", variant="primary", elem_id="analyze-button")
    with gr.Group(elem_id="report-section"):
        gr.HTML('<div class="section-heading"><h2>Your PDF report</h2><span>Available after analysis</span></div>')
        report_file = gr.File(label="Download PDF report", interactive=False, height=85, elem_id="report-download")
    feedback_context = gr.State(None)
    with gr.Group(visible=False, elem_id="feedback-panel") as feedback_group:
        gr.Markdown(
            "### Help improve this prototype\n"
            "Was the result correct? Your feedback is saved for supervised review. "
            "It does not automatically change the active model."
        )
        feedback_rating = gr.Radio(
            ["Correct", "Wrong", "Not sure"],
            label="Prediction feedback",
            value="Not sure",
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

    with gr.Accordion("Project details & feedback review", open=False, elem_id="project-details"):
        gr.Markdown(
            "One-click training is paused after a model regression. Feedback remains saved. "
            "A new model must be trained separately on reviewed data and tested across "
            "all supported classes before activation."
        )
        stats_md = gr.Markdown(value=refresh_stats_ui)
        with gr.Row():
            train_btn = gr.Button("Check retraining status")
            refresh_btn = gr.Button("🔄 Refresh Feedback Stats")
        train_status = gr.Markdown()

    analyze_button.click(
        analyze,
        inputs=image,
        outputs=[result, report_file, feedback_context, feedback_group],
    )
    image.change(reset_analysis, outputs=[result, report_file, feedback_context, feedback_group])
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
    gr.HTML('<aside class="safety-banner" aria-label="Educational use notice"><strong>Educational prototype.</strong> '
            'Results may be incorrect and do not replace a dermatologist’s assessment.</aside>', elem_id="educational-notice")
    gr.HTML(f'<footer class="site-footer"><span><strong>{BRAND_NAME}</strong> · AI Skin Disease Detection</span>'
            '<span>Research & education · Not for medical diagnosis</span></footer>')


if __name__ == "__main__":
    # On cloud platforms (Render, HF Spaces), the platform provides the public URL.
    # Only enable share=True locally.
    is_cloud = bool(os.environ.get("RENDER") or os.environ.get("SPACE_ID"))
    _, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=(not is_cloud),
        **UI_LAUNCH_KWARGS,
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_SHARE_URL={share_url}", flush=True)
    if share_url:
        Path("public_url.txt").write_text(share_url, encoding="utf-8")
