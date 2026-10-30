"""Simple Gradio UI for the AI Skin Disease Detection prototype."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from uuid import uuid4

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
from upload_preparation import prepare_upload
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
ANALYSIS_LOCK = Lock()


def analysis_stage(message):
    return (
        '<section class="analysis-card" role="status" aria-live="polite" aria-atomic="true">'
        '<p>ANALYSIS IN PROGRESS</p>'
        f'<h3>{message}</h3>'
        '<p>Please keep this page open. Your result will appear here.</p></section>'
    )


def analyze(image_path):
    if not image_path:
        yield "Please upload a skin image.", None, None, gr.update(visible=False)
        return
    if not ANALYSIS_LOCK.acquire(blocking=False):
        yield "The server is busy. Please try again shortly.", None, None, gr.update(visible=False)
        return
    started = perf_counter()
    request_id = uuid4().hex
    try:
        # preprocess=False avoids Gradio decoding a full-resolution phone image
        # before our bounded decoder can inspect it.
        if isinstance(image_path, dict):
            image_path = image_path.get("path")
        image_path = Path(image_path)
        yield analysis_stage("Preparing your photo"), None, None, gr.update(visible=False)
        with TemporaryDirectory(prefix="skinscanix-") as temporary:
            prepared = prepare_upload(image_path, Path(temporary) / "input.png")
            print(f"analysis {request_id}: prepared in {perf_counter() - started:.2f}s", flush=True)
            yield analysis_stage("Checking image quality"), None, None, gr.update(visible=False)
            quality = check_image(str(prepared))
            yield analysis_stage("Analysing the skin image"), None, None, gr.update(visible=False)
            result = predictor.predict(str(prepared))
            print(f"analysis {request_id}: predicted in {perf_counter() - started:.2f}s", flush=True)
            yield analysis_stage("Creating your PDF report"), None, None, gr.update(visible=False)
            report_path = REPORT_DIR / f"{request_id}_report.pdf"
            create_report(prepared, result, quality, report_path)

        elapsed = perf_counter() - started
        completed = (
            '<div role="status" aria-live="polite" style="font-size:12px;'
            'line-height:1.5;margin-bottom:8px;color:var(--body-text-color-subdued)">'
            f'<strong>Analysis complete · {elapsed:.1f}s</strong><br>'
            'Photo prepared &#10003; &nbsp; Quality checked &#10003; &nbsp; '
            'Analysed &#10003; &nbsp; PDF created &#10003;</div>'
        )
        text = completed + render_result(result, quality)
        feedback_context = {"image_path": str(image_path), "result": result}
        yield text, str(report_path), feedback_context, gr.update(visible=True)
    except Exception as exc:
        print(f"Analysis failed: {type(exc).__name__}: {exc}", flush=True)
        if isinstance(exc, ValueError):
            import html
            yield html.escape(str(exc)), None, None, gr.update(visible=False)
            return
        message = (
            "<section class=\"analysis-card\"><h3>Analysis could not finish</h3>"
            "<p>Please try another clear image. If this keeps happening, refresh "
            "the page and try again later.</p></section>"
        )
        yield message, None, None, gr.update(visible=False)
    finally:
        ANALYSIS_LOCK.release()
        print(f"analysis {request_id}: finished in {perf_counter() - started:.2f}s", flush=True)


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


UI_LAUNCH_KWARGS = {"css": CSS, "head": THEME_HEAD, "show_error": False,
    "max_file_size": "10mb", "theme": gr.themes.Base(
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
            gr.HTML('''<div class="section-heading"><h2>Skin image</h2>
              <button popovertarget="photo-guide" type="button"
                style="border:0;background:transparent;color:var(--body-text-color-subdued);font:inherit;font-size:13px;cursor:pointer;text-decoration:underline">Photo guide</button></div>
              <div id="photo-guide" popover style="position:fixed;inset:50% auto auto 50%;transform:translate(-50%,-50%);margin:0;width:min(420px,calc(100vw - 32px));max-height:80dvh;overflow:auto;box-sizing:border-box;padding:24px;border:1px solid var(--border-color-primary);border-radius:16px;background:var(--background-fill-primary);color:var(--body-text-color);box-shadow:0 16px 60px #0006">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:16px">
                  <h3 style="margin:0">Photo &amp; camera guide</h3>
                  <button type="button" popovertarget="photo-guide" popovertargetaction="hide" aria-label="Close photo guide"
                    style="background:transparent;border:0;color:inherit;font-size:24px;cursor:pointer">&#215;</button>
                </div>
                <p><strong>Centre the skin area.</strong> Fill most of the frame and leave a little surrounding skin visible.</p>
                <p><strong>Keep it sharp.</strong> Hold still and move only as close as your camera can focus. Avoid digital zoom and filters.</p>
                <p><strong>Use even light.</strong> Avoid glare and strong shadows.</p>
                <p><strong>Capture or upload.</strong> Select the camera icon, or take a photo with your phone and upload it here. Avoid identifiable details where possible.</p>
                <small>Close with &#215;, Escape, or a click outside this guide.</small>
              </div>''')
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
        queue=True,
        concurrency_limit=1,
        concurrency_id="skin-analysis",
        show_progress="hidden",
        stream_every=0.1,
        preprocess=False,
    )
    image.change(
        reset_analysis,
        outputs=[result, report_file, feedback_context, feedback_group],
        queue=False,
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
    gr.HTML('<aside class="safety-banner" aria-label="Educational use notice"><strong>Educational prototype.</strong> '
            'Results may be incorrect and do not replace a dermatologist’s assessment.</aside>', elem_id="educational-notice")
    gr.HTML(f'<footer class="site-footer"><span><strong>{BRAND_NAME}</strong> · AI Skin Disease Detection</span>'
            '<span>Research & education · Not for medical diagnosis</span></footer>')


demo.queue(max_size=3, default_concurrency_limit=1)

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
