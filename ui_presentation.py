"""Presentation only: do not change model decisions, thresholds, or report data."""
from html import escape
from pathlib import Path

CSS = (Path(__file__).parent / "static" / "dermaai.css").read_text(encoding="utf-8")
THEME_HEAD = '<style id="dermaai-theme-tokens">' + (Path(__file__).parent / "static" / "theme-tokens.css").read_text(encoding="utf-8") + '</style>'

EMPTY_RESULT = '''<section class="result-empty" aria-label="Analysis result" aria-live="polite">
<div class="empty-symbol" aria-hidden="true">◎</div>
<span class="eyebrow">READY WHEN YOU ARE</span>
<h3>Your result will appear here</h3>
<p>Upload a clear photo and select <strong>Analyze image</strong>.
You’ll see the model’s assessment, image quality, and a downloadable report.</p>
<div class="empty-note">Results can be uncertain or incorrect. A photo alone cannot confirm a diagnosis.</div>
</section>'''


def render_result(result, quality):
    category = result.get("category", "lesion")
    titles = {
        "lesion": ("review", "Potential skin condition"),
        "normal": ("neutral", "No condition flagged by the model"),
        "non_skin": ("warning", "Skin image not confirmed"),
        "uncertain": ("warning", "Uncertain result"),
        "non_acne": ("neutral", "No strong acne pattern"),
    }
    tone, title = titles.get(category, ("neutral", "Demo mode"))
    confidence = result.get("confidence")
    score = f"{confidence:.1%}" if result.get("status") != "demo" and isinstance(confidence, (int, float)) else "Unavailable"
    if result.get("uncertainty_reason") == "model_disagreement":
        score = "Unavailable — models disagree"
    suggestion = ""
    if category == "uncertain" and result.get("top3"):
        condition, probability = result["top3"][0]
        suggestion = f'<p class="suggestion">Top possible condition (not confirmed): <strong>{escape(str(condition))}</strong> ({probability:.1%})</p>'
    quality_label = "Acceptable image quality" if quality.ok else "Photo needs attention"
    return f'''<section class="result-card" aria-live="polite">
      <div class="status-pill {tone}"><span aria-hidden="true">●</span> {title}</div>
      <div class="eyebrow result-label">MODEL ASSESSMENT · NOT A DIAGNOSIS</div>
      <h3 class="condition-name">{escape(str(result['label']))}</h3>
      <div class="score-row"><span>Model confidence</span><strong>{escape(score)}</strong></div>
      <p class="fine-print">This score is not the probability that you have this condition.</p>
      {suggestion}
      <div class="quality-section"><h4>{quality_label}</h4><p>{escape(str(quality.message))}</p>
        <div class="quality-grid"><div><span>Sharpness score</span><strong>{quality.blur_score:.1f}</strong></div>
        <div><span>Brightness</span><strong>{quality.brightness:.1f}</strong></div></div>
      </div>
      <p class="result-disclaimer">Do not use this result to diagnose or rule out a condition.
      Consult a qualified dermatologist for assessment.</p>
    </section>'''
