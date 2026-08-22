"""
app.py — Flask Web App for AI Skin Disease Detection
──────────────────────────────────────────────────────
Handles:
  • Image upload → predict → quality check → PDF report
  • Displays non-skin rejection, disease info, top-3 predictions
  • REST API endpoint at /api/predict
"""

import argparse
from datetime import datetime
from pathlib import Path

from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

from config import GEMINI_API_KEY, LABELS_PATH, MAX_UPLOAD_MB, MODEL_PATH, OLLAMA_MODEL, OLLAMA_URL, REPORT_DIR, UPLOAD_DIR
from predictor import SkinPredictor
from quality import check_image
from report import create_report

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
predictor = SkinPredictor(MODEL_PATH, LABELS_PATH, gemini_api_key=GEMINI_API_KEY,
                          ollama_model=OLLAMA_MODEL, ollama_url=OLLAMA_URL)

# ─── HTML Template ─────────────────────────────────────────────────────────────
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Skin Disease Detection</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --primary: #3b82f6; --primary-dark: #2563eb;
    --text: #f1f5f9; --muted: #94a3b8; --border: #334155;
    --green: #059669; --red: #dc2626; --orange: #d97706;
    --yellow: #ca8a04; --purple: #7c3aed;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 20px; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: var(--muted); margin-bottom: 24px; font-size: 0.95rem; }
  .card { background: var(--surface); border-radius: 16px; padding: 28px; margin-bottom: 20px; border: 1px solid var(--border); }
  .notice { background: #1c1917; border-left: 4px solid var(--orange); padding: 12px 16px; border-radius: 0 8px 8px 0; color: #fbbf24; font-size: 0.9rem; margin-bottom: 20px; }
  .upload-zone { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  input[type=file] { color: var(--muted); background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px; flex: 1; }
  .btn { background: var(--primary); color: white; border: none; border-radius: 8px; padding: 12px 24px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; white-space: nowrap; }
  .btn:hover { background: var(--primary-dark); }
  .error { background: #450a0a; border: 1px solid var(--red); border-radius: 10px; padding: 14px; color: #fca5a5; margin-top: 16px; }

  /* Result boxes */
  .result { border-radius: 14px; padding: 24px; margin-top: 20px; border: 1px solid rgba(255,255,255,0.1); }
  .result-lesion   { background: linear-gradient(135deg, #1e1b4b, #312e81); }
  .result-normal   { background: linear-gradient(135deg, #052e16, #14532d); }
  .result-non_skin { background: linear-gradient(135deg, #1c1100, #431407); }
  .result-uncertain{ background: linear-gradient(135deg, #1c1400, #422006); }
  .result-demo     { background: linear-gradient(135deg, #0f172a, #1e293b); }

  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; background: rgba(255,255,255,0.15); margin-bottom: 12px; }
  .result h2 { font-size: 1.4rem; margin: 8px 0 16px; }
  .meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin: 16px 0; }
  .meta-item { background: rgba(0,0,0,0.25); border-radius: 8px; padding: 10px 14px; }
  .meta-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .meta-value { font-size: 15px; font-weight: 600; }

  .description-box { background: rgba(0,0,0,0.2); border-radius: 10px; padding: 14px; margin: 12px 0; font-size: 0.92rem; line-height: 1.6; }
  .severity-box { display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; background: rgba(255,255,255,0.1); margin: 6px 0; }

  .top3 { margin-top: 16px; }
  .top3-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 10px; }
  .top3-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
  .top3-label { flex: 1; font-size: 0.88rem; }
  .top3-bar-bg { flex: 2; background: rgba(0,0,0,0.3); border-radius: 4px; height: 8px; overflow: hidden; }
  .top3-bar-fill { height: 100%; background: rgba(255,255,255,0.5); border-radius: 4px; transition: width 0.5s; }
  .top3-pct { font-size: 0.82rem; color: var(--muted); min-width: 42px; text-align: right; }

  .report-link { display: inline-flex; align-items: center; gap: 8px; margin-top: 16px; color: #93c5fd; font-weight: 600; text-decoration: none; }
  .report-link:hover { text-decoration: underline; }
  .disclaimer { font-size: 0.82rem; color: var(--muted); margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); }
  .quality-warn { background: rgba(202,138,4,0.15); border: 1px solid #ca8a04; border-radius: 8px; padding: 8px 12px; font-size: 0.85rem; color: #fbbf24; margin-top: 10px; }
  .non-skin-tips { list-style: none; margin-top: 12px; }
  .non-skin-tips li { padding: 5px 0; }
  .non-skin-tips li::before { content: "→ "; color: var(--muted); }
</style>
</head>
<body>
<div class="container">

  <div class="card">
    <h1>🩺 AI Skin Disease Detection</h1>
    <p class="subtitle">Upload a skin photo to get an AI-powered screening result with disease information and a PDF report.</p>
    <div class="notice">⚕️ <strong>Educational Prototype:</strong> This system is not a medical diagnostic device. Always consult a qualified dermatologist.</div>

    <form method="post" enctype="multipart/form-data">
      <div class="upload-zone">
        <input type="file" id="image-input" name="image" accept="image/*" capture="environment" required>
        <button class="btn" type="submit">🔍 Analyze Image</button>
      </div>
    </form>

    {% if error %}
    <div class="error">⚠️ {{ error }}</div>
    {% endif %}
  </div>

  {% if result %}
  <div class="result result-{{ result.category }}">
    <span class="badge">{{ result.category | upper }}</span>
    <h2>{{ result.label }}</h2>

    {% if result.category == 'non_skin' %}
    <p style="color: #fca5a5; margin-bottom: 12px;">This image does not appear to contain human skin. The AI rejected it before classification.</p>
    <ul class="non-skin-tips">
      <li>Take a clear, close-up photo of the <strong>skin area</strong> you want analyzed</li>
      <li>Ensure good lighting — avoid harsh shadows or flash glare</li>
      <li>Make sure the skin fills most of the frame</li>
      <li>Keep the camera steady to avoid blur</li>
    </ul>

    {% elif result.category == 'uncertain' %}
    <p style="color: #fde68a; margin-bottom: 12px;">Confidence is too low to make a reliable prediction. Please try a clearer, better-lit photo.</p>

    {% else %}
    <div class="meta-grid">
      <div class="meta-item">
        <div class="meta-label">Confidence</div>
        <div class="meta-value">{{ formatted_confidence }}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Image Quality</div>
        <div class="meta-value">{{ "✅ Good" if quality.ok else "⚠️ Poor" }}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Backend</div>
        <div class="meta-value">{{ result.status | upper }}</div>
      </div>
    </div>

    {% if result.disease_info %}
    <div class="description-box">{{ result.disease_info.description }}</div>
    <div class="severity-box">{{ result.disease_info.severity }}</div>
    {% endif %}

    {% if result.top3 %}
    <div class="top3">
      <div class="top3-title">Top Predictions</div>
      {% for label, prob in result.top3 %}
      <div class="top3-row">
        <div class="top3-label">{{ loop.index }}. {{ label | replace('_', ' ') }}</div>
        <div class="top3-bar-bg"><div class="top3-bar-fill" style="width: {{ (prob * 100) | round | int }}%"></div></div>
        <div class="top3-pct">{{ "%.1f%%" | format(prob * 100) }}</div>
      </div>
      {% endfor %}
    </div>
    {% endif %}
    {% endif %}

    {% if not quality.ok %}
    <div class="quality-warn">⚠️ {{ quality.message }}</div>
    {% endif %}

    {% if report %}
    <a class="report-link" href="/reports/{{ report }}">📄 Download PDF Report</a>
    {% endif %}

    <p class="disclaimer">⚕️ This result is for educational purposes only. It is not a medical diagnosis. Consult a qualified healthcare professional or dermatologist.</p>
  </div>
  {% endif %}

</div>
</body>
</html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    error = result = quality = report_name = formatted_confidence = None

    if request.method == "POST":
        upload = request.files.get("image")
        if not upload or not upload.filename:
            error = "Please select an image file."
        else:
            name = secure_filename(upload.filename)
            stem = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = UPLOAD_DIR / f"{stem}_{name}"
            upload.save(image_path)

            quality = check_image(str(image_path))
            result = predictor.predict(str(image_path))

            report_path = REPORT_DIR / f"{stem}_report.pdf"
            create_report(image_path, result, quality, report_path)
            report_name = report_path.name

            if result["status"] == "demo":
                formatted_confidence = "Unavailable (demo mode)"
            else:
                formatted_confidence = f"{result['confidence']:.1%}"

    return render_template_string(
        HTML,
        error=error,
        result=result,
        quality=quality,
        report=report_name,
        formatted_confidence=formatted_confidence,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return {"error": "No image uploaded"}, 400

    name = secure_filename(upload.filename)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = UPLOAD_DIR / f"{stem}_{name}"
    upload.save(image_path)

    quality = check_image(str(image_path))
    result = predictor.predict(str(image_path))
    report_path = REPORT_DIR / f"{stem}_report.pdf"
    create_report(image_path, result, quality, report_path)

    disease_info = result.get("disease_info", {})
    top3 = [{"condition": c, "confidence": f"{p:.1%}"} for c, p in result.get("top3", [])]

    return {
        "status": "success",
        "category": result["category"],
        "label": result["label"],
        "raw_class": result["raw_class"],
        "confidence": (
            f"{result['confidence']:.1%}" if result["status"] != "demo" else "Unavailable"
        ),
        "disease_description": disease_info.get("description", ""),
        "severity": disease_info.get("severity", ""),
        "top3": top3,
        "notes": result.get("notes", ""),
        "quality": quality.message,
        "quality_ok": quality.ok,
        "report_url": f"{request.host_url}reports/{report_path.name}",
        "backend": result["status"],
    }


@app.route("/reports/<path:name>")
def reports(name):
    return send_from_directory(REPORT_DIR, name, as_attachment=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Skin Disease Detection Flask App")
    parser.add_argument("--camera", action="store_true", help="Reserved for Picamera2 integration")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
