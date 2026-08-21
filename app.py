import argparse
from datetime import datetime
from pathlib import Path
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

from config import MODEL_PATH, LABELS_PATH, UPLOAD_DIR, REPORT_DIR, MAX_UPLOAD_MB
from predictor import SkinPredictor
from quality import check_image
from report import create_report

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
predictor = SkinPredictor(MODEL_PATH, LABELS_PATH)

HTML = """<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>AI Skin Screening Prototype</title>
<style>
body{font-family:Arial,sans-serif;background:#f4f7fb;color:#172b4d;max-width:900px;margin:40px auto;padding:0 20px}
.panel{background:white;border-radius:16px;padding:28px;box-shadow:0 4px 20px #102a4318}
h1{margin-top:0;color:#102a43}
.notice{background:#fff7ed;border-left:4px solid #f28c28;padding:12px;margin:18px 0}
.result-lesion{background:#102a43;color:white;padding:18px;border-radius:12px;margin-top:20px}
.result-normal{background:#1b4332;color:white;padding:18px;border-radius:12px;margin-top:20px}
.result-non_skin{background:#7f4f24;color:white;padding:18px;border-radius:12px;margin-top:20px}
.result-uncertain{background:#6c584c;color:white;padding:18px;border-radius:12px;margin-top:20px}
.result-demo{background:#334155;color:white;padding:18px;border-radius:12px;margin-top:20px}
button{background:#1976d2;color:white;border:0;border-radius:8px;padding:12px 20px;font-size:16px;cursor:pointer}
input{margin:18px 0}
.muted{color:#52606d}
.error{color:#c53030}
.badge{display:inline-block;padding:4px 8px;border-radius:4px;font-weight:bold;font-size:13px;background:rgba(255,255,255,0.2)}
</style>
</head>
<body>
<div class='panel'>
  <h1>AI Skin Disease Detection</h1>
  <p class='muted'>Capture/upload an image, check quality, classify skin health/lesion, and generate a PDF report.</p>
  <div class='notice'><b>Important:</b> This educational prototype is not a medical diagnostic device.</div>
  
  <form method='post' enctype='multipart/form-data'>
    <input type='file' name='image' accept='image/*' capture='environment' required><br>
    <button type='submit'>Analyze image</button>
  </form>

  {% if error %}
    <p class='error'><b>{{error}}</b></p>
  {% endif %}

  {% if result %}
    <div class='result-{{result.category}}'>
      <span class='badge'>{{result.category | upper}}</span>
      <h2>{{result.label}}</h2>
      <p><b>Confidence:</b> {{formatted_confidence}}</p>
      <p><b>Quality Assessment:</b> {{quality.message}}</p>
      {% if report %}
        <p><a style='color:#9fe7e7;font-weight:bold' href='/reports/{{report}}'>📄 Download PDF report</a></p>
      {% endif %}
    </div>
  {% endif %}
</div>
</body>
</html>"""

@app.route('/', methods=['GET','POST'])
def index():
    error = None; result = None; quality = None; report_name = None; formatted_confidence = None
    if request.method == 'POST':
        upload = request.files.get('image')
        if not upload or not upload.filename:
            error = 'Please select an image.'
        else:
            name = secure_filename(upload.filename)
            stem = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_path = UPLOAD_DIR / f'{stem}_{name}'
            upload.save(image_path)
            quality = check_image(str(image_path))
            result = predictor.predict(str(image_path))
            report_path = REPORT_DIR / f'{stem}_report.pdf'
            create_report(image_path, result, quality, report_path)
            report_name = report_path.name
            formatted_confidence = f"{result['confidence']:.1%}" if result['status'] == 'model' else 'Unavailable (demo mode)'
    return render_template_string(HTML, error=error, result=result, quality=quality, report=report_name, formatted_confidence=formatted_confidence)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    upload = request.files.get('image')
    if not upload or not upload.filename:
        return {'error': 'No image uploaded'}, 400
    name = secure_filename(upload.filename)
    stem = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = UPLOAD_DIR / f'{stem}_{name}'
    upload.save(image_path)
    quality = check_image(str(image_path))
    if not quality.ok:
        return {'error': quality.message}, 400
    result = predictor.predict(str(image_path))
    report_path = REPORT_DIR / f'{stem}_report.pdf'
    create_report(image_path, result, quality, report_path)
    report_url = f"{request.host_url}reports/{report_path.name}"
    return {
        'status': 'success',
        'label': result['label'],
        'category': result['category'],
        'confidence': f"{result['confidence']:.1%}" if result['status'] == 'model' else 'Unavailable (demo mode)',
        'quality': quality.message,
        'report_url': report_url
    }

@app.route('/reports/<path:name>')
def reports(name):
    return send_from_directory(REPORT_DIR, name, as_attachment=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--camera', action='store_true', help='Reserved for Picamera2 capture integration')
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
