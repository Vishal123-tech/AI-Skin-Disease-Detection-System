from pathlib import Path
from datetime import datetime
import gradio as gr

from config import MODEL_PATH, LABELS_PATH, UPLOAD_DIR, REPORT_DIR
from predictor import SkinPredictor
from quality import check_image
from report import create_report

predictor = SkinPredictor(MODEL_PATH, LABELS_PATH)

def analyze(image_path):
    if not image_path:
        return "Please upload a skin image.", None
    image_path = Path(image_path)
    quality = check_image(str(image_path))
    if not quality.ok:
        return f"### Image quality needs improvement\n\n{quality.message}", None
    result = predictor.predict(str(image_path))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORT_DIR / f'{stamp}_report.pdf'
    create_report(image_path, result, quality, report_path)
    confidence = f"{result['confidence']:.1%}" if result['status'] == 'model' else 'Unavailable'
    text = f"""### Prediction: {result['label']}

**Confidence:** {confidence}

**Image quality:** {quality.message}

**Blur score:** {quality.blur_score:.1f}  
**Brightness:** {quality.brightness:.1f}

> Educational screening prototype only. This result is not a diagnosis. Consult a qualified healthcare professional."""
    return text, str(report_path)

with gr.Blocks(title='AI Skin Disease Detection') as demo:
    gr.Markdown('# 🩺 AI Skin Disease Detection\nUpload a skin image to run the TensorFlow Lite classifier and generate a PDF report.')
    gr.Markdown('> **Important:** This is an educational prototype, not a medical diagnostic device.')
    with gr.Row():
        image = gr.Image(type='filepath', label='Upload skin image')
        result = gr.Markdown('Your result will appear here.')
    analyze_button = gr.Button('Analyze Image', variant='primary')
    report_file = gr.File(label='Download PDF report')
    analyze_button.click(analyze, inputs=image, outputs=[result, report_file])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
