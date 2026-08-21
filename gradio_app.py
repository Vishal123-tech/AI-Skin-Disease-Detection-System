from pathlib import Path
from datetime import datetime
import os
import gradio as gr

from config import MODEL_PATH, LABELS_PATH, UPLOAD_DIR, REPORT_DIR
from predictor import SkinPredictor
from quality import check_image
from report import create_report

predictor = SkinPredictor(MODEL_PATH, LABELS_PATH)
PORT = int(os.environ.get("PORT", 7860))

def analyze(image_path):
    if not image_path:
        return "Please upload a skin image.", None
    image_path = Path(image_path)
    quality = check_image(str(image_path))
    
    result = predictor.predict(str(image_path))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORT_DIR / f'{stamp}_report.pdf'
    create_report(image_path, result, quality, report_path)
    
    confidence = f"{result['confidence']:.1%}" if result['status'] == 'model' else 'Unavailable'
    category = result.get('category', 'lesion')

    if category == 'normal':
        category_header = "🟢 **Classification:** Normal Healthy Skin"
    elif category in ('non_skin', 'uncertain'):
        category_header = "🟠 **Classification:** Non-Skin / Uncertain Input"
    elif category == 'lesion':
        category_header = "🔴 **Classification:** Potential Skin Lesion Detected"
    else:
        category_header = "ℹ️ **Classification:** Demo Mode"

    quality_notice = "" if quality.ok else f"\n\n> ⚠️ **Quality Warning:** {quality.message}"

    text = f"""### {category_header}

**Result:** {result['label']}  
**Confidence:** {confidence}  

---
**Image Quality Metrics:**  
- **Quality Status:** {quality.message}  
- **Blur Score:** {quality.blur_score:.1f}  
- **Brightness:** {quality.brightness:.1f}{quality_notice}

> **Disclaimer:** Educational screening prototype only. This result is not a medical diagnosis. Consult a qualified healthcare professional or dermatologist."""
    return text, str(report_path)

with gr.Blocks(title='AI Skin Disease Detection') as demo:
    gr.Markdown('# 🩺 AI Skin Disease Detection\nUpload a skin image to run the classifier (detecting normal skin, non-skin images, or skin lesions) and generate a PDF report.')
    gr.Markdown('> **Important:** This is an educational prototype, not a medical diagnostic device.')
    with gr.Row():
        image = gr.Image(type='filepath', label='Upload skin image')
        result = gr.Markdown('Your result will appear here.')
    analyze_button = gr.Button('Analyze Image', variant='primary')
    report_file = gr.File(label='Download PDF report')
    analyze_button.click(analyze, inputs=image, outputs=[result, report_file])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=PORT, share=False)
