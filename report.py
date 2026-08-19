from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

DISCLAIMER = "Educational screening prototype only. This result is not a diagnosis and must not replace assessment by a qualified healthcare professional."

def create_report(image_path: Path, result: dict, quality, output_path: Path) -> Path:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=42, bottomMargin=42)
    story = [Paragraph("AI Skin Image Analysis Report", styles["Title"]), Spacer(1, 12)]
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Image(str(image_path), width=3.6*inch, height=3.0*inch))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Prediction: <b>{result['label']}</b>", styles["Heading2"]))
    story.append(Paragraph(f"Confidence: {result['confidence']:.1%}" if result['status'] == 'model' else "Confidence: unavailable in demo mode", styles["Normal"]))
    story.append(Paragraph(f"Image quality: {quality.message}", styles["Normal"]))
    story.append(Paragraph(f"Blur score: {quality.blur_score:.1f}; brightness: {quality.brightness:.1f}", styles["Normal"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph(DISCLAIMER, styles["Normal"]))
    doc.build(story)
    return output_path
