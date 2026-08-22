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
    
    label = result.get('label', 'Unknown')
    conf_val = result.get('confidence', 0.0)
    conf_str = f"{conf_val:.1%}" if isinstance(conf_val, (int, float)) and result.get('status') in ('model', 'ollama', 'gemini') else "Unavailable (demo mode)"

    story.append(Paragraph(f"Classification Result: <b>{label}</b>", styles["Heading2"]))
    story.append(Paragraph(f"Confidence Level: {conf_str}", styles["Normal"]))
    story.append(Paragraph(f"Image Quality Assessment: {quality.message}", styles["Normal"]))
    story.append(Paragraph(f"Blur score: {quality.blur_score:.1f}; Brightness: {quality.brightness:.1f}", styles["Normal"]))
    
    category = result.get('category', '')
    if category == 'normal':
        recommendation = "The analysis indicates normal skin characteristics with no skin lesions detected."
    elif category in ('non_skin', 'uncertain'):
        recommendation = "The provided image was classified as non-skin or uncertain. Please ensure clear close-up photos of skin areas are submitted."
    elif category == 'lesion':
        recommendation = "A potential skin lesion pattern was identified. If you have concerns about skin changes, consult a certified dermatologist."
    else:
        recommendation = "System running in demonstration mode."
        
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Clinical Summary:</b> {recommendation}", styles["Normal"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph(DISCLAIMER, styles["Normal"]))
    doc.build(story)
    return output_path
