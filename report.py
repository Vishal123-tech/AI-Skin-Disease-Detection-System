"""Professional PDF report generation for the skin-screening prototype.

This module is intentionally presentation-only. It consumes the existing
prediction and image-quality values without changing model decisions,
preprocessing, confidence calculations, or classification logic.
"""

from __future__ import annotations

import html
import io
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DISCLAIMER = (
    "Educational screening prototype only. This AI-generated result is not a "
    "medical diagnosis and should not replace evaluation by a qualified "
    "healthcare professional."
)

NAVY = colors.HexColor("#15324B")
CHARCOAL = colors.HexColor("#25313C")
ACCENT = colors.HexColor("#2477A8")
MUTED = colors.HexColor("#657482")
LINE = colors.HexColor("#D8E1E8")
CARD = colors.HexColor("#F5F8FA")
WHITE = colors.white
GREEN = colors.HexColor("#2D7D5B")
AMBER = colors.HexColor("#B36A12")
RED = colors.HexColor("#A83B42")
BLUE_GRAY = colors.HexColor("#596C7C")

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 17 * mm
RIGHT_MARGIN = 17 * mm
TOP_MARGIN = 21 * mm
BOTTOM_MARGIN = 24 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN


def _plain_text(value: Any, fallback: str = "Not available") -> str:
    """Normalize dynamic text for Helvetica and remove decorative emoji."""

    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("So")
        and char not in {"\ufe0f", "\u200d"}
    )
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or fallback


def _safe(value: Any, fallback: str = "Not available") -> str:
    return html.escape(_plain_text(value, fallback))


def _metric(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{number:.1f}" if math.isfinite(number) else "Not available"


def _confidence(result: dict) -> str:
    value = result.get("confidence")
    if result.get("status") not in {"model", "gemini"}:
        return "Unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not math.isfinite(number):
        return "Unavailable"
    return f"{max(0.0, min(1.0, number)):.1%}"


def _category_presentation(result: dict) -> tuple[str, colors.Color]:
    category = str(result.get("category", "")).lower()
    mapping = {
        "normal": ("NORMAL SKIN", GREEN),
        "non_skin": ("NON-SKIN IMAGE", BLUE_GRAY),
        "uncertain": ("UNCERTAIN", AMBER),
        "non_acne": ("NO STRONG ACNE PATTERN", GREEN),
        "lesion": ("POTENTIAL SKIN CONDITION", RED),
        "demo": ("DEMONSTRATION MODE", BLUE_GRAY),
    }
    return mapping.get(category, (category.replace("_", " ").upper() or "AI RESULT", ACCENT))


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=sample["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=1 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=CHARCOAL,
        ),
        "kicker": ParagraphStyle(
            "Kicker", parent=sample["Normal"], fontName="Helvetica",
            fontSize=7.5, leading=10, textColor=MUTED, spaceBefore=1.5 * mm,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=sample["Normal"], fontName="Helvetica",
            fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_RIGHT,
        ),
        "section": ParagraphStyle(
            "Section", parent=sample["Heading2"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=NAVY, spaceAfter=0,
        ),
        "section_number": ParagraphStyle(
            "SectionNumber", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=WHITE, alignment=TA_CENTER,
        ),
        "body": ParagraphStyle(
            "Body", parent=sample["BodyText"], fontName="Helvetica",
            fontSize=9.3, leading=14, textColor=CHARCOAL, spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "Small", parent=sample["BodyText"], fontName="Helvetica",
            fontSize=7.8, leading=11, textColor=MUTED,
        ),
        "card_label": ParagraphStyle(
            "CardLabel", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=7.5, leading=9, textColor=MUTED, spaceAfter=1.5 * mm,
        ),
        "card_value": ParagraphStyle(
            "CardValue", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=NAVY,
        ),
        "result": ParagraphStyle(
            "Result", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=18, leading=22, textColor=NAVY,
        ),
        "confidence": ParagraphStyle(
            "Confidence", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=20, leading=23, textColor=NAVY, alignment=TA_RIGHT,
        ),
        "confidence_label": ParagraphStyle(
            "ConfidenceLabel", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=7.5, leading=9, textColor=MUTED, alignment=TA_RIGHT,
        ),
        "badge": ParagraphStyle(
            "Badge", parent=sample["Normal"], fontName="Helvetica-Bold",
            fontSize=7.5, leading=9, textColor=WHITE, alignment=TA_CENTER,
        ),
        "info_box": ParagraphStyle(
            "InfoBox", parent=sample["BodyText"], fontName="Helvetica",
            fontSize=9.2, leading=14, textColor=CHARCOAL, backColor=CARD,
            borderColor=LINE, borderWidth=0.7,
            borderPadding=(4 * mm, 4 * mm, 4 * mm, 4 * mm),
            borderRadius=3 * mm, spaceAfter=2.5 * mm,
        ),
        "next_step": ParagraphStyle(
            "NextStep", parent=sample["BodyText"], fontName="Helvetica",
            fontSize=9.2, leading=14, textColor=CHARCOAL,
            backColor=colors.HexColor("#EEF6FA"),
            borderColor=colors.HexColor("#BAD6E5"), borderWidth=0.7,
            borderPadding=(4 * mm, 4 * mm, 4 * mm, 4 * mm),
            borderRadius=3 * mm,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=sample["Normal"], fontName="Helvetica",
            fontSize=6.5, leading=8, textColor=MUTED, alignment=TA_LEFT,
        ),
    }


class FramedImage(Flowable):
    """Aspect-ratio-preserving, downsampled image in a framed container."""

    def __init__(
        self,
        image_path: Path,
        max_width: float = CONTENT_WIDTH,
        max_height: float = 78 * mm,
        caption: str = "Input image analyzed by the AI screening model",
    ) -> None:
        super().__init__()
        self.max_width = max_width
        self.max_height = max_height
        self.padding = 3.5 * mm
        self.caption_height = 8 * mm
        self.caption = caption

        with PILImage.open(image_path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            source.thumbnail((1800, 1800), PILImage.Resampling.LANCZOS)
            self.pixel_width, self.pixel_height = source.size
            buffer = io.BytesIO()
            source.save(buffer, format="JPEG", quality=90, optimize=True)
            self._image_bytes = buffer.getvalue()

        available_width = max_width - 2 * self.padding
        available_height = max_height - 2 * self.padding - self.caption_height
        scale = min(
            available_width / max(self.pixel_width, 1),
            available_height / max(self.pixel_height, 1),
        )
        self.draw_width = self.pixel_width * scale
        self.draw_height = self.pixel_height * scale
        self.width = max_width
        self.height = self.draw_height + 2 * self.padding + self.caption_height

    def draw(self) -> None:
        self.canv.setFillColor(WHITE)
        self.canv.setStrokeColor(LINE)
        self.canv.setLineWidth(0.8)
        self.canv.roundRect(0, 0, self.width, self.height, 3 * mm, fill=1, stroke=1)
        image_x = (self.width - self.draw_width) / 2
        image_y = self.padding + self.caption_height
        self.canv.drawImage(
            ImageReader(io.BytesIO(self._image_bytes)), image_x, image_y,
            width=self.draw_width, height=self.draw_height,
            preserveAspectRatio=True, anchor="c", mask="auto",
        )
        self.canv.setFillColor(MUTED)
        self.canv.setFont("Helvetica", 7.2)
        self.canv.drawCentredString(self.width / 2, 3.7 * mm, self.caption)


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds the disclaimer and Page X of Y footer."""

    def __init__(self, *args, **kwargs) -> None:
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states: list[dict] = []
        self.setTitle("AI Skin Image Analysis - AI-Assisted Skin Screening Report")
        self.setAuthor("AI Skin Disease Detection Prototype")
        self.setSubject("Educational AI-assisted skin image screening report")

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_running_header()
            self._draw_footer(page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_running_header(self) -> None:
        if self._pageNumber <= 1:
            return
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(NAVY)
        self.drawString(LEFT_MARGIN, PAGE_HEIGHT - 10.5 * mm, "AI Skin Image Analysis")
        self.setFont("Helvetica", 7.2)
        self.setFillColor(MUTED)
        self.drawRightString(
            PAGE_WIDTH - RIGHT_MARGIN,
            PAGE_HEIGHT - 10.5 * mm,
            "AI-Assisted Skin Screening Report | Continued",
        )
        self.setStrokeColor(LINE)
        self.setLineWidth(0.6)
        self.line(
            LEFT_MARGIN,
            PAGE_HEIGHT - 13.5 * mm,
            PAGE_WIDTH - RIGHT_MARGIN,
            PAGE_HEIGHT - 13.5 * mm,
        )
        self.restoreState()

    def _draw_footer(self, page_count: int) -> None:
        self.saveState()
        self.setStrokeColor(LINE)
        self.setLineWidth(0.6)
        self.line(LEFT_MARGIN, 19 * mm, PAGE_WIDTH - RIGHT_MARGIN, 19 * mm)
        disclaimer = Paragraph(html.escape(DISCLAIMER), _styles()["footer"])
        disclaimer.wrapOn(self, CONTENT_WIDTH, 11 * mm)
        disclaimer.drawOn(self, LEFT_MARGIN, 9.2 * mm)
        self.setFont("Helvetica-Bold", 6.8)
        self.setFillColor(NAVY)
        self.drawString(LEFT_MARGIN, 5.5 * mm, "AI Skin Image Analysis | Educational Screening Prototype")
        self.setFont("Helvetica", 6.8)
        self.setFillColor(MUTED)
        self.drawRightString(
            PAGE_WIDTH - RIGHT_MARGIN, 5.5 * mm,
            f"Page {self._pageNumber} of {page_count}",
        )
        self.restoreState()


def _section_heading(number: str, title: str, styles: dict[str, ParagraphStyle]) -> Table:
    number_box = Table(
        [[Paragraph(number, styles["section_number"])]],
        colWidths=[9 * mm], rowHeights=[6 * mm],
    )
    number_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    heading = Table(
        [[number_box, Paragraph(html.escape(title), styles["section"])]],
        colWidths=[12 * mm, CONTENT_WIDTH - 12 * mm],
    )
    heading.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (1, 0), (-1, 0), 0.6, LINE),
    ]))
    return heading


def create_header(generated_at: datetime, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    mark_style = ParagraphStyle("Mark", parent=styles["badge"], fontSize=11, leading=13)
    mark = Table([[Paragraph("AI", mark_style)]], colWidths=[12 * mm], rowHeights=[12 * mm])
    mark.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    title_stack = [
        Paragraph("AI Skin Image Analysis", styles["title"]),
        Paragraph("AI-Assisted Skin Screening Report", styles["subtitle"]),
        Paragraph(
            "Computer Vision | Image Analysis | Educational Screening Prototype",
            styles["kicker"],
        ),
    ]
    metadata = Paragraph(
        "<b>REPORT GENERATED</b><br/>" + generated_at.strftime("%d %b %Y<br/>%I:%M %p"),
        styles["meta"],
    )
    header = Table(
        [[mark, title_stack, metadata]],
        colWidths=[16 * mm, CONTENT_WIDTH - 52 * mm, 36 * mm],
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
        ("LINEBELOW", (0, 0), (-1, -1), 1.1, ACCENT),
    ]))
    return [header, Spacer(1, 5 * mm)]


def create_image_section(image_path: Path, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    return [
        _section_heading("01", "ANALYZED IMAGE", styles),
        Spacer(1, 3 * mm),
        FramedImage(image_path),
        Spacer(1, 5 * mm),
    ]


def create_result_card(result: dict, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    badge_text, badge_color = _category_presentation(result)
    badge = Table(
        [[Paragraph(html.escape(badge_text), styles["badge"])]],
        colWidths=[54 * mm], rowHeights=[7 * mm],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    result_name = _safe(result.get("label") or result.get("raw_class"), "Unknown result")
    left_content = [badge, Spacer(1, 2.2 * mm), Paragraph(result_name, styles["result"])]
    right_content = [
        Paragraph("MODEL CONFIDENCE", styles["confidence_label"]),
        Paragraph(_confidence(result), styles["confidence"]),
    ]
    result_table = Table(
        [
            [Paragraph("AI CLASSIFICATION", styles["card_label"]), ""],
            [left_content, right_content],
            [Paragraph(
                "Model confidence reflects certainty for this classification task; "
                "it is not a measure of medical certainty.", styles["small"],
            ), ""],
        ],
        colWidths=[CONTENT_WIDTH * 0.70, CONTENT_WIDTH * 0.30],
    )
    result_table.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("SPAN", (0, 2), (1, 2)),
        ("BACKGROUND", (0, 0), (-1, -1), CARD),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5 * mm),
        ("LINEBEFORE", (1, 1), (1, 1), 0.7, LINE),
    ]))
    return [
        _section_heading("02", "AI CLASSIFICATION RESULT", styles),
        Spacer(1, 3 * mm), KeepTogether(result_table), Spacer(1, 5 * mm),
    ]


def create_quality_section(quality: Any, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    ok = bool(getattr(quality, "ok", False)) if quality is not None else False
    message = _safe(getattr(quality, "message", None), "Not available")
    status = "Acceptable" if ok else "Review recommended"
    status_color = GREEN if ok else AMBER
    cells = [
        ["IMAGE QUALITY", status, status_color],
        ["BLUR SCORE", _metric(getattr(quality, "blur_score", None)), NAVY],
        ["BRIGHTNESS", _metric(getattr(quality, "brightness", None)), NAVY],
    ]
    flowable_cells = []
    for label, value, value_color in cells:
        value_style = ParagraphStyle(
            f"QualityValue{label}", parent=styles["card_value"], textColor=value_color,
        )
        flowable_cells.append([
            Paragraph(label, styles["card_label"]),
            Paragraph(html.escape(value), value_style),
        ])
    metrics = Table([flowable_cells], colWidths=[CONTENT_WIDTH / 3] * 3)
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
    ]))
    return [
        _section_heading("03", "IMAGE QUALITY", styles),
        Spacer(1, 3 * mm), KeepTogether(metrics), Spacer(1, 1.5 * mm),
        Paragraph(f"<b>Quality assessment:</b> {message}", styles["small"]),
        Spacer(1, 5 * mm),
    ]


def create_summary_section(result: dict, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    disease_info = result.get("disease_info") or {}
    description = disease_info.get("description")
    notes = result.get("notes")
    paragraphs: list[str] = []
    if description:
        paragraphs.append(f"<b>AI screening summary</b><br/>{_safe(description)}")
    if notes and _plain_text(notes) != _plain_text(description):
        paragraphs.append(f"<b>Model observations</b><br/>{_safe(notes)}")
    if not paragraphs:
        paragraphs.append(
            "<b>AI screening summary</b><br/>No additional model summary was provided for this result."
        )

    content: list[Flowable] = [
        _section_heading("04", "ANALYSIS SUMMARY", styles), Spacer(1, 3 * mm),
    ]
    for text in paragraphs:
        content.append(Paragraph(text, styles["info_box"]))

    top3 = result.get("top3") or []
    clean_top3: list[tuple[str, str]] = []
    for item in top3[:3]:
        try:
            label, probability = item
            probability_text = f"{float(probability):.1%}"
        except (TypeError, ValueError):
            continue
        clean_top3.append((_safe(label), probability_text))
    if clean_top3:
        rows = [[
            Paragraph("MODEL ALTERNATIVE", styles["card_label"]),
            Paragraph("CONFIDENCE", styles["card_label"]),
        ]]
        alt_confidence = ParagraphStyle(
            "AltConfidence", parent=styles["body"], alignment=TA_RIGHT,
        )
        for label, probability in clean_top3:
            rows.append([
                Paragraph(label, styles["body"]),
                Paragraph(probability, alt_confidence),
            ])
        alternatives = Table(
            rows, colWidths=[CONTENT_WIDTH * 0.78, CONTENT_WIDTH * 0.22], repeatRows=1,
        )
        alternatives.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F5")),
            ("BOX", (0, 0), (-1, -1), 0.6, LINE),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
            ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1 * mm),
        ]))
        content.extend([Spacer(1, 1 * mm), alternatives])
    content.append(Spacer(1, 5 * mm))
    return content


def create_next_step_section(
    result: dict,
    quality: Any,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    quality_ok = bool(getattr(quality, "ok", False)) if quality is not None else False
    quality_message = getattr(quality, "message", None) if quality is not None else None
    disease_info = result.get("disease_info") or {}
    next_step = quality_message if not quality_ok and quality_message else disease_info.get("severity")
    if not next_step:
        return []
    return [
        _section_heading("05", "INTERPRETATION / NEXT STEP", styles),
        Spacer(1, 3 * mm),
        Paragraph(f"<b>Next step</b><br/>{_safe(next_step)}", styles["next_step"]),
        Spacer(1, 3 * mm),
    ]


def create_report(image_path: Path, result: dict, quality: Any, output_path: Path) -> Path:
    """Create a polished A4 report from existing application values."""

    image_path = Path(image_path)
    output_path = Path(output_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Analyzed image not found: {image_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=RIGHT_MARGIN, leftMargin=LEFT_MARGIN,
        topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
        title="AI Skin Image Analysis - AI-Assisted Skin Screening Report",
        author="AI Skin Disease Detection Prototype",
        subject="Educational AI-assisted skin image screening report",
    )
    story: list[Flowable] = []
    story.extend(create_header(datetime.now(), styles))
    story.extend(create_image_section(image_path, styles))
    story.extend(create_result_card(result, styles))
    story.extend(create_quality_section(quality, styles))
    story.append(CondPageBreak(78 * mm))
    story.extend(create_summary_section(result, styles))
    story.extend(create_next_step_section(result, quality, styles))
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path
