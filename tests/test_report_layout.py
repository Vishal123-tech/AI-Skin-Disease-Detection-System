"""Run with the bundled PDF runtime (reportlab, Pillow and pypdf)."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
from report import create_report


def sample_result():
    return {
        "category": "lesion", "status": "model", "label": "Possible Acne Vulgaris - Screening Result Only",
        "confidence": 0.823,
        "disease_info": {
            "description": "Example report text for layout testing, not a real assessment. "
                           "Model predictions can be uncertain and must not be used as a diagnosis.",
            "severity": "Consult a qualified healthcare professional for assessment.",
        },
        "notes": "Illustrative values only. No model inference was performed for this sample.",
        "top3": [("Acne Vulgaris", .823), ("Fungal Infection", .102), ("Eczema", .075)],
    }


@unittest.skipIf(PdfReader is None, "Run with the bundled PDF runtime for layout checks")
class ReportLayoutTests(unittest.TestCase):
    def test_single_page_across_image_shapes_and_result_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape in [(240, 800), (800, 240), (480, 480)]:
                photo = root / "input.png"
                Image.new("RGB", shape, "#d6dce3").save(photo)
                for category, label, confidence in [
                    ("lesion", "Possible Acne Vulgaris - Screening Result Only", .823),
                    ("lesion", "Benign Keratosis-like Lesions", .712),
                    ("uncertain", "Uncertain - Models disagree; no condition confirmed", None),
                    ("non_skin", "Not a Confirmed Skin Image - Please upload a clear photo of skin", .629),
                    ("normal", "Normal Skin", .832),
                ]:
                    for good_quality in [True, False]:
                        with self.subTest(shape=shape, category=category, quality=good_quality):
                            result = sample_result()
                            result.update(category=category, label=label, confidence=confidence)
                            original = copy.deepcopy(result)
                            quality = SimpleNamespace(ok=good_quality, blur_score=35.6, brightness=72.2,
                                message="Image quality is acceptable." if good_quality else
                                "Image is too blurry. Hold the camera steady and retake it in even lighting.")
                            pdf = create_report(photo, result, quality, root / "report.pdf")
                            reader = PdfReader(pdf)
                            self.assertEqual(len(reader.pages), 1)
                            text = " ".join(reader.pages[0].extract_text().split())
                            self.assertIn(label, text)
                            self.assertIn("82.3%" if confidence == .823 else "SkinScanix", text)
                            if confidence is None:
                                self.assertIn("Unavailable", text)
                            self.assertIn("not a medical diagnosis", text)
                            self.assertIn("35.6", text)
                            self.assertIn("Fungal Infection", text)
                            self.assertIn(quality.message, text)
                            self.assertEqual(result, original)

    def test_long_notes_are_not_silently_cut_off(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "input.png"
            Image.new("RGB", (224, 224), "gray").save(photo)
            result = sample_result()
            result["notes"] = "Long supplementary observation. " * 500 + "END_OF_NOTES"
            pdf = create_report(photo, result, None, root / "long.pdf")
            reader = PdfReader(pdf)
            self.assertIn("END_OF_NOTES", " ".join(p.extract_text() for p in reader.pages))


if __name__ == "__main__":
    unittest.main()
