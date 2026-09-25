"""Verify uncertain results survive the UI, JSON API, and PDF generation."""
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


class PredictionSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Surface tests use controlled predictions, not heavyweight model loads.
        with patch("predictor.SkinPredictor"):
            cls.ui = importlib.import_module("gradio_app")
            cls.api = importlib.import_module("app")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "sample.png"
        Image.new("RGB", (224, 224), (130, 130, 130)).save(self.image)
        self.result = {
            "category": "uncertain", "status": "model", "raw_class": "uncertain",
            "label": "Uncertain — Models disagree; no condition confirmed",
            "confidence": None, "uncertainty_reason": "model_disagreement",
            "top3": [], "probabilities": {}, "model_used": "SCIN + focused acne cross-check",
            "disease_info": {"description": "Models disagree", "severity": "No condition confirmed"},
            "notes": "Model disagreement: SCIN: Fungal Infection; focused acne model: Acne",
        }

    def test_gradio_and_pdf_accept_unavailable_confidence(self):
        with patch.object(self.ui.predictor, "predict", return_value=self.result), patch.object(self.ui, "REPORT_DIR", self.root):
            text, pdf, context, _ = self.ui.analyze(str(self.image))
        self.assertIn("Unavailable — models disagree", text)
        self.assertNotIn("Top possible condition", text)
        self.assertIsNone(context["result"]["confidence"])
        self.assertTrue(Path(pdf).read_bytes().startswith(b"%PDF"))

    def test_website_headers_include_the_skinscanix_logo(self):
        from branding import LOGO_PATH, website_header
        import base64
        header = website_header()
        self.assertIn('alt="SkinScanix logo"', header)
        self.assertIn(base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii"), header)
        response = self.api.app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'alt="SkinScanix logo"', response.data)
        html_components = [c for c in self.ui.demo.config["components"] if c["type"] == "html"]
        self.assertTrue(any('alt="SkinScanix logo"' in c["props"].get("value", "") for c in html_components))
        self.assertIn("SkinScanix", self.ui.demo.config["title"])

    def test_new_upload_clears_stale_result_report_and_feedback(self):
        text, pdf, context, visibility = self.ui.reset_analysis()
        self.assertIn("Your result will appear here", text)
        self.assertIsNone(pdf)
        self.assertIsNone(context)
        self.assertFalse(visibility["visible"])

    def test_flask_json_and_browser_accept_unavailable_confidence(self):
        client = self.api.app.test_client()
        with patch.object(self.api.predictor, "predict", return_value=self.result), patch.object(self.api, "UPLOAD_DIR", self.root), patch.object(self.api, "REPORT_DIR", self.root):
            response = client.post("/api/predict", data={"image": (io.BytesIO(self.image.read_bytes()), "photo.png")})
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["confidence"], "Unavailable")
            self.assertEqual(data["category"], "uncertain")
            self.assertEqual(data["raw_class"], "uncertain")
            response = client.post("/", data={"image": (io.BytesIO(self.image.read_bytes()), "photo.png")})
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Unavailable", response.data)


if __name__ == "__main__":
    unittest.main()
