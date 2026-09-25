"""Presentation tests: preserve uncertain scores and escape model text."""
import unittest
from copy import deepcopy
from quality import QualityResult
from ui_presentation import render_result, EMPTY_RESULT


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.quality = QualityResult(True, 50.0, 120.0, "Image quality is acceptable.")
        self.result = {"category": "lesion", "status": "model", "label": "Acne Vulgaris", "confidence": .8}

    def test_result_does_not_change_inference(self):
        original = deepcopy(self.result)
        html = render_result(self.result, self.quality)
        self.assertEqual(self.result, original)
        self.assertIn("80.0%", html)
        self.assertIn("Acne Vulgaris", html)
        self.assertIn("not the probability", html)

    def test_untrusted_text_is_escaped(self):
        self.result["label"] = '<script>alert("x")</script>'
        html = render_result(self.result, self.quality)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_disagreement_not_displayed_as_zero(self):
        self.result.update(category="uncertain", confidence=None, uncertainty_reason="model_disagreement")
        html = render_result(self.result, self.quality)
        self.assertIn("Unavailable — models disagree", html)
        self.assertNotIn("0.0%", html)

    def test_quality_warning_is_visible(self):
        html = render_result(self.result, QualityResult(False, 2, 100, "Please retake the photo."))
        self.assertIn("Photo needs attention", html)
        self.assertIn("Please retake the photo.", html)

    def test_empty_state_has_instructions(self):
        self.assertIn("Analyze image", EMPTY_RESULT)
        self.assertIn('aria-live="polite"', EMPTY_RESULT)


if __name__ == "__main__":
    unittest.main()
