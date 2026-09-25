"""Routing regression tests with controlled outputs, not accuracy benchmarks."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from predictor import SkinPredictor
from report import _confidence


class AcneRoutingTests(unittest.TestCase):
    def setUp(self):
        p = self.p = SkinPredictor.__new__(SkinPredictor)
        p.acne_interpreter = object()
        p.acne_focus = False
        p.acne_priority = True
        p.min_confidence = 0.55
        p.labels = ["Acne Vulgaris", "Eczema", "Fungal Infection", "Psoriasis"]
        p.legacy_interpreter = None
        p.legacy_labels = []
        p._run_local_model = Mock(return_value=np.array([0.02, 0.02, 0.94, 0.02]))
        p._run_skin_gate = Mock(return_value=(True, 0.99))
        p._predict_acne_local = Mock(return_value={
            "category": "lesion", "raw_class": "Acne", "confidence": 0.973,
        })
        skin = patch("predictor.skin_gate_hsv", return_value=(True, 0.8))
        doc = patch("predictor.looks_like_document", return_value=False)
        self.skin = skin.start()
        self.doc = doc.start()
        self.addCleanup(skin.stop)
        self.addCleanup(doc.stop)

    def test_acne_cannot_override_fungal(self):
        r = self.p._predict_local("sample.jpg")
        self.p._run_local_model.assert_called_once_with("sample.jpg")
        self.assertEqual(r["category"], "uncertain")
        self.assertEqual(r["raw_class"], "uncertain")
        self.assertEqual(r["uncertainty_reason"], "model_disagreement")
        self.assertIsNone(r["confidence"])
        self.assertEqual(r["top3"], [])
        self.assertEqual(r["probabilities"], {})
        self.assertEqual(r["model_assessments"][0]["label"], "Fungal Infection")
        self.assertEqual(_confidence(r), "Unavailable")

    def test_agreeing_acne_does_not_inflate_confidence(self):
        self.p._run_local_model.return_value = np.array([0.88, 0.04, 0.04, 0.04])
        r = self.p._predict_local("sample.jpg")
        self.assertEqual(r["raw_class"], "Acne Vulgaris")
        self.assertEqual(r["category"], "lesion")
        self.assertAlmostEqual(r["confidence"], 0.88)

    def test_not_acne_preserves_fungal(self):
        self.p._predict_acne_local.return_value.update(category="non_acne", raw_class="Not_Acne")
        self.assertEqual(self.p._predict_local("sample.jpg")["raw_class"], "Fungal Infection")

    def test_not_acne_conflicts_with_broad_acne(self):
        self.p._run_local_model.return_value = np.array([0.94, 0.02, 0.02, 0.02])
        self.p._predict_acne_local.return_value.update(category="non_acne", raw_class="Not_Acne")
        self.assertEqual(self.p._predict_local("sample.jpg")["category"], "uncertain")

    def test_weak_broad_model_is_not_promoted(self):
        self.p._run_local_model.return_value = np.array([0.24, 0.24, 0.28, 0.24])
        self.assertEqual(self.p._predict_local("sample.jpg")["category"], "uncertain")

    def test_uncertain_specialist_does_not_veto_fungal(self):
        self.p._predict_acne_local.return_value["category"] = "uncertain"
        self.assertEqual(self.p._predict_local("sample.jpg")["raw_class"], "Fungal Infection")

    def test_document_rejection_wins(self):
        self.doc.return_value = True
        self.assertEqual(self.p._predict_local("sample.jpg")["category"], "non_skin")
        self.p._predict_acne_local.assert_not_called()

    def test_gate_rejection_wins(self):
        self.skin.return_value = (True, 0.2)
        self.p._run_skin_gate.return_value = (False, 0.99)
        self.assertEqual(self.p._predict_local("sample.jpg")["category"], "non_skin")
        self.p._predict_acne_local.assert_not_called()

    def test_legacy_result_is_cross_checked(self):
        self.p._run_local_model.return_value = np.array([0.24, 0.24, 0.28, 0.24])
        self.p.legacy_interpreter = object()
        self.p.legacy_labels = ["Melanoma", "Melanocytic Nevi"]
        self.p._run_legacy_model = Mock(return_value=np.array([0.9, 0.1]))
        r = self.p._predict_local("sample.jpg")
        self.assertEqual(r["category"], "uncertain")
        self.assertEqual(r["model_assessments"][0]["label"], "Melanoma")

    def test_disabled_or_missing_specialist(self):
        for available, enabled in [(True, False), (False, True)]:
            with self.subTest(available=available, enabled=enabled):
                self.p.acne_interpreter = object() if available else None
                self.p.acne_priority = enabled
                self.assertEqual(self.p._predict_local("sample.jpg")["raw_class"], "Fungal Infection")
        self.p._predict_acne_local.assert_not_called()

    def test_explicit_binary_mode(self):
        self.p.acne_focus = True
        self.assertEqual(self.p._predict_local("sample.jpg")["raw_class"], "Acne")
        self.p._run_local_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
