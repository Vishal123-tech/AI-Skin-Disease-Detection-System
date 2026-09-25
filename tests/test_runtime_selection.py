"""Deployment pin and feedback regression tests; not clinical validation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from predictor import SkinPredictor
from self_train import train_on_feedback


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "skin_model.tflite"
        self.path.write_bytes(b"test artifact")
        self.p = SkinPredictor.__new__(SkinPredictor)
        self.p.model_path = self.path
        self.p.labels = ["Acne Vulgaris", "Eczema", "Fungal Infection", "Psoriasis"]
        self.p.preprocess_mode = "minus_one_to_one"
        self.p.acne_priority = True
        self.manifest = {
            "backend": "tflite", "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "class_names": self.p.labels, "input_preprocessing": "raw_0_255",
            "acne_crosscheck": False,
        }

    def save_manifest(self):
        self.path.with_suffix(".runtime.json").write_text(json.dumps(self.manifest))

    def test_pin_restores_raw_input_and_disables_acne_veto(self):
        self.save_manifest()
        self.assertEqual(self.p._load_runtime_selection(), "tflite")
        self.assertEqual(self.p.preprocess_mode, "raw_0_255")
        self.assertFalse(self.p.acne_priority)

    def test_stale_pin_fails_instead_of_silently_switching_models(self):
        self.save_manifest()
        self.path.write_bytes(b"different model")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.p._load_runtime_selection()

    def test_label_order_mismatch_fails(self):
        self.manifest["class_names"] = list(reversed(self.p.labels))
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "label order"):
            self.p._load_runtime_selection()

    def test_unknown_preprocessing_fails(self):
        self.manifest["input_preprocessing"] = "guess"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            self.p._load_runtime_selection()

    def test_unpinned_project_keeps_automatic_selection(self):
        self.assertEqual(self.p._load_runtime_selection(), "auto")

    def test_feedback_cannot_overwrite_active_checkpoint(self):
        with patch("self_train.load_usable_feedback", return_value=[{"label": "Eczema"}]), \
                patch("self_train.collect_replay_samples") as replay, \
                patch("self_train.mark_learned_ids") as mark:
            result = train_on_feedback(min_samples=1)
        self.assertFalse(result["success"])
        self.assertTrue(result["requires_review"])
        replay.assert_not_called()
        mark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
