"""Check the experiment's split and balancing safeguards."""
import unittest
import numpy as np

from train_candidate import LABELS, case_weights, check_separation, metrics, split_training


class CandidateTrainingTests(unittest.TestCase):
    def records(self):
        return [{"label": label, "case": f"{index}-{case}", "sha256": f"{index}-{case}-{view}"}
                for index, label in enumerate(LABELS) for case in range(10)
                for view in range(1 + index % 3)]

    def test_all_images_from_case_stay_together(self):
        rows = self.records()
        train, val = split_training(rows)
        check_separation(train, val)
        self.assertEqual(len(train) + len(val), len(rows))
        self.assertEqual(set(r["label"] for r in val), set(LABELS))
        self.assertEqual((train, val), split_training(rows))

    def test_cross_case_duplicate_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            check_separation([{"case": "a", "sha256": "same"}], [{"case": "b", "sha256": "same"}])

    def test_classes_receive_equal_sampler_mass_despite_uneven_images(self):
        rows = self.records()
        weights = case_weights(rows).numpy()
        for label in LABELS:
            self.assertAlmostEqual(sum(w for w, r in zip(weights, rows) if r["label"] == label), 1.0)

    def test_single_class_collapse_does_not_get_high_macro_recall(self):
        target = np.array([0, 1, 1, 1, 2, 3])
        scores = np.array([[0, 1, 0, 0]] * len(target))
        result = metrics(target, scores)
        self.assertAlmostEqual(result["macro_recall"], 0.25)
        self.assertAlmostEqual(result["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
