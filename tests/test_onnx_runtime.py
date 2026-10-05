import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from predictor import SkinPredictor, prepare_onnx_image


class OnnxRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'skin_model.tflite'
        self.artifact = self.path.with_suffix('.onnx')
        self.artifact.write_bytes(b'onnx test')
        self.p = SkinPredictor.__new__(SkinPredictor)
        self.p.model_path = self.path
        self.p.labels = ['Acne Vulgaris', 'Eczema', 'Fungal Infection', 'Psoriasis', 'Normal Skin']
        self.p.acne_focus = self.p.acne_priority = True
        self.manifest = {'backend': 'onnx', 'sha256': hashlib.sha256(b'onnx test').hexdigest(),
            'class_names': self.p.labels, 'force_local': True, 'acne_focus': False,
            'acne_crosscheck': False, 'input_preprocessing': 'imagenet_rgb_exif_resize256_centercrop224'}
        self.path.with_suffix('.runtime.json').write_text(json.dumps(self.manifest))

    def test_verified_pin_disables_both_acne_overrides(self):
        self.assertEqual(self.p._load_runtime_selection(), 'onnx')
        self.assertFalse(self.p.acne_focus)
        self.assertFalse(self.p.acne_priority)

    def test_corrupt_onnx_cannot_fall_back(self):
        self.artifact.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.p._load_local_model()

    def test_pinned_local_ignores_gemini_environment(self):
        labels = self.root / 'labels.txt'
        labels.write_text('\n'.join(self.p.labels))
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'not-a-real-key', 'ACNE_FOCUS_MODE': '1'}), \
                patch.object(SkinPredictor, '_load_local_model'), \
                patch.object(SkinPredictor, '_load_legacy_model'), \
                patch.object(SkinPredictor, '_load_gate_model'), \
                patch.object(SkinPredictor, '_load_acne_model'):
            predictor = SkinPredictor(self.path, labels)
        self.assertIsNone(predictor.gemini_model)
        self.assertFalse(predictor.acne_focus)

    def test_preprocessing_matches_torchvision_and_exif(self):
        from PIL import ImageOps
        from torchvision import transforms
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        rng = np.random.default_rng(42)
        for width, height, orientation in [(319, 507, 1), (501, 317, 1), (79, 80, 6), (256, 401, 1)]:
            path = self.root / 'input.jpg'
            image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
            exif = image.getexif()
            exif[274] = orientation
            image.save(path, exif=exif)
            with Image.open(path) as source:
                expected = transform(ImageOps.exif_transpose(source).convert('RGB')).numpy()[None]
            np.testing.assert_allclose(prepare_onnx_image(str(path)), expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
