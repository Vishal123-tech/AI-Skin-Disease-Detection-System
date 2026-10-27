import tempfile
import unittest
from pathlib import Path
from PIL import Image
from upload_preparation import prepare_upload


class UploadPreparationTests(unittest.TestCase):
    def test_large_phone_jpeg_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "phone.jpg", Path(folder) / "small.png"
            Image.new("RGB", (8000, 6000), (180, 130, 100)).save(source)
            prepare_upload(source, target)
            with Image.open(target) as image:
                self.assertEqual(image.size, (1280, 960))

    def test_large_png_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "large.png"
            Image.new("RGB", (2500, 2000)).save(source)
            with self.assertRaisesRegex(ValueError, "4 megapixels"):
                prepare_upload(source, Path(folder) / "out.png")

    def test_small_image_pixels_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "small.png", Path(folder) / "out.png"
            original = Image.new("RGB", (320, 240), (123, 72, 84))
            original.save(source)
            prepare_upload(source, target)
            with Image.open(target) as image:
                self.assertEqual(image.tobytes(), original.tobytes())

    def test_exif_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "rotated.jpg", Path(folder) / "out.png"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (300, 200)).save(source, exif=exif)
            prepare_upload(source, target)
            with Image.open(target) as image:
                self.assertEqual(image.size, (200, 300))
