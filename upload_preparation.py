"""Bound image decoding before Gradio/Pillow expands phone uploads in RAM."""
from pathlib import Path

from PIL import Image, ImageOps


def prepare_upload(source, destination):
    source = Path(source)
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Please upload an image smaller than 10 MB.")
    with Image.open(source) as image:
        width, height = image.size
        if width * height > 50_000_000:
            raise ValueError("Please resize the photo to 50 megapixels or less.")
        # JPEG draft downsamples in the decoder, before allocating the full image.
        if image.format == "JPEG":
            image.draft("RGB", (1280, 1280))
        elif width * height > 4_000_000:
            raise ValueError("Please resize this image to 4 megapixels or upload a JPEG.")
        image.thumbnail((1280, 1280))
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        prepared.save(destination, format="PNG")
    return Path(destination)
