from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class QualityResult:
    ok: bool
    blur_score: float
    brightness: float
    message: str

def check_image(path: str, min_width: int = 200, min_height: int = 200) -> QualityResult:
    image = cv2.imread(path)
    if image is None:
        return QualityResult(False, 0.0, 0.0, "The image could not be read.")
    h, w = image.shape[:2]
    if w < min_width or h < min_height:
        return QualityResult(False, 0.0, 0.0, f"Image is too small ({w}x{h}); use at least {min_width}x{min_height} pixels.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    if blur < 40:
        return QualityResult(False, blur, brightness, "Image is too blurry. Hold the camera steady and retake it.")
    if brightness < 35 or brightness > 225:
        return QualityResult(False, blur, brightness, "Lighting is unsuitable. Use even, indirect light and retake it.")
    return QualityResult(True, blur, brightness, "Image quality is acceptable.")
