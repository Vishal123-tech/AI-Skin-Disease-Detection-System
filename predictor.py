from pathlib import Path
from PIL import Image
import numpy as np

class SkinPredictor:
    def __init__(self, model_path: Path, labels_path: Path, image_size=(224, 224)):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.image_size = image_size
        self.interpreter = None
        self.labels = []
        if self.labels_path.exists():
            self.labels = [x.strip() for x in self.labels_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if self.model_path.exists():
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                self.interpreter.allocate_tensors()
                input_shape = self.interpreter.get_input_details()[0]["shape"]
                if len(input_shape) == 4:
                    self.image_size = (int(input_shape[2]), int(input_shape[1]))
            except Exception as exc:
                # Keep the web app usable when the optional TensorFlow package is
                # not installed or is unavailable on the current platform.
                self.interpreter = None
                self.load_error = str(exc)

    @property
    def demo_mode(self):
        return self.interpreter is None

    def predict(self, image_path: str):
        if self.demo_mode:
            label = "Demo mode — TensorFlow runtime unavailable" if self.model_path.exists() else "Demo mode — model not installed"
            return {"label": label, "confidence": 0.0, "status": "demo"}
        image = Image.open(image_path).convert("RGB").resize(self.image_size)
        data = np.asarray(image, dtype=np.float32)[None, ...]
        details = self.interpreter.get_input_details()[0]
        if details["dtype"] == np.uint8:
            scale, zero = details["quantization"]
            data = (data / scale + zero).astype(np.uint8) if scale else data.astype(np.uint8)
        else:
            data = (data / 127.5) - 1.0
        self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"])[0]
        index = int(np.argmax(output))
        confidence = float(output[index])
        label = self.labels[index] if index < len(self.labels) else f"Class {index}"
        return {"label": label, "confidence": confidence, "status": "model"}
