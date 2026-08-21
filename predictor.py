from pathlib import Path
from PIL import Image
import numpy as np

class SkinPredictor:
    def __init__(self, model_path: Path, labels_path: Path, image_size=(224, 224), min_confidence=0.45):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.interpreter = None
        self.pt_model = None
        self.device = None
        self.labels = []
        self.load_error = None
        self.backend = None

        if self.labels_path.exists():
            self.labels = [x.strip() for x in self.labels_path.read_text(encoding="utf-8").splitlines() if x.strip()]

        # 1. Try PyTorch .pt model first if available
        pt_path = self.model_path.with_suffix('.pt')
        if pt_path.exists():
            try:
                import torch
                import torch.nn as nn
                from torchvision import models, transforms
                
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = models.mobilenet_v2()
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, len(self.labels))
                )
                model.load_state_dict(torch.load(pt_path, map_location=self.device))
                model.eval()
                model = model.to(self.device)
                
                self.pt_model = model
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                self.backend = "pytorch"
            except Exception as exc:
                self.pt_model = None
                self.load_error = str(exc)

        # 2. Try LiteRT / TFLite model if PyTorch was not loaded
        if self.pt_model is None and self.model_path.exists():
            try:
                try:
                    import ai_edge_litert.interpreter as tflite
                    self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
                    self.backend = "litert"
                except ImportError:
                    try:
                        import tflite_runtime.interpreter as tflite
                        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
                        self.backend = "tflite_runtime"
                    except ImportError:
                        import tensorflow as tf
                        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                        self.backend = "tensorflow"

                self.interpreter.allocate_tensors()
                input_shape = self.interpreter.get_input_details()[0]["shape"]
                if len(input_shape) == 4:
                    self.image_size = (int(input_shape[2]), int(input_shape[1]))
            except Exception as exc:
                self.interpreter = None
                self.load_error = str(exc)

    @property
    def demo_mode(self):
        return self.interpreter is None and self.pt_model is None

    def predict(self, image_path: str):
        if self.demo_mode:
            label = "Demo mode — ML runtime unavailable" if (self.model_path.exists() or self.model_path.with_suffix('.pt').exists()) else "Demo mode — model not installed"
            return {
                "label": label,
                "raw_class": "demo",
                "confidence": 0.0,
                "category": "demo",
                "status": "demo",
                "probabilities": {}
            }

        image = Image.open(image_path).convert("RGB")

        if self.backend == "pytorch" and self.pt_model is not None:
            import torch
            tensor_img = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.pt_model(tensor_img)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        else:
            image_resized = image.resize(self.image_size)
            data = np.asarray(image_resized, dtype=np.float32)[None, ...]
            details = self.interpreter.get_input_details()[0]
            if details["dtype"] == np.uint8:
                scale, zero = details["quantization"]
                data = (data / scale + zero).astype(np.uint8) if scale else data.astype(np.uint8)
            else:
                data = (data / 127.5) - 1.0

            self.interpreter.set_tensor(details["index"], data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"])[0]

            if np.max(output) > 1.0 or np.min(output) < 0.0:
                exp_out = np.exp(output - np.max(output))
                probs = exp_out / np.sum(exp_out)
            else:
                probs = output

        index = int(np.argmax(probs))
        confidence = float(probs[index])
        raw_label = self.labels[index] if index < len(self.labels) else f"Class {index}"

        probabilities = {
            self.labels[i] if i < len(self.labels) else f"Class {i}": float(probs[i])
            for i in range(len(probs))
        }

        clean_name = raw_label.replace("_", " ").strip()
        norm_name = clean_name.lower()

        if confidence < self.min_confidence:
            category = "uncertain"
            display_label = f"Uncertain / Non-Skin Image (Confidence: {confidence:.1%})"
        elif "non skin" in norm_name or "other" in norm_name or "background" in norm_name:
            category = "non_skin"
            display_label = "Non-Skin Image / Background Object"
        elif "normal" in norm_name or "healthy" in norm_name:
            category = "normal"
            display_label = "Normal Healthy Skin (No Lesion Detected)"
        else:
            category = "lesion"
            display_label = clean_name

        return {
            "label": display_label,
            "raw_class": raw_label,
            "confidence": confidence,
            "category": category,
            "status": "model",
            "probabilities": probabilities
        }
