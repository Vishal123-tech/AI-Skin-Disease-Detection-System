import argparse
from pathlib import Path
import sys

DEFAULT_CLASSES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis like Lesions",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Vascular Lesions",
    "Normal_Skin",
    "Other_Non_Skin"
]

def train_pytorch(train_dir: Path, val_dir: Path, output_path: Path, labels_path: Path, epochs: int, batch_size: int):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, models, transforms
    from torch.utils.data import DataLoader

    print("--- Using PyTorch Engine for Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    print(f"Detected {len(class_names)} classes: {class_names}")

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(class_names), encoding="utf-8")
    print(f"Saved labels to: {labels_path}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=1e-3)

    print("\n--- Starting Feature Extraction Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path = output_path.with_suffix('.pt')
    torch.save(model.state_dict(), pt_path)
    print(f"Successfully saved PyTorch model to: {pt_path}")

def train_tensorflow(train_dir: Path, val_dir: Path, output_path: Path, labels_path: Path, epochs: int, fine_tune_epochs: int, batch_size: int):
    import tensorflow as tf

    print("--- Using TensorFlow Engine for Training ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(224, 224), batch_size=batch_size, label_mode="categorical", shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(224, 224), batch_size=batch_size, label_mode="categorical", shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Detected {len(class_names)} classes: {class_names}")

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(class_names), encoding="utf-8")
    print(f"Saved labels to: {labels_path}")

    inputs = tf.keras.Input(shape=(224, 224, 3))
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
    ])
    x = data_aug(inputs)
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    if fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    print(f"Successfully saved TFLite model to: {output_path}")

def train_fallback(train_dir: Path, val_dir: Path, output_path: Path, labels_path: Path):
    print("--- Using Dataset Configurator & Labels Generator ---")
    if train_dir.exists():
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])
    else:
        class_names = DEFAULT_CLASSES

    if not class_names:
        class_names = DEFAULT_CLASSES

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(class_names), encoding="utf-8")
    print(f"Successfully updated labels.txt with {len(class_names)} classes: {class_names}")

def main():
    parser = argparse.ArgumentParser(description="Train Skin Disease & Normal/Non-Skin Classifier")
    parser.add_argument("--data", required=True, help="Dataset directory containing train/ and val/ subdirectories")
    parser.add_argument("--output", default="models/skin_model.tflite", help="Path to save output model")
    parser.add_argument("--labels", default="models/labels.txt", help="Path to save output labels text file")
    parser.add_argument("--epochs", type=int, default=8, help="Number of initial training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    root = Path(args.data)
    train_dir = root / "train"
    val_dir = root / "val"
    output_path = Path(args.output)
    labels_path = Path(args.labels)

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Dataset root must contain 'train' and 'val' subdirectories under {root}")

    try:
        import torch
        train_pytorch(train_dir, val_dir, output_path, labels_path, args.epochs, args.batch_size)
    except ImportError:
        try:
            import tensorflow as tf
            train_tensorflow(train_dir, val_dir, output_path, labels_path, args.epochs, args.fine_tune_epochs, args.batch_size)
        except ImportError:
            print("Notice: Heavy ML frameworks are still installing or not loaded. Updating dataset labels config...")
            train_fallback(train_dir, val_dir, output_path, labels_path)

if __name__ == "__main__":
    main()
