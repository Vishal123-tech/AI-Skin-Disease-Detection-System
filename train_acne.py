"""Train a focused acne-vs-not-acne TensorFlow Lite classifier.

This script intentionally does not train from the project's placeholder
``data/skin`` images.  It expects real, consented and correctly labelled
images arranged as:

    data/acne/
      train/Acne/
      train/Not_Acne/
      val/Acne/
      val/Not_Acne/
      test/Acne/       # optional but strongly recommended
      test/Not_Acne/   # optional but strongly recommended

The output is suitable for the server-side TFLite predictor.  The Raspberry
Pi still only captures and uploads the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def count_images(directory: Path) -> dict[str, int]:
    return {
        class_dir.name: sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for class_dir in sorted(directory.iterdir())
        if class_dir.is_dir()
    }


def require_binary_dataset(root: Path) -> tuple[Path, Path]:
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"{root} must contain train/ and val/ directories. "
            "See docs/ACNE_TRAINING.md."
        )

    train_classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    val_classes = sorted(path.name for path in val_dir.iterdir() if path.is_dir())
    if len(train_classes) != 2 or train_classes != val_classes:
        raise ValueError(
            "Acne training requires exactly the same two folders in train/ and val/. "
            f"Found train={train_classes}, val={val_classes}."
        )
    if not any("acne" in name.lower() for name in train_classes):
        raise ValueError(
            "One class folder must contain 'Acne' in its name, for example Acne."
        )

    counts = count_images(train_dir)
    if min(counts.values(), default=0) < 20:
        raise ValueError(
            "Each class needs at least 20 real images before training. "
            f"Current train counts: {counts}"
        )
    return train_dir, val_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a focused acne classifier")
    parser.add_argument("--data", default="data/acne", help="Dataset root")
    parser.add_argument("--output", default="models/acne_model.tflite")
    parser.add_argument("--labels", default="models/acne_labels.txt")
    parser.add_argument("--metadata", default="models/acne_model_metadata.json")
    parser.add_argument("--report", default="reports/acne_training_report.json")
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data)
    train_dir, val_dir = require_binary_dataset(root)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    test_dir = root / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=args.seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = train_ds.class_names
    if len(class_names) != 2:
        raise ValueError(f"Expected two classes, found {class_names}")

    counts = count_images(train_dir)
    total = sum(counts.values())
    class_weights = {
        index: total / (2 * max(counts[class_name], 1))
        for index, class_name in enumerate(class_names)
    }
    print(f"Classes: {class_names}")
    print(f"Train counts: {counts}")
    print(f"Class weights: {class_weights}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="acne_augmentation",
    )
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.3, patience=2, min_lr=1e-7
        ),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=2)],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Fine tune only the final part of MobileNetV2.  Keeping BatchNorm frozen
    # makes small medical-image datasets less unstable.
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=2)],
    )
    if args.fine_tune_epochs > 0:
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )
        for key, values in history_fine.history.items():
            history.history.setdefault(key, []).extend(values)

    test_metrics = {}
    if test_dir.is_dir() and sorted(path.name for path in test_dir.iterdir() if path.is_dir()) == class_names:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMAGE_SIZE,
            batch_size=args.batch_size,
            label_mode="categorical",
            shuffle=False,
        ).prefetch(autotune)
        values = model.evaluate(test_ds, return_dict=True, verbose=0)
        test_metrics = {key: float(value) for key, value in values.items()}
        print(f"Test metrics: {test_metrics}")
    else:
        print("No matching test/ directory found; validation is not a final test score.")

    output_path = Path(args.output)
    labels_path = Path(args.labels)
    metadata_path = Path(args.metadata)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_path.write_bytes(converter.convert())
    labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "source": "ACNE04 or an explicitly licensed/consented project dataset",
                "architecture": "MobileNetV2 transfer learning with final-layer fine tuning",
                "input_size": list(IMAGE_SIZE),
                # MobileNetV2 preprocessing is embedded in the Keras graph, so
                # the exported TFLite model receives raw 0..255 RGB pixels.
                "input_preprocessing": "raw_0_255",
                "class_names": class_names,
                "positive_class": next(name for name in class_names if "acne" in name.lower()),
                "scope": "Educational acne screening prototype; not a medical diagnosis",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "classes": class_names,
                "train_counts": counts,
                "class_weights": class_weights,
                "validation_best_auc": float(max(history.history.get("val_auc", [0.0]))),
                "test_metrics": test_metrics,
                "warning": "Do not make medical claims from this prototype; test on an independent phone-camera set.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved acne TFLite model: {output_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
