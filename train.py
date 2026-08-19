import argparse
from pathlib import Path
import tensorflow as tf

def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--output', default='models/skin_model.tflite'); p.add_argument('--labels', default='models/labels.txt'); p.add_argument('--epochs', type=int, default=8); args = p.parse_args()
    root = Path(args.data); train_dir=root/'train'; val_dir=root/'val'
    train = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(224,224), batch_size=32, label_mode='categorical')
    val = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(224,224), batch_size=32, label_mode='categorical')
    labels = train.class_names; Path(args.labels).parent.mkdir(parents=True, exist_ok=True); Path(args.labels).write_text('\n'.join(labels), encoding='utf-8')
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet'); base.trainable=False
    model = tf.keras.Sequential([tf.keras.layers.Rescaling(1./127.5, offset=-1), base, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(.2), tf.keras.layers.Dense(len(labels), activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']); model.fit(train, validation_data=val, epochs=args.epochs)
    converter=tf.lite.TFLiteConverter.from_keras_model(model); Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_bytes(converter.convert()); print(f'Saved {args.output} and {args.labels}')
if __name__ == '__main__': main()
