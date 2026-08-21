import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw

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

def create_sample_image(class_name: str, path: Path):
    img = Image.new("RGB", (224, 224), color=(random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)))
    draw = ImageDraw.Draw(img)
    
    if class_name == "Normal_Skin":
        skin_base = (random.randint(210, 245), random.randint(170, 210), random.randint(140, 180))
        img = Image.new("RGB", (224, 224), color=skin_base)
    elif class_name == "Other_Non_Skin":
        for i in range(0, 224, 20):
            draw.rectangle([i, 0, i+10, 224], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    else:
        skin_base = (230, 195, 170)
        img = Image.new("RGB", (224, 224), color=skin_base)
        draw = ImageDraw.Draw(img)
        spot_color = (random.randint(80, 140), random.randint(40, 90), random.randint(30, 80))
        draw.ellipse([80, 80, 144, 144], fill=spot_color)
        
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def main():
    parser = argparse.ArgumentParser(description="Set up skin disease classification dataset structure.")
    parser.add_argument("--data", default="data/skin", help="Root directory for dataset")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample synthetic images for testing")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sample images per class per split")
    args = parser.parse_args()

    root = Path(args.data)
    splits = ["train", "val"]

    print(f"Setting up dataset folder structure in: {root.resolve()}")
    for split in splits:
        for cls_name in DEFAULT_CLASSES:
            cls_dir = root / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            if args.generate_samples:
                for i in range(args.num_samples):
                    img_path = cls_dir / f"sample_{i+1:03d}.jpg"
                    create_sample_image(cls_name, img_path)

    if args.generate_samples:
        print(f"Successfully generated {args.num_samples} sample images per class for train and val!")
    else:
        print("\nDataset structure ready. Place your image files inside the class folders under train/ and val/.")

if __name__ == "__main__":
    main()
