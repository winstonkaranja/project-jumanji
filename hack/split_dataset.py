import os
import random
import shutil

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/images/test", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/test", exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    random.shuffle(images)

    train_cut = int(len(images) * train_ratio)
    val_cut = int(len(images) * (train_ratio + val_ratio))

    splits = {
        'train': images[:train_cut],
        'val': images[train_cut:val_cut],
        'test': images[val_cut:]
    }

    for split, files in splits.items():
        for img in files:
            label = img.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            shutil.copy(f"{image_dir}/{img}", f"{output_dir}/images/{split}/{img}")
            if os.path.exists(f"{label_dir}/{label}"):
                shutil.copy(f"{label_dir}/{label}", f"{output_dir}/labels/{split}/{label}")

if __name__ == "__main__":
    split_dataset(
        image_dir='Dataset/JPEGImages',
        label_dir='Dataset/labels',  # you should have XML converted to YOLO txt!
        output_dir='Dataset/output_dataset'
    )
