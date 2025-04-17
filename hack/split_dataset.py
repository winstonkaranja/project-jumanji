import os
import random
import shutil

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    # Create output directories
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/images/test", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/test", exist_ok=True)

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Get all image files (case insensitive)
    images = []
    for f in os.listdir(image_dir):
        lowercase_f = f.lower()
        if lowercase_f.endswith('.jpg') or lowercase_f.endswith('.jpeg') or lowercase_f.endswith('.png'):
            images.append(f)

    # Shuffle images
    random.shuffle(images)

    # Calculate split points
    train_cut = int(len(images) * train_ratio)
    val_cut = int(len(images) * (train_ratio + val_ratio))

    # Create splits
    splits = {
        'train': images[:train_cut],
        'val': images[train_cut:val_cut],
        'test': images[val_cut:]
    }

    # Track statistics
    stats = {'total_images': len(images), 'images_with_labels': 0, 'images_without_labels': 0}
    for split, files in splits.items():
        stats[f'{split}_images'] = len(files)
        stats[f'{split}_with_labels'] = 0

    # Copy files to their respective directories
    for split, files in splits.items():
        for img in files:
            # Get base name without extension
            base_name = os.path.splitext(img)[0]
            label = f"{base_name}.txt"
            
            # Copy image
            shutil.copy(f"{image_dir}/{img}", f"{output_dir}/images/{split}/{img}")
            
            # Copy label if it exists
            label_path = f"{label_dir}/{label}"
            if os.path.exists(label_path):
                # Check if label file is not empty
                if os.path.getsize(label_path) > 0:
                    shutil.copy(label_path, f"{output_dir}/labels/{split}/{label}")
                    stats['images_with_labels'] += 1
                    stats[f'{split}_with_labels'] += 1
                else:
                    print(f"Warning: Empty label file for {img}")
                    stats['images_without_labels'] += 1
            else:
                print(f"Warning: No label file found for {img}")
                stats['images_without_labels'] += 1

    # Print statistics
    print(f"\nDataset Split Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Images with labels: {stats['images_with_labels']}")
    print(f"Images without labels: {stats['images_without_labels']}")
    print(f"\nTrain set: {stats['train_images']} images ({stats['train_with_labels']} with labels)")
    print(f"Validation set: {stats['val_images']} images ({stats['val_with_labels']} with labels)")
    print(f"Test set: {stats['test_images']} images ({stats['test_with_labels']} with labels)")

    return stats

if __name__ == "__main__":
    split_dataset(
        image_dir='Dataset/JPEGImages',
        label_dir='Dataset/labels',
        output_dir='Dataset/output_dataset',
        random_seed=42  # For reproducibility
    )