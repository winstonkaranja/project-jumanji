import os
import shutil
from collections import Counter
import random

def balance_dataset(label_dir, image_dir, output_label_dir, output_image_dir, target_count=500):
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    class_files = {}
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    class_files.setdefault(cls_id, []).append(file)
                    break  # only count the first class per file (assumes single-class per file)

    for cls_id, files in class_files.items():
        current_count = len(files)
        if current_count >= target_count:
            sampled = random.sample(files, target_count)
        else:
            sampled = files.copy()
            while len(sampled) < target_count:
                sampled.append(random.choice(files))

        for idx, file in enumerate(sampled):
            base_name = file.replace('.txt', '')
            image_file = f"{base_name}.jpg"
            label_src = os.path.join(label_dir, file)
            image_src = os.path.join(image_dir, image_file)

            new_name = f"{base_name}_{idx:04d}"

            label_dst = os.path.join(output_label_dir, f"{new_name}.txt")
            image_dst = os.path.join(output_image_dir, f"{new_name}.jpg")

            shutil.copyfile(label_src, label_dst)
            shutil.copyfile(image_src, image_dst)

    print("✅ Dataset balanced and saved to:", output_label_dir, output_image_dir)

# Paths
label_dir = 'Dataset/labels'
image_dir = 'Dataset/JPEGImages'
output_label_dir = 'Dataset/output_dataset_balanced/labels'
output_image_dir = 'Dataset/output_dataset_balanced/images'

# Run the function
balance_dataset(label_dir, image_dir, output_label_dir, output_image_dir)
