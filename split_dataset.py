import os
import shutil
import random
from pathlib import Path

# Path asal dan tujuan
source_dir = Path("D:/laskarAI/terra-scan/dataset")
output_dir = Path("D:/laskarAI/terra-scan/dataset_split")

train_ratio = 0.8  # 80% training, 20% testing

# Buat folder output
for split in ["train", "test"]:
    for class_folder in source_dir.iterdir():
        if class_folder.is_dir():
            target_dir = output_dir / split / class_folder.name
            target_dir.mkdir(parents=True, exist_ok=True)

# Proses pembagian file
for class_folder in source_dir.iterdir():
    if class_folder.is_dir():
        images = list(class_folder.glob("*.jpg"))
        random.shuffle(images)

        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        for img in train_images:
            shutil.copy(img, output_dir / "train" / class_folder.name / img.name)

        for img in test_images:
            shutil.copy(img, output_dir / "test" / class_folder.name / img.name)

print("✅ Dataset berhasil dipisah ke folder 'dataset_split/train' dan 'dataset_split/test'")