"""
prepare_data.py

Efficient data preparation script for large datasets.
✅ Doesn't load all images into memory.
✅ Splits dataset into train/test folders.
✅ Ready for use with ImageDataGenerator.
"""

import os
import shutil
import random
from tqdm import tqdm

# ------------- CONFIG -------------
DATASET_DIR = "combined_dataset"
OUTPUT_DIR = "dataset_ready"
SPLIT_RATIO = 0.8  # 80% train, 20% test
CLASSES = ["WithMask", "WithoutMask"]
# ----------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def prepare_dataset():
    print("🔍 Starting dataset preparation...")
    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")

    # Create output directories
    for subset in [train_dir, test_dir]:
        for cls in CLASSES:
            ensure_dir(os.path.join(subset, cls))

    for cls in CLASSES:
        print(f"\nProcessing class: {cls}")
        src_folder = os.path.join(DATASET_DIR, cls)
        all_images = [os.path.join(src_folder, f) for f in os.listdir(src_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"📦 Found {len(all_images)} images in {cls}")
        random.shuffle(all_images)

        split_index = int(len(all_images) * SPLIT_RATIO)
        train_images = all_images[:split_index]
        test_images = all_images[split_index:]

        print(f"📂 Copying {len(train_images)} train and {len(test_images)} test images...")

        for img_path in tqdm(train_images, desc=f"{cls} Train"):
            shutil.copy2(img_path, os.path.join(train_dir, cls, os.path.basename(img_path)))

        for img_path in tqdm(test_images, desc=f"{cls} Test"):
            shutil.copy2(img_path, os.path.join(test_dir, cls, os.path.basename(img_path)))

    print("\n✅ Dataset prepared successfully!")
    print(f"📁 Train directory: {train_dir}")
    print(f"📁 Test directory: {test_dir}")

if __name__ == "__main__":
    prepare_dataset()


# cd FACE_MASK_DETECTOR
# .\venv\Scripts\Activate.ps1
# python prepare_data.py



