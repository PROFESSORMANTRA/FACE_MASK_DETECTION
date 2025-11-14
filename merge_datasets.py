#!/usr/bin/env python3
"""
merge_datasets.py
Final version for Soham’s FACE_MASK_DETECTOR project.
Handles deeply nested folders (e.g., data3/00000, data3/10000)
and classifies correctly worn masks vs. incorrect/without masks.
"""

import os
import shutil
from tqdm import tqdm
import logging

# ---------------- CONFIG ----------------
DATASETS = ["data1", "data2", "data3"]
OUTPUT_DIR = "combined_dataset"
CLASS_WITH = "WithMask"
CLASS_WITHOUT = "WithoutMask"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".jfif", ".webp")
LOGFILE = "merge_log.txt"
# ----------------------------------------


# Setup logger
logging.basicConfig(filename=LOGFILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# Keywords to detect mask type
WITH_KEYWORDS = [
    "with_mask", "withmask", "mask", "masked", "face_mask", "mask_correct", "maskon"
]

WITHOUT_KEYWORDS = [
    "without_mask", "withoutmask", "no_mask", "nomask", "incorrect_mask",
    "mask_incorrect", "mask_chin", "mask_mouth_chin", "mask_nose", "mask_nose_mouth",
    "mask_under_nose", "maskchin", "maskmouthchin", "maskmouth", "mouth_chin"
]


def classify_file(fname):
    """
    Classify based on filename keywords.
    Returns: "with", "without", or None.
    """
    name = fname.lower()

    # If has both mask and incorrect indicators → without
    for w in WITHOUT_KEYWORDS:
        if w in name:
            return "without"

    # If has 'mask' but not incorrect → with
    if "mask" in name:
        return "with"

    return None


def safe_copy(src, dst_folder):
    ensure_dir(dst_folder)
    base, ext = os.path.splitext(os.path.basename(src))
    target = os.path.join(dst_folder, base + ext)
    counter = 1
    while os.path.exists(target):
        target = os.path.join(dst_folder, f"{base}_{counter}{ext}")
        counter += 1
    try:
        shutil.copy2(src, target)
        return True
    except Exception as e:
        logging.exception(f"Failed copying {src}: {e}")
        return False


def merge_datasets():
    with_dir = os.path.join(OUTPUT_DIR, CLASS_WITH)
    without_dir = os.path.join(OUTPUT_DIR, CLASS_WITHOUT)
    ensure_dir(with_dir)
    ensure_dir(without_dir)

    total_with, total_without, total_skipped = 0, 0, 0

    print("🔍 Starting deep merge of datasets (recursively scanning all subfolders)...\n")

    for ds in DATASETS:
        if not os.path.exists(ds):
            print(f"⚠️ Dataset path not found: {ds}")
            continue

        print(f"📂 Merging from: {ds}")
        for root, dirs, files in os.walk(ds):
            image_files = [f for f in files if f.lower().endswith(IMAGE_EXTS)]
            for fname in tqdm(image_files, desc=os.path.basename(root) or ds, unit="img"):
                src = os.path.join(root, fname)
                cls = classify_file(fname)

                if cls == "with":
                    if safe_copy(src, with_dir):
                        total_with += 1
                elif cls == "without":
                    if safe_copy(src, without_dir):
                        total_without += 1
                else:
                    total_skipped += 1
                    logging.info(f"Skipped unclassified: {src}")

    print("\n✅ Merge completed successfully.")
    print(f"🟢 Total WithMask images: {total_with}")
    print(f"🔴 Total WithoutMask images: {total_without}")
    print(f"⚪ Skipped/unreadable files: {total_skipped}")
    print(f"📄 Log saved to: {LOGFILE}")


if __name__ == "__main__":
    merge_datasets()


# cd FACE_MASK_DETECTOR
# .\venv\Scripts\Activate.ps1
# python merge_datasets.py

