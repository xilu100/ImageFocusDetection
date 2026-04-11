import csv
from pathlib import Path

import cv2
import numpy as np

from src.tools import util


# Pad image dimensions to be divisible by patch size.
def resize_img(img: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# Convert a BGR image to single-channel grayscale.
def grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Append one processing record to the metadata CSV.
def save_info(csv_path: Path, info: dict):
    file_exists = csv_path.exists()
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'original_size', 'current_size', 'original_filename'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(info)


# Read already processed original filenames to support resume runs.
def load_processed_files(csv_path: Path) -> set:
    processed = set()
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row['original_filename'])
    return processed


# Collect image files with supported extensions in stable order.
def get_image_paths(input_dir: Path, extensions=None):
    if extensions is None:
        extensions = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in extensions:
        paths.extend(sorted(input_dir.glob(ext)))
    return paths


# Ensure the target directory exists before writing outputs.
def prepare_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


# Normalize one source image and record its metadata.
def process_single_image(img_path: Path,
                         output_dir: Path,
                         sample_counter: int,
                         patch_size: int,
                         csv_path: Path,
                         prefix: str = "sample"):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Can not read: {img_path}")
        return False

    original_h, original_w = img.shape[:2]
    img = resize_img(img, patch_size)
    img = grayscale(img)
    current_h, current_w = img.shape[:2]

    output_filename = f"{prefix}{sample_counter}.png"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), img)

    info = {
        'filename': output_filename,
        'original_size': f"{original_w}x{original_h}",
        'current_size': f"{current_w}x{current_h}",
        'original_filename': img_path.name
    }
    save_info(csv_path, info)
    print(f"Processed {img_path.name} -> {output_filename}")
    return True


# Process all images in one split and skip previously finished files.
def process_image_set(input_dir: Path,
                      output_dir: Path,
                      csv_path: Path,
                      patch_size: int,
                      prefix: str):
    prepare_output_dir(output_dir)
    processed_files = load_processed_files(csv_path)
    img_paths = get_image_paths(input_dir)

    sample_counter = 1
    for img_path in img_paths:
        if img_path.name in processed_files:
            print(f"Skip already processed: {img_path.name}")
            continue
        success = process_single_image(img_path, output_dir, sample_counter, patch_size, csv_path, prefix)
        if success:
            sample_counter += 1


# Run normalization for both training and validation datasets.
def normalize_images(patch_size=32):
    root_dir = util.get_root_dir()

    # Train set
    train_input_dir = root_dir / "data" / "raw" / "train_img"
    train_output_dir = root_dir / "data" / "normalized"
    train_csv_path = train_output_dir / "samples_info.csv"
    process_image_set(train_input_dir, train_output_dir, train_csv_path, patch_size, prefix="sample")

    # Valid set
    valid_input_dir = root_dir / "data" / "raw" / "valid_img"
    valid_output_dir = root_dir / "data" / "valid_normalized"
    valid_csv_path = valid_output_dir / "valid_samples_info.csv"
    process_image_set(valid_input_dir, valid_output_dir, valid_csv_path, patch_size, prefix="valid_sample")


if __name__ == "__main__":
    normalize_images(patch_size=32)
