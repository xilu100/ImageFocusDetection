from pathlib import Path

import cv2
import pandas as pd

from tools import util


# Image slices
# Split one image into a grid of fixed-size patches and keep coordinates.
def slice_image(img, patch_size: int):
    h, w = img.shape[:2]
    n_rows = h // patch_size
    n_cols = w // patch_size

    if n_rows == 0 or n_cols == 0:
        return []

    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * patch_size
            x_start = col * patch_size
            patch = img[y_start:y_start + patch_size, x_start:x_start + patch_size]
            patches.append((row, col, patch))
    return patches, n_rows, n_cols


# Single image processing
# Save all patches from one image into its own output folder.
def process_single_image(img_path: Path,
                         output_dir: Path,
                         prefix: str,
                         idx: int,
                         patch_size: int):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Can not read : {img_path}")
        return 0, 0, 0

    patches, n_rows, n_cols = slice_image(img, patch_size)
    if not patches:
        print(f"Image too small: {img_path.name}")
        return 0, 0, 0

    img_output_dir = output_dir / f"{prefix}{idx}"
    img_output_dir.mkdir(parents=True, exist_ok=True)

    for row, col, patch in patches:
        patch_name = f"{prefix}{idx}_{row}_{col}.png"
        patch_path = img_output_dir / patch_name
        cv2.imwrite(str(patch_path), patch)

    total_patches = len(patches)
    print(f"[{prefix}] {img_path.name} -> {prefix}{idx}: {n_rows}x{n_cols}, patches={total_patches}")
    return n_rows, n_cols, total_patches


# Dataset processing
# Iterate over normalized images listed in CSV and segment them.
def process_dataset(input_dir: Path,
                    output_dir: Path,
                    prefix: str,
                    csv_name: str,
                    patch_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = input_dir / csv_name
    df = pd.read_csv(csv_path)

    for idx, img_name in enumerate(df["filename"], start=1):
        img_path = input_dir / img_name
        process_single_image(img_path, output_dir, prefix, idx, patch_size)


# Interface
# Execute patch extraction for train and validation splits.
def segment_images(patch_size=32):
    root_dir = util.get_root_dir()

    # Train set
    train_input_dir = root_dir / "data/normalized"
    train_output_dir = root_dir / "data/samples"
    train_prefix = "sample"
    train_csv_name = "samples_info.csv"
    process_dataset(train_input_dir, train_output_dir, train_prefix, train_csv_name, patch_size)

    # Valid set
    valid_input_dir = root_dir / "data/valid_normalized"
    valid_output_dir = root_dir / "data/valid_samples"
    valid_prefix = "valid_sample"
    valid_csv_name = "valid_samples_info.csv"
    process_dataset(valid_input_dir, valid_output_dir, valid_prefix, valid_csv_name, patch_size)


if __name__ == "__main__":
    segment_images(patch_size=32)
