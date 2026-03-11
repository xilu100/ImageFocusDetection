import math
from pathlib import Path

import cv2
import pandas as pd


def segment_images(estimated_patches=5000):
    # Directories
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    parent_dir = current_dir.parent
    root_dir = parent_dir.parent

    input_dir = root_dir / "data/normalized"
    output_dir = root_dir / "data/samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    csv_path = input_dir / "samples_info.csv"
    df = pd.read_csv(csv_path)

    # Process each image
    for img_name in df["filename"]:
        img_path = input_dir / img_name

        aspect_ratio_str = df[df["filename"] == img_name]["aspect_ratio"].values[0]
        width_ratio, height_ratio = map(int, aspect_ratio_str.split(":"))

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]

        # Calculate rows and columns to approximate square patches
        ratio_total = width_ratio / height_ratio
        n_cols = math.sqrt(estimated_patches * ratio_total)
        n_rows = estimated_patches / n_cols

        n_cols = int(round(n_cols))
        n_rows = int(round(n_rows))

        # Patch size (square)
        patch_size_w = w // n_cols
        patch_size_h = h // n_rows
        patch_size = min(patch_size_w, patch_size_h)

        # Create output folder for the image
        img_base = Path(img_name).stem
        img_output_dir = output_dir / img_base
        img_output_dir.mkdir(parents=True, exist_ok=True)

        total_patches = 0

        # Split patches
        for row in range(n_rows):
            for col in range(n_cols):
                y_start = row * patch_size
                x_start = col * patch_size
                y_end = min(y_start + patch_size, h)
                x_end = min(x_start + patch_size, w)

                patch = img[y_start:y_end, x_start:x_end]

                patch_name = f"{img_base}_{row}_{col}.png"
                patch_path = img_output_dir / patch_name

                cv2.imwrite(str(patch_path), patch)
                total_patches += 1

        print(
            f"{img_name} split completed: "
            f"{n_rows} rows x {n_cols} cols, "
            f"patch size={patch_size}, "
            f"total patches={total_patches}"
        )


if __name__ == "__main__":
    segment_images(estimated_patches=5000)