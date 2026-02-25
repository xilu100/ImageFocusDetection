import math
import os

import cv2
import pandas as pd


def segment_images(estimated_patches=5000):
    # Directories
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    input_dir = os.path.join(root_dir, 'data/normalized')
    output_dir = os.path.join(root_dir, 'data/samples')
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    csv_path = os.path.join(input_dir, 'samples_info.csv')
    df = pd.read_csv(csv_path)

    # Process each image
    for img_name in df['filename']:
        img_path = os.path.join(input_dir, img_name)
        aspect_ratio_str = df[df['filename'] == img_name]['aspect_ratio'].values[0]  # e.g., "3:2"
        width_ratio, height_ratio = map(int, aspect_ratio_str.split(':'))

        img = cv2.imread(img_path)
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
        img_base = os.path.splitext(img_name)[0]
        img_output_dir = os.path.join(output_dir, img_base)
        os.makedirs(img_output_dir, exist_ok=True)

        total_patches = 0
        # Split patches (0-based coordinates for easy reconstruction)
        for row in range(n_rows):
            for col in range(n_cols):
                y_start = row * patch_size
                x_start = col * patch_size
                y_end = min(y_start + patch_size, h)
                x_end = min(x_start + patch_size, w)

                patch = img[y_start:y_end, x_start:x_end]
                patch_name = f"{img_base}_{row}_{col}.png"  # 0-based
                cv2.imwrite(os.path.join(img_output_dir, patch_name), patch)
                total_patches += 1

        print(
            f"{img_name} split completed: {n_rows} rows x {n_cols} cols, patch size={patch_size}, total patches={total_patches}")


if __name__ == "__main__":
    segment_images(estimated_patches=5000)
