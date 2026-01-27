import json
import math
import os
from fractions import Fraction

import cv2


def segmentation(target_blocks=10000):
    # Paths
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    input_dir = os.path.join(parent_dir, "normalised")
    output_dir = os.path.join(parent_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)

    # Load records.json
    record_path = os.path.join(input_dir, "records.json")
    with open(record_path, "r") as f:
        records = json.load(f)

    for record in records:
        sample_name = record["sample"]

        # Original image file
        original_file = f"{sample_name}.jpg"
        img_path = os.path.join(input_dir, original_file)
        img_path = os.path.abspath(img_path).replace("\\", "/")  # avoid path issues

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        # Create output folder
        sample_output_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        height, width = img.shape[:2]

        # Compute image ratio n:m
        ratio = Fraction(width, height).limit_denominator(100)
        n, m = ratio.numerator, ratio.denominator

        # Compute t so that n*m*t^2 ≈ 100
        # target_blocks = 10000
        t = max(1, round(math.sqrt(target_blocks / (n * m))))

        # Compute rows and columns
        cols = n * t
        rows = m * t

        # Compute block size (try to make square blocks)
        block_w = width // cols
        block_h = height // rows
        block_size = min(block_w, block_h)

        # Split image into blocks
        for row in range(rows):
            for col in range(cols):
                x1 = col * block_size
                y1 = row * block_size
                x2 = min(x1 + block_size, width)
                y2 = min(y1 + block_size, height)

                block = img[y1:y2, x1:x2]
                block_filename = f"{sample_name}_{row}_{col}.jpg"
                cv2.imwrite(os.path.join(sample_output_dir, block_filename), block)

        print(f"Finished {sample_name}: {rows} rows x {cols} cols, total ~{rows * cols} blocks")


if __name__ == "__main__":
    segmentation()
