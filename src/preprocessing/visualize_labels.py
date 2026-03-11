from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def visualize():
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    parent_dir = current_dir.parent
    root_dir = parent_dir.parent

    raw_dir = root_dir / "data/raw/train_img"
    input_dir = root_dir / "data/samples_labels"
    samples_info_path = root_dir / "data/normalized/samples_info.csv"

    samples_info = pd.read_csv(samples_info_path)

    # Map sample name -> original filename
    sample_to_original = {
        row["filename"]: row["original_filename"]
        for _, row in samples_info.iterrows()
    }

    for folder in input_dir.iterdir():
        if not folder.is_dir() or not folder.name.endswith("_labels"):
            continue

        sample_name = folder.name.replace("_labels", "")
        csv_path = folder / f"{sample_name}_laplacian.csv"

        if not csv_path.exists():
            continue

        print(f"Processing {sample_name}")
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            continue

        # Get original image path
        original_key = f"{sample_name}.png"
        if original_key not in sample_to_original:
            print(f"Original mapping not found for {sample_name}")
            continue

        original_filename = sample_to_original[original_key]
        original_path = raw_dir / original_filename

        if not original_path.exists():
            print(f"Original image not found: {original_path}")
            continue

        original_img = cv2.imread(str(original_path))
        h, w = original_img.shape[:2]

        # Estimate grid size from max row/col in CSV
        rows = []
        cols = []

        for name in df["filename"]:
            parts = name.replace(".png", "").split("_")
            rows.append(int(parts[1]))
            cols.append(int(parts[2]))

        max_row = max(rows)
        max_col = max(cols)

        # Patch size based on original image size
        patch_h = h // (max_row + 1)
        patch_w = w // (max_col + 1)

        for thresh_id in range(1, 5):
            mask = np.zeros_like(original_img)

            for _, row in df.iterrows():
                if row[f"y_thresh{thresh_id}"] == 1:
                    parts = row["filename"].replace(".png", "").split("_")
                    r = int(parts[1])
                    c = int(parts[2])

                    y1 = r * patch_h
                    y2 = y1 + patch_h
                    x1 = c * patch_w
                    x2 = x1 + patch_w

                    cv2.rectangle(
                        mask,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        -1
                    )

            overlay = cv2.addWeighted(original_img, 0.7, mask, 0.3, 0)

            output_path = folder / f"{sample_name}_thresh{thresh_id}_overlay.png"

            cv2.imwrite(str(output_path), overlay)

        print(f"Finished {sample_name}")


if __name__ == "__main__":
    visualize()