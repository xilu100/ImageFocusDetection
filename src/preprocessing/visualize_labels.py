import os

import cv2
import numpy as np
import pandas as pd


def visualize():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)

    raw_dir = os.path.join(root_dir, "data/raw/train_img")
    input_dir = os.path.join(root_dir, "data/samples_labels")
    samples_info_path = os.path.join(root_dir, "data/normalized/samples_info.csv")

    samples_info = pd.read_csv(samples_info_path)

    # Map sample name -> original filename
    sample_to_original = {
        row["filename"]: row["original_filename"]
        for _, row in samples_info.iterrows()
    }

    # Iterate over each sampleX_labels folder
    for folder in os.listdir(input_dir):

        if not folder.endswith("_labels"):
            continue

        sample_name = folder.replace("_labels", "")
        csv_path = os.path.join(input_dir, folder, f"{sample_name}_laplacian.csv")

        if not os.path.exists(csv_path):
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
        original_path = os.path.join(raw_dir, original_filename)

        if not os.path.exists(original_path):
            print(f"Original image not found: {original_path}")
            continue

        original_img = cv2.imread(original_path)
        h, w = original_img.shape[:2]

        # Infer patch size from current_size in samples_info
        current_size = samples_info[samples_info["filename"] == original_key]["current_size"].values[0]
        cur_w, cur_h = map(int, current_size.split("x"))

        # Estimate grid size from max row/col
        rows = []
        cols = []

        for name in df["filename"]:
            parts = name.replace(".png", "").split("_")
            rows.append(int(parts[1]))
            cols.append(int(parts[2]))

        max_row = max(rows)
        max_col = max(cols)

        patch_h = cur_h // (max_row + 1)
        patch_w = cur_w // (max_col + 1)

        # Resize original to current_size for alignment
        resized_img = cv2.resize(original_img, (cur_w, cur_h))

        for thresh_id in range(1, 5):

            mask = np.zeros_like(resized_img)

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

            overlay = cv2.addWeighted(resized_img, 0.7, mask, 0.3, 0)

            output_path = os.path.join(
                input_dir,
                folder,
                f"{sample_name}_thresh{thresh_id}_overlay.png"
            )

            cv2.imwrite(output_path, overlay)

        print(f"Finished {sample_name}")


if __name__ == "__main__":
    visualize()
