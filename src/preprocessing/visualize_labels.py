from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PATCH_SIZE = 32


def parse_row_col(filename: str):
    name = filename.replace(".png", "")
    parts = name.split("_")
    return int(parts[-2]), int(parts[-1])


def visualize():
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    parent_dir = current_dir.parent
    root_dir = parent_dir.parent

    raw_dir = root_dir / "data/raw/train_img"
    input_dir = root_dir / "data/samples_labels"
    samples_info_path = root_dir / "data/normalized/samples_info.csv"

    samples_info = pd.read_csv(samples_info_path)

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

        print(f"\nProcessing {sample_name}")

        df = pd.read_csv(csv_path)

        if len(df) == 0:
            continue

        key = f"{sample_name}.png"

        if key not in sample_to_original:
            print("Original mapping missing")
            continue

        original_path = raw_dir / sample_to_original[key]

        if not original_path.exists():
            print("Original image missing")
            continue

        original_img = cv2.imread(str(original_path))
        h, w = original_img.shape[:2]

        # =========================
        # 解析 row / col
        # =========================

        rows = []
        cols = []

        for name in df["filename"]:
            r, c = parse_row_col(name)
            rows.append(r)
            cols.append(c)

        grid_rows = max(rows) + 1
        grid_cols = max(cols) + 1

        print("patch grid:", grid_rows, grid_cols)

        # =========================
        # 每个阈值生成 heatmap
        # =========================

        for thresh_id in range(1, 5):

            heatmap = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

            for _, row in df.iterrows():

                if row[f"y_thresh{thresh_id}"] != 1:
                    continue

                r, c = parse_row_col(row["filename"])

                heatmap[r, c] = 1

            # =========================
            # resize 到原图尺寸
            # =========================

            heatmap_img = cv2.resize(
                heatmap,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            mask = np.zeros_like(original_img)

            mask[heatmap_img == 1] = (0, 0, 255)

            overlay = cv2.addWeighted(
                original_img,
                0.7,
                mask,
                0.3,
                0
            )

            output_path = folder / f"{sample_name}_thresh{thresh_id}_overlay.png"

            cv2.imwrite(str(output_path), overlay)

        print("Finished")


if __name__ == "__main__":
    visualize()
