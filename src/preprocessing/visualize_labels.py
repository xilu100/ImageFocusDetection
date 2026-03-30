from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def parse_row_col(filename: str):
    name = filename.replace(".png", "")
    parts = name.split("_")
    return int(parts[-2]), int(parts[-1])


def visualize():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    # 原图路径
    train_raw_dir = root_dir / "data/raw/train_img"
    valid_raw_dir = root_dir / "data/raw/valid_img"

    # 映射表
    train_info_path = root_dir / "data/normalized/samples_info.csv"
    valid_info_path = root_dir / "data/valid_normalized/valid_samples_info.csv"

    # 读取映射表
    train_info = pd.read_csv(train_info_path)
    train_map = {row["filename"]: row["original_filename"] for _, row in train_info.iterrows()}

    valid_info = pd.read_csv(valid_info_path)
    valid_map = {row["filename"]: row["original_filename"] for _, row in valid_info.iterrows()}

    # 支持两套数据
    input_dirs = [
        (root_dir / "data/samples_labels", train_map, train_raw_dir),
        (root_dir / "data/valid_samples_labels", valid_map, valid_raw_dir)
    ]

    for input_dir, sample_to_original, raw_dir in input_dirs:
        for folder in input_dir.iterdir():
            if not folder.is_dir() or not folder.name.endswith("_labels"):
                continue

            sample_name = folder.name.replace("_labels", "")
            csv_path = folder / f"{sample_name}_combined.csv"  # 改为 combined 文件

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
            rows, cols = [], []
            for name in df["filename"]:
                r, c = parse_row_col(name)
                rows.append(r)
                cols.append(c)

            grid_rows = max(rows) + 1
            grid_cols = max(cols) + 1

            print("patch grid:", grid_rows, grid_cols)

            # =========================
            # 阈值图 heatmap
            # =========================
            for thresh_id in range(1, 5):
                heatmap = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

                for _, row in df.iterrows():
                    if row[f"y_thresh{thresh_id}"] != 1:
                        continue
                    r, c = parse_row_col(row["filename"])
                    heatmap[r, c] = 1

                # resize 到原图尺寸
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

            # =========================
            # 连续图 heatmap
            # =========================
            score_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            for _, row in df.iterrows():
                r, c = parse_row_col(row["filename"])
                score_map[r, c] = row["combined_score"]

            # 归一化到 0~255
            score_map_norm = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
            score_map_norm = score_map_norm.astype(np.uint8)

            heatmap_color = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)

            heatmap_resized = cv2.resize(
                heatmap_color,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            overlay_score = cv2.addWeighted(
                original_img,
                0.6,
                heatmap_resized,
                0.4,
                0
            )

            output_path = folder / f"{sample_name}_score_overlay.png"
            cv2.imwrite(str(output_path), overlay_score)

            print("Finished")


if __name__ == "__main__":
    visualize()