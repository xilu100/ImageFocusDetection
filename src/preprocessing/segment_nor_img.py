from pathlib import Path

import cv2
import pandas as pd


def segment_images(patch_size=32):
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    # 两套数据配置：训练集 + 验证集
    configs = [
        {
            "input_dir": root_dir / "data/normalized",
            "output_dir": root_dir / "data/samples",
            "prefix": "sample",
            "csv_name": "samples_info.csv"  # 训练集 CSV
        },
        {
            "input_dir": root_dir / "data/valid_normalized",
            "output_dir": root_dir / "data/valid_samples",
            "prefix": "valid_sample",
            "csv_name": "valid_samples_info.csv"  # 验证集 CSV
        }
    ]

    # 遍历配置处理每个数据集
    for cfg in configs:
        input_dir = cfg["input_dir"]
        output_dir = cfg["output_dir"]
        prefix = cfg["prefix"]
        csv_name = cfg["csv_name"]

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = input_dir / csv_name
        df = pd.read_csv(csv_path)

        for idx, img_name in enumerate(df["filename"], start=1):
            img_path = input_dir / img_name
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"Cannot read image: {img_path}")
                continue

            h, w = img.shape[:2]
            n_rows = h // patch_size
            n_cols = w // patch_size

            if n_rows == 0 or n_cols == 0:
                print(f"Image too small: {img_name}")
                continue

            img_output_dir = output_dir / f"{prefix}{idx}"
            img_output_dir.mkdir(parents=True, exist_ok=True)

            total_patches = 0

            for row in range(n_rows):
                for col in range(n_cols):
                    y_start = row * patch_size
                    x_start = col * patch_size

                    patch = img[
                        y_start:y_start + patch_size,
                        x_start:x_start + patch_size
                    ]

                    patch_name = f"{prefix}{idx}_{row}_{col}.png"
                    patch_path = img_output_dir / patch_name

                    cv2.imwrite(str(patch_path), patch)
                    total_patches += 1

            print(
                f"[{prefix}] {img_name} -> {prefix}{idx}: "
                f"{n_rows}x{n_cols}, patches={total_patches}"
            )


if __name__ == "__main__":
    segment_images(patch_size=32)
