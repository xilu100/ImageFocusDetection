from pathlib import Path

import cv2
import pandas as pd


def segment_images(patch_size=32):
    # Directories
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    input_dir = root_dir / "data/normalized"
    output_dir = root_dir / "data/samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    csv_path = input_dir / "samples_info.csv"
    df = pd.read_csv(csv_path)

    # Process each image
    for img_name in df["filename"]:
        img_path = input_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]

        # 计算可完整切出的行列数
        n_rows = h // patch_size
        n_cols = w // patch_size

        if n_rows == 0 or n_cols == 0:
            print(f"Image too small to cut 64x64 patches: {img_name}")
            continue

        # 创建输出文件夹
        img_base = Path(img_name).stem
        img_output_dir = output_dir / img_base
        img_output_dir.mkdir(parents=True, exist_ok=True)

        total_patches = 0

        # 按64x64切块
        for row in range(n_rows):
            for col in range(n_cols):
                y_start = row * patch_size
                x_start = col * patch_size
                patch = img[y_start:y_start + patch_size, x_start:x_start + patch_size]

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
    segment_images(patch_size=32)
