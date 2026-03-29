import csv
from pathlib import Path

import cv2


def compute_laplacian_score(gray_image):
    """
    Compute Laplacian variance score for a grayscale image.
    Higher value means sharper image.
    """
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def label(thresholds=None):
    # ---- Define thresholds internally ----
    if thresholds is None:
        thresholds = [200, 210, 220, 230]

    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    # ✅ 两套数据配置（samples + valid_samples）
    configs = [
        {
            "input_dir": root_dir / "data" / "samples",
            "output_root": root_dir / "data" / "samples_labels",
        },
        {
            "input_dir": root_dir / "data" / "valid_samples",
            "output_root": root_dir / "data" / "valid_samples_labels",
        }
    ]

    # ✅ 统一处理逻辑
    for cfg in configs:
        input_dir = cfg["input_dir"]
        output_root = cfg["output_root"]

        output_root.mkdir(parents=True, exist_ok=True)

        for sample_folder in input_dir.iterdir():
            if not sample_folder.is_dir():
                continue

            sample_output_dir = output_root / f"{sample_folder.name}_labels"
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = sample_output_dir / f"{sample_folder.name}_laplacian.csv"

            with csv_path.open(mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)

                # Header
                header = ["filename", "laplacian_score"]
                for i in range(len(thresholds)):
                    header.append(f"y_thresh{i + 1}")
                writer.writerow(header)

                for file_path in sample_folder.iterdir():
                    if not file_path.is_file():
                        continue

                    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    score = compute_laplacian_score(image)

                    row = [file_path.name, score]

                    for thresh in thresholds:
                        row.append(1 if score >= thresh else 0)

                    writer.writerow(row)

            print(f"Saved: {csv_path}")


if __name__ == "__main__":
    label(thresholds=[200, 210, 220, 230])
