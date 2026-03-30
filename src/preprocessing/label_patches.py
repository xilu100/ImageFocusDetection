import csv
from pathlib import Path

import cv2
import numpy as np


# -----------------------------
# 1. Laplacian 清晰度
# -----------------------------
def compute_laplacian_score(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


# -----------------------------
# 2. FFT 高频能量占比
# -----------------------------
def compute_fft_score(gray_image):
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray_image.shape
    ch, cw = h // 2, w // 2

    radius = min(h, w) // 10  # 低频区域大小

    mask = np.ones_like(magnitude)
    mask[ch - radius:ch + radius, cw - radius:cw + radius] = 0

    high_freq_energy = (magnitude * mask).sum()
    total_energy = magnitude.sum()

    return high_freq_energy / (total_energy + 1e-8)


# -----------------------------
# 3. 归一化函数
# -----------------------------
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val + 1e-8)


# -----------------------------
# 4. 融合评分
# -----------------------------
def compute_combined_score(gray_image,
                           lap_min=0,
                           lap_max=1000,
                           fft_min=0,
                           fft_max=1,
                           alpha=0.5):
    lap = compute_laplacian_score(gray_image)
    fft = compute_fft_score(gray_image)

    lap_norm = normalize(lap, lap_min, lap_max)
    fft_norm = normalize(fft, fft_min, fft_max)

    score = alpha * lap_norm + (1 - alpha) * fft_norm

    return score, lap, fft


# -----------------------------
# 5. 主流程
# -----------------------------
def label(thresholds=None):
    # ✅ 4个阈值（0~1区间）
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6]

    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

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

    for cfg in configs:
        input_dir = cfg["input_dir"]
        output_root = cfg["output_root"]

        output_root.mkdir(parents=True, exist_ok=True)

        for sample_folder in input_dir.iterdir():
            if not sample_folder.is_dir():
                continue

            sample_output_dir = output_root / f"{sample_folder.name}_labels"
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = sample_output_dir / f"{sample_folder.name}_combined.csv"

            with csv_path.open(mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)

                # Header
                header = ["filename", "combined_score", "laplacian", "fft"]
                for i in range(len(thresholds)):
                    header.append(f"y_thresh{i + 1}")
                writer.writerow(header)

                for file_path in sample_folder.iterdir():
                    if not file_path.is_file():
                        continue

                    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    score, lap, fft = compute_combined_score(image)

                    row = [file_path.name, score, lap, fft]

                    # 4个阈值打标签
                    for thresh in thresholds:
                        row.append(1 if score >= thresh else 0)

                    writer.writerow(row)

            print(f"Saved: {csv_path}")


# -----------------------------
# 6. 入口
# -----------------------------
if __name__ == "__main__":
    label(thresholds=[0.3, 0.4, 0.5, 0.6])