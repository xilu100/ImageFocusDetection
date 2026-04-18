import csv
from pathlib import Path

import cv2
import numpy as np

from tools import util


# Measure sharpness using Laplacian response variance.
def compute_laplacian(img: np.ndarray) -> float:
    img = img.astype(np.float32)
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return float(lap.var())


# Estimate edge strength from Sobel gradient magnitude.
def compute_sobel(img: np.ndarray) -> float:
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag))


# Compute the ratio of high-frequency energy in the Fourier domain.
def compute_fft(img: np.ndarray) -> float:
    img = img.astype(np.float32)

    img = (img - img.mean()) / (img.std() + 1e-8)

    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)

    magnitude = np.log1p(np.abs(f_shift))

    h, w = img.shape
    cy, cx = h // 2, w // 2

    radius = int(min(cy, cx) * 0.3)

    y, x = np.ogrid[:h, :w]
    mask = ((y - cy) ** 2 + (x - cx) ** 2) >= radius ** 2

    return float(magnitude[mask].sum() / (magnitude.sum() + 1e-8))


# Scale score arrays to [0, 1] for fair fusion.
def normalize(arr):
    arr = np.array(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


# Fuse multiple focus cues with size-aware weighting rules.
def fuse_score(size, lap, sobel, fft):
    if size <= 32:
        return 0.7 * sobel + 0.3 * lap

    elif size <= 128:
        return 0.5 * lap + 0.5 * fft

    else:
        return 0.7 * fft + 0.3 * lap


# Generate per-patch focus metrics and fused scores.
def score_cal(images):
    lap_scores = []
    sobel_scores = []
    fft_scores = []
    sizes = []

    for img in images:
        h, w = img.shape
        size = int(min(h, w))
        sizes.append(size)

        lap_scores.append(compute_laplacian(img))
        sobel_scores.append(compute_sobel(img))

        if size >= 32:
            fft_scores.append(compute_fft(img))
        else:
            fft_scores.append(0.0)

    lap_n = normalize(lap_scores)
    sobel_n = normalize(sobel_scores)
    fft_n = normalize(fft_scores)

    total_scores = []
    for i in range(len(images)):
        score = fuse_score(
            sizes[i],
            lap_n[i],
            sobel_n[i],
            fft_n[i]
        )
        total_scores.append(score)

    return lap_scores, fft_scores, total_scores


# Write patch scores and labels into a structured CSV file.
def write_scores_csv(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "lap_score",
            "fft_score",
            "total_score",
            "label"
        ])
        writer.writerows(rows)


# Label patches in each sample folder using percentile thresholding.
def process_dataset(input_dir: Path, output_dir: Path, top_percent=85, low_percent=10):
    if not (0 <= low_percent <= 100 and 0 <= top_percent <= 100):
        raise ValueError("low_percent and top_percent must be in [0, 100].")
    if low_percent >= top_percent:
        raise ValueError("low_percent must be smaller than top_percent.")

    for sample_folder in input_dir.iterdir():
        if not sample_folder.is_dir():
            continue

        print(f"Processing: {sample_folder.name}")

        sample_output_dir = output_dir / f"{sample_folder.name}_labels"
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = sample_output_dir / f"{sample_folder.name}.csv"

        images = []
        filenames = []

        for file_path in sample_folder.iterdir():
            if not file_path.is_file():
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            images.append(img)
            filenames.append(file_path.name)

        if len(images) == 0:
            continue

        lap_scores, fft_scores, total_scores = score_cal(images)

        top_threshold = np.percentile(total_scores, top_percent)
        low_threshold = np.percentile(total_scores, low_percent)

        rows = []
        for i in range(len(images)):
            if total_scores[i] >= top_threshold:
                patch_label = 1
            elif total_scores[i] <= low_threshold:
                patch_label = -1
            else:
                patch_label = 0

            rows.append([
                filenames[i],
                float(lap_scores[i]),
                float(fft_scores[i]),
                float(total_scores[i]),
                patch_label
            ])

        write_scores_csv(csv_path, rows)
        print(f"Saved: {csv_path}")


# Run automatic labeling for train and validation sample sets.
def label(top_percent=85, low_percent=10):
    root_dir = util.get_root_dir()

    train_input_dir = root_dir / "data/samples"
    train_output_dir = root_dir / "data/samples_labels"

    valid_input_dir = root_dir / "data/valid_samples"
    valid_output_dir = root_dir / "data/valid_samples_labels"

    process_dataset(train_input_dir, train_output_dir, top_percent, low_percent)
    process_dataset(valid_input_dir, valid_output_dir, top_percent, low_percent)


if __name__ == "__main__":
    label(top_percent=85, low_percent=10)

