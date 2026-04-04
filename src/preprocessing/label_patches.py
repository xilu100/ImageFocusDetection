import csv
from pathlib import Path

import cv2
import numpy as np

from src.tools import util


# =============================
# 1. Score Components
# =============================

def tenengrad_score(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g2 = gx**2 + gy**2
    return np.mean(g2)


def fft_bandpass_score(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(1 + np.abs(fshift))

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

    r_low = min(h, w) // 20
    r_high = min(h, w) // 5

    mask = (dist > r_low) & (dist < r_high)

    return mag[mask].mean()


def local_contrast_score(gray: np.ndarray) -> float:
    mean = cv2.GaussianBlur(gray, (9, 9), 0)
    contrast = (gray - mean) ** 2
    return np.mean(contrast)


# =============================
# 2. Robust Normalize
# =============================

def robust_normalize_array(values: np.ndarray) -> np.ndarray:
    if len(values) < 10:
        return values

    p1, p99 = np.percentile(values, [1, 99])
    return np.clip((values - p1) / (p99 - p1 + 1e-8), 0, 1)


# =============================
# 3. Compute Scores (Batch)
# =============================

def compute_scores_batch(images: list):
    grad_list = []
    fft_list = []
    contrast_list = []

    for gray in images:
        grad_list.append(tenengrad_score(gray))
        fft_list.append(fft_bandpass_score(gray))
        contrast_list.append(local_contrast_score(gray))

    grad_arr = np.array(grad_list)
    fft_arr = np.array(fft_list)
    contrast_arr = np.array(contrast_list)

    # --- normalize ---
    grad_n = robust_normalize_array(grad_arr)
    fft_n = robust_normalize_array(fft_arr)
    contrast_n = robust_normalize_array(contrast_arr)

    # --- fusion（非线性）---
    scores = (grad_n * fft_n * contrast_n) ** (1 / 3)

    return scores, grad_arr, fft_arr, contrast_arr


# =============================
# 4. CSV Writing
# =============================

def write_scores_csv(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "score",
            "grad",
            "fft",
            "contrast",
            "label"
        ])
        writer.writerows(rows)


# =============================
# 5. Dataset Processing
# =============================

def process_dataset(input_dir: Path, output_dir: Path, top_percent=85):
    for sample_folder in input_dir.iterdir():
        if not sample_folder.is_dir():
            continue

        print(f"Processing: {sample_folder.name}")

        sample_output_dir = output_dir / f"{sample_folder.name}_labels"
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = sample_output_dir / f"{sample_folder.name}.csv"

        images = []
        filenames = []

        # --- load images ---
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

        # --- compute scores ---
        scores, grads, ffts, contrasts = compute_scores_batch(images)

        # --- dynamic threshold（核心）---
        threshold = np.percentile(scores, top_percent)

        # --- build rows ---
        rows = []
        for i in range(len(images)):
            label = 1 if scores[i] >= threshold else 0

            rows.append([
                filenames[i],
                float(scores[i]),
                float(grads[i]),
                float(ffts[i]),
                float(contrasts[i]),
                label
            ])

        # --- save ---
        write_scores_csv(csv_path, rows)
        print(f"Saved: {csv_path}")


# =============================
# 6. Interface
# =============================

def label(top_percent=85):
    root_dir = util.get_root_dir()

    train_input_dir = root_dir / "data/samples"
    train_output_dir = root_dir / "data/samples_labels"

    valid_input_dir = root_dir / "data/valid_samples"
    valid_output_dir = root_dir / "data/valid_samples_labels"

    process_dataset(train_input_dir, train_output_dir, top_percent)
    process_dataset(valid_input_dir, valid_output_dir, top_percent)


# =============================
# 7. Main
# =============================

if __name__ == "__main__":
    label(top_percent=85)