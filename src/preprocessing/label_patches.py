import csv
from pathlib import Path

import cv2
import numpy as np

from src.tools import util


# Score Calculation
def laplacian_score(gray_image: np.ndarray) -> float:
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def fft_score(gray_image: np.ndarray) -> float:
    f = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    h, w = gray_image.shape
    ch, cw = h // 2, w // 2
    radius = min(h, w) // 10

    mask = np.ones_like(magnitude)
    mask[ch - radius:ch + radius, cw - radius:cw + radius] = 0

    high_freq_energy = (magnitude * mask).sum()
    total_energy = magnitude.sum()

    return high_freq_energy / (total_energy + 1e-8)


def normalize(x: float, min_val: float, max_val: float) -> float:
    return (x - min_val) / (max_val - min_val + 1e-8)


def compute_combined_score(gray_image: np.ndarray,
                           lap_min=0, lap_max=1000,
                           fft_min=0, fft_max=1,
                           alpha=0.5) -> tuple:
    lap = laplacian_score(gray_image)
    fft = fft_score(gray_image)

    lap_norm = normalize(lap, lap_min, lap_max)
    fft_norm = normalize(fft, fft_min, fft_max)

    score = alpha * lap_norm + (1 - alpha) * fft_norm
    return score, lap, fft


# CSV processing
def write_scores_csv(csv_path: Path, scores: list, thresholds: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "combined_score", "Laplacian", "fft"]
        header += [f"y_thresh{i + 1}" for i in range(len(thresholds))]
        writer.writerow(header)
        for row in scores:
            writer.writerow(row)


# Processing a single image (dataset)
def process_dataset(input_dir: Path, output_dir: Path, thresholds: list):
    for sample_folder in input_dir.iterdir():
        if not sample_folder.is_dir():
            continue

        sample_output_dir = output_dir / f"{sample_folder.name}_labels"
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = sample_output_dir / f"{sample_folder.name}_combined.csv"
        scores = []

        for file_path in sample_folder.iterdir():
            if not file_path.is_file():
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Can not read : {file_path}")

            score, lap, fft = compute_combined_score(img)
            row = [file_path.name, score, lap, fft]
            row += [1 if score >= t else 0 for t in thresholds]
            scores.append(row)

        write_scores_csv(csv_path, scores, thresholds)
        print(f"Saved: {csv_path}")


# Interface
def label(thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6]
    root_dir = util.get_root_dir()

    train_input_dir = root_dir / "data/samples"
    train_output_dir = root_dir / "data/samples_labels"

    valid_input_dir = root_dir / "data/valid_samples"
    valid_output_dir = root_dir / "data/valid_samples_labels"

    process_dataset(train_input_dir, train_output_dir, thresholds)
    process_dataset(valid_input_dir, valid_output_dir, thresholds)


if __name__ == "__main__":
    label(thresholds=[0.3, 0.4, 0.5, 0.6])
