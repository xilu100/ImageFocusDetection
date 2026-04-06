import csv
from pathlib import Path

import cv2
import numpy as np

from src.tools import util


def compute_laplacian_var(img: np.ndarray) -> float:
    img = img.astype(np.float32)
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return float(lap.var())


def compute_sobel_energy(img: np.ndarray) -> float:
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag))


def compute_fft_highfreq(img: np.ndarray) -> float:
    img = img.astype(np.float32)

    img = (img - img.mean()) / (img.std() + 1e-8)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude = np.log1p(np.abs(fshift))

    h, w = img.shape
    cy, cx = h // 2, w // 2

    radius = int(min(cy, cx) * 0.3)

    y, x = np.ogrid[:h, :w]
    mask = ((y - cy) ** 2 + (x - cx) ** 2) >= radius ** 2

    return float(magnitude[mask].sum() / (magnitude.sum() + 1e-8))


def normalize(arr):
    arr = np.array(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def fuse_score(size, lap, sobel, fft):
    if size < 32:
        return 0.7 * sobel + 0.3 * lap

    elif size <= 128:
        return 0.5 * lap + 0.5 * fft

    else:
        return 0.7 * fft + 0.3 * lap


def score_cal(images):
    lap_scores = []
    sobel_scores = []
    fft_scores = []
    sizes = []

    for img in images:
        h, w = img.shape
        size = min(h, w)
        sizes.append(size)

        lap_scores.append(compute_laplacian_var(img))
        sobel_scores.append(compute_sobel_energy(img))

        if size >= 32:
            fft_scores.append(compute_fft_highfreq(img))
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

        threshold = np.percentile(total_scores, top_percent)

        rows = []
        for i in range(len(images)):
            label = 1 if total_scores[i] >= threshold else 0

            rows.append([
                filenames[i],
                float(lap_scores[i]),
                float(fft_scores[i]),
                float(total_scores[i]),
                label
            ])

        write_scores_csv(csv_path, rows)
        print(f"Saved: {csv_path}")


def label(top_percent=85):
    root_dir = util.get_root_dir()

    train_input_dir = root_dir / "data/samples"
    train_output_dir = root_dir / "data/samples_labels"

    valid_input_dir = root_dir / "data/valid_samples"
    valid_output_dir = root_dir / "data/valid_samples_labels"

    process_dataset(train_input_dir, train_output_dir, top_percent)
    process_dataset(valid_input_dir, valid_output_dir, top_percent)


if __name__ == "__main__":
    label(top_percent=85)
