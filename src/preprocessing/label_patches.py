import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from tools import util

LAP_ANCHOR_DEFAULT = 300.0
SOBEL_ANCHOR_DEFAULT = 12.0
TEXTURE_STD_THRESHOLD = 3.0
TEXTURE_SOBEL_THRESHOLD = 2.0


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


def squash_score(value: float, anchor: float) -> float:
    value = max(0.0, float(value))
    anchor = max(1e-8, float(anchor))
    return float(value / (value + anchor))


def is_textureless_patch(
        img: np.ndarray,
        sobel_score: float,
        std_threshold: float = TEXTURE_STD_THRESHOLD,
        sobel_threshold: float = TEXTURE_SOBEL_THRESHOLD,
) -> bool:
    intensity_std = float(np.std(img.astype(np.float32)))
    return intensity_std < std_threshold and sobel_score < sobel_threshold


# Fuse multiple focus cues with size-aware weighting rules.
def fuse_score(size, lap, sobel, fft):
    if size <= 32:
        return 0.7 * sobel + 0.3 * lap

    elif size <= 128:
        return 0.5 * lap + 0.5 * fft

    else:
        return 0.7 * fft + 0.3 * lap


def compute_absolute_focus_score(
        size: int,
        lap_score: float,
        sobel_score: float,
        fft_score: float,
        lap_anchor: float,
        sobel_anchor: float,
) -> float:
    lap_norm = squash_score(lap_score, lap_anchor)
    sobel_norm = squash_score(sobel_score, sobel_anchor)
    fft_norm = float(np.clip(fft_score, 0.0, 1.0))
    return float(fuse_score(size, lap_norm, sobel_norm, fft_norm))


# Generate per-patch focus metrics and fused scores.
def score_cal(images, lap_anchor: float = LAP_ANCHOR_DEFAULT, sobel_anchor: float = SOBEL_ANCHOR_DEFAULT):
    lap_scores = []
    sobel_scores = []
    fft_scores = []
    sizes = []
    textureless_flags = []

    for img in images:
        h, w = img.shape
        size = int(min(h, w))
        sizes.append(size)

        lap_scores.append(compute_laplacian(img))
        sobel_score = compute_sobel(img)
        sobel_scores.append(sobel_score)
        textureless_flags.append(is_textureless_patch(img, sobel_score))

        if size >= 32:
            fft_scores.append(compute_fft(img))
        else:
            fft_scores.append(0.0)

    total_scores = []
    for i in range(len(images)):
        score = compute_absolute_focus_score(
            sizes[i],
            lap_scores[i],
            sobel_scores[i],
            fft_scores[i],
            lap_anchor,
            sobel_anchor,
        )
        total_scores.append(score)

    return lap_scores, sobel_scores, fft_scores, total_scores, textureless_flags


# Write patch scores and labels into a structured CSV file.
def write_scores_csv(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "lap_score",
            "sobel_score",
            "fft_score",
            "total_score",
            "label"
        ])
        writer.writerows(rows)


# Label patches in each sample folder using absolute thresholding.
# - If sharp_threshold > blur_threshold: tri-class labels {0, 1, 2} (+ -1 textureless).
# - If sharp_threshold == blur_threshold: binary labels {0, 1} (+ -1 textureless).
def process_single_folder(
        sample_folder: Path,
        output_dir: Path,
        sharp_threshold: float,
        blur_threshold: float,
        lap_anchor: float,
        sobel_anchor: float,
):
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
        return f"Skipped (no images): {sample_folder.name}"

    lap_scores, sobel_scores, fft_scores, total_scores, textureless_flags = score_cal(
        images, lap_anchor=lap_anchor, sobel_anchor=sobel_anchor
    )

    rows = []
    binary_mode = np.isclose(sharp_threshold, blur_threshold, atol=1e-8)
    for i in range(len(images)):
        if textureless_flags[i]:
            patch_label = -1
        else:
            if binary_mode:
                patch_label = 1 if total_scores[i] >= sharp_threshold else 0
            elif total_scores[i] >= sharp_threshold:
                patch_label = 1
            elif total_scores[i] < blur_threshold:
                patch_label = 0
            else:
                patch_label = 2

        rows.append([
            filenames[i],
            float(lap_scores[i]),
            float(sobel_scores[i]),
            float(fft_scores[i]),
            float(total_scores[i]),
            patch_label
        ])

    write_scores_csv(csv_path, rows)
    return f"Saved: {csv_path}"


def process_dataset(
        input_dir: Path,
        output_dir: Path,
        sharp_threshold=0.75,
        blur_threshold=0.4,
        lap_anchor: float = LAP_ANCHOR_DEFAULT,
        sobel_anchor: float = SOBEL_ANCHOR_DEFAULT,
        max_workers: int | None = None,
):
    if not (0.0 <= blur_threshold <= 1.0 and 0.0 <= sharp_threshold <= 1.0):
        raise ValueError("blur_threshold and sharp_threshold must be in [0, 1].")
    if blur_threshold > sharp_threshold:
        raise ValueError("blur_threshold must be smaller than or equal to sharp_threshold.")
    if np.isclose(blur_threshold, sharp_threshold, atol=1e-8):
        print(
            f"Binary labeling mode enabled: blur_threshold == sharp_threshold == {sharp_threshold:.3f}"
        )

    sample_folders = [folder for folder in input_dir.iterdir() if folder.is_dir()]
    if not sample_folders:
        return

    worker_count = max_workers if max_workers is not None else max(1, (os.cpu_count() or 1) // 2)
    if worker_count is None or worker_count <= 1:
        for folder in sample_folders:
            message = process_single_folder(
                folder,
                output_dir,
                sharp_threshold,
                blur_threshold,
                lap_anchor,
                sobel_anchor,
            )
            print(message)
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_single_folder,
                folder,
                output_dir,
                sharp_threshold,
                blur_threshold,
                lap_anchor,
                sobel_anchor,
            )
            for folder in sample_folders
        ]
        for future in as_completed(futures):
            print(future.result())


# Run automatic labeling for train and validation sample sets.
def label(
        sharp_threshold=0.75,
        blur_threshold=0.4,
        lap_anchor: float = LAP_ANCHOR_DEFAULT,
        sobel_anchor: float = SOBEL_ANCHOR_DEFAULT,
        max_workers=None,
):
    # Backward compatibility: older pipeline may still pass 75/10 style percentage values.
    if sharp_threshold > 1.0 or blur_threshold > 1.0:
        sharp_threshold = float(sharp_threshold) / 100.0
        blur_threshold = float(blur_threshold) / 100.0
        print(
            "Deprecated percentage thresholds detected. Converted to score thresholds: "
            f"sharp_threshold={sharp_threshold:.3f}, blur_threshold={blur_threshold:.3f}"
        )

    root_dir = util.get_root_dir()

    train_input_dir = root_dir / "data/samples"
    train_output_dir = root_dir / "data/samples_labels"

    valid_input_dir = root_dir / "data/valid_samples"
    valid_output_dir = root_dir / "data/valid_samples_labels"

    process_dataset(
        train_input_dir,
        train_output_dir,
        sharp_threshold,
        blur_threshold,
        lap_anchor=lap_anchor,
        sobel_anchor=sobel_anchor,
        max_workers=max_workers,
    )
    process_dataset(
        valid_input_dir,
        valid_output_dir,
        sharp_threshold,
        blur_threshold,
        lap_anchor=lap_anchor,
        sobel_anchor=sobel_anchor,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    label(sharp_threshold=0.75, blur_threshold=0.4)
