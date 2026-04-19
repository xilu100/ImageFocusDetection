from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from tools import pca, util


# Extract patch grid coordinates encoded in a patch filename.
def parse_row_col(filename: str) -> tuple[int, int]:
    name = filename.replace(".png", "")
    parts = name.split("_")
    return int(parts[-2]), int(parts[-1])


# Build a lookup from normalized filenames to original source names.
def load_mapping(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {row["filename"]: row["original_filename"] for _, row in df.iterrows()}


# Create a tri-state map: -1 (uncertain / very blurry), 0 (blurry), 1 (sharp).
def generate_label_heatmap(df: pd.DataFrame, grid_rows: int, grid_cols: int) -> np.ndarray:
    heatmap = np.full((grid_rows, grid_cols), fill_value=99, dtype=np.int16)
    for _, row in df.iterrows():
        r, c = parse_row_col(row["filename"])
        heatmap[r, c] = int(row["label"])
    return heatmap


# Place continuous focus scores onto their patch grid positions.
def generate_score_heatmap(df: pd.DataFrame, grid_rows: int, grid_cols: int) -> np.ndarray:
    score_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for _, row in df.iterrows():
        r, c = parse_row_col(row["filename"])
        score_map[r, c] = row["total_score"]
    return score_map


# Blend a tri-state label mask onto the original image for inspection.
def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray,
                             sharp_color=(0, 0, 255), uncertain_color=(255, 0, 0),
                             alpha=0.3) -> np.ndarray:
    overlay = image.copy()

    for label, color in ((1, sharp_color), (-1, uncertain_color)):
        region = heatmap == label
        if not np.any(region):
            continue

        color_patch = np.zeros_like(image, dtype=np.uint8)
        color_patch[:] = color
        blended = cv2.addWeighted(image, 1 - alpha, color_patch, alpha, 0)
        overlay[region] = blended[region]

    return overlay


# Render and blend a colorized score map onto the original image.
def overlay_score_map_on_image(image: np.ndarray, score_map: np.ndarray,
                               alpha=0.4, beta=0.6) -> np.ndarray:
    score_map_norm = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, beta, heatmap_color, alpha, 0)
    return overlay


# Generate visual overlays for every labeled sample folder.
def process_single_label_folder(folder: Path, raw_dir: Path, samples_dir: Path, sample_map: dict, patch_size: int = 32):
    sample_name = folder.name.replace("_labels", "")
    csv_path = folder / f"{sample_name}.csv"

    if not csv_path.exists():
        return f"Skipped (csv missing): {sample_name}"

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return f"Skipped (empty csv): {sample_name}"

    key = f"{sample_name}.png"
    if key not in sample_map:
        return f"Skipped (original mapping missing): {sample_name}"

    original_path = raw_dir / sample_map[key]
    if not original_path.exists():
        return f"Skipped (original image missing): {sample_name}"

    original_img = cv2.imread(str(original_path))
    if original_img is None:
        return f"Skipped (cannot read original): {sample_name}"
    h, w = original_img.shape[:2]

    parsed = [parse_row_col(name) for name in df["filename"]]
    rows, cols = zip(*parsed)
    rows = [int(r) for r in rows]
    cols = [int(c) for c in cols]
    grid_rows, grid_cols = max(rows) + 1, max(cols) + 1

    label_heatmap = generate_label_heatmap(df, grid_rows, grid_cols)
    label_heatmap_resized = cv2.resize(label_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    label_overlay = overlay_heatmap_on_image(original_img, label_heatmap_resized)
    output_path = folder / f"{sample_name}_label_overlay.png"
    cv2.imwrite(str(output_path), label_overlay)

    score_map = generate_score_heatmap(df, grid_rows, grid_cols)
    score_map_resized = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_NEAREST)
    score_overlay = overlay_score_map_on_image(original_img, score_map_resized)
    output_path = folder / f"{sample_name}_score_overlay.png"
    cv2.imwrite(str(output_path), score_overlay)

    sample_folder = samples_dir / sample_name
    X, y = collect_sample_features_for_pca(df, sample_folder, patch_size)
    if X is not None:
        save_pca_2d_plot(
            X=X,
            y=y,
            output_path=folder / f"{sample_name}_pca_2d_distribution.png",
            title=f"{sample_name} Patch Distribution (PCA 2D)"
        )
        save_pca_3d_plot(
            X=X,
            y=y,
            output_path=folder / f"{sample_name}_pca_3d_distribution.png",
            title=f"{sample_name} Patch Distribution (PCA 3D)"
        )

    return f"Finished: {sample_name} (grid={grid_rows}x{grid_cols})"


def process_dataset(input_dir: Path, raw_dir: Path, samples_dir: Path, sample_map: dict, patch_size: int = 32, max_workers: int | None = None):
    folders = [folder for folder in input_dir.iterdir() if folder.is_dir() and folder.name.endswith("_labels")]
    if not folders:
        return

    worker_count = max_workers if max_workers is not None else max(1, (os.cpu_count() or 1) // 2)
    if worker_count is None or worker_count <= 1:
        for folder in folders:
            print(process_single_label_folder(folder, raw_dir, samples_dir, sample_map, patch_size))
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_single_label_folder, folder, raw_dir, samples_dir, sample_map, patch_size)
            for folder in folders
        ]
        for future in as_completed(futures):
            print(future.result())


def collect_sample_features_for_pca(df: pd.DataFrame, sample_folder: Path, patch_size: int = 32):
    X_list = []
    y_list = []

    if "label" not in df.columns or "filename" not in df.columns:
        return None, None

    if not sample_folder.exists():
        return None, None

    for _, row in df.iterrows():
        img_path = sample_folder / str(row["filename"])
        if not img_path.exists():
            continue

        try:
            x = util.img_to_X(str(img_path), patch_size)
        except ValueError:
            continue

        X_list.append(x)
        y_list.append(int(row["label"]))

    if not X_list:
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def finalize_and_save_plot(output_path: Path, message_prefix: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {message_prefix}: {output_path}")


def save_pca_3d_plot(X: np.ndarray, y: np.ndarray, output_path: Path, title: str):
    if len(X) < 3:
        print(f"Skip PCA plot (need at least 3 samples): {output_path}")
        return

    X_3d, _ = pca.reduce_dimensions(X, n_components=3)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    label_style = {
        -1: ("Uncertain / Very Blurry (-1)", "tab:blue"),
        0: ("Blurry (0)", "tab:gray"),
        1: ("Sharp (1)", "tab:red"),
    }

    unique_labels = sorted(int(lbl) for lbl in np.unique(y).tolist())
    for lbl in unique_labels:
        idx = y == lbl
        label_name, color = label_style.get(lbl, (f"Label {lbl}", "black"))
        ax.scatter(
            X_3d[idx, 0],
            X_3d[idx, 1],
            X_3d[idx, 2],
            s=8,
            alpha=0.7,
            c=color,
            label=label_name
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    finalize_and_save_plot(output_path, "PCA 3D plot")


def save_pca_2d_plot(X: np.ndarray, y: np.ndarray, output_path: Path, title: str):
    if len(X) < 2:
        print(f"Skip PCA plot (need at least 2 samples): {output_path}")
        return

    X_2d, _ = pca.reduce_dimensions(X, n_components=2)

    plt.figure(figsize=(8, 6))
    label_style = {
        -1: ("Uncertain / Very Blurry (-1)", "tab:blue"),
        0: ("Blurry (0)", "tab:gray"),
        1: ("Sharp (1)", "tab:red"),
    }

    unique_labels = sorted(int(lbl) for lbl in np.unique(y).tolist())
    for lbl in unique_labels:
        idx = y == lbl
        label_name, color = label_style.get(lbl, (f"Label {lbl}", "black"))
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=8,
            alpha=0.7,
            c=color,
            label=label_name
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    finalize_and_save_plot(output_path, "PCA 2D plot")


# Run visualization for both train and validation labeling outputs.
def visualize(max_workers=None):
    root_dir = util.get_root_dir()

    train_raw_dir = root_dir / "data/raw/train_img"
    train_labels_dir = root_dir / "data/samples_labels"
    train_samples_dir = root_dir / "data/samples"
    train_info_path = root_dir / "data/normalized/samples_info.csv"
    train_map = load_mapping(train_info_path)
    process_dataset(train_labels_dir, train_raw_dir, train_samples_dir, train_map, max_workers=max_workers)

    valid_raw_dir = root_dir / "data/raw/valid_img"
    valid_labels_dir = root_dir / "data/valid_samples_labels"
    valid_samples_dir = root_dir / "data/valid_samples"
    valid_info_path = root_dir / "data/valid_normalized/valid_samples_info.csv"
    valid_map = load_mapping(valid_info_path)
    process_dataset(valid_labels_dir, valid_raw_dir, valid_samples_dir, valid_map, max_workers=max_workers)


if __name__ == "__main__":
    visualize()
