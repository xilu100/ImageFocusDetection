from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.tools import util


# Extract patch grid coordinates encoded in a patch filename.
def parse_row_col(filename: str) -> tuple[int, int]:
    name = filename.replace(".png", "")
    parts = name.split("_")
    return int(parts[-2]), int(parts[-1])


# Build a lookup from normalized filenames to original source names.
def load_mapping(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {row["filename"]: row["original_filename"] for _, row in df.iterrows()}


# Create a binary map that marks only positive-labeled patches.
def generate_label_heatmap(df: pd.DataFrame, grid_rows: int, grid_cols: int) -> np.ndarray:
    heatmap = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
    for _, row in df.iterrows():
        if row["label"] != 1:
            continue
        r, c = parse_row_col(row["filename"])
        heatmap[r, c] = 1
    return heatmap


# Place continuous focus scores onto their patch grid positions.
def generate_score_heatmap(df: pd.DataFrame, grid_rows: int, grid_cols: int) -> np.ndarray:
    score_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for _, row in df.iterrows():
        r, c = parse_row_col(row["filename"])
        score_map[r, c] = row["total_score"]
    return score_map


# Blend a binary label mask onto the original image for inspection.
def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray,
                             color=(0, 0, 255), alpha=0.3, beta=0.7) -> np.ndarray:
    mask = np.zeros_like(image)
    mask[heatmap == 1] = color
    overlay = cv2.addWeighted(image, beta, mask, alpha, 0)
    return overlay


# Render and blend a colorized score map onto the original image.
def overlay_score_map_on_image(image: np.ndarray, score_map: np.ndarray,
                               alpha=0.4, beta=0.6) -> np.ndarray:
    score_map_norm = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, beta, heatmap_color, alpha, 0)
    return overlay


# Generate visual overlays for every labeled sample folder.
def process_dataset(input_dir: Path, raw_dir: Path, sample_map: dict):
    for folder in input_dir.iterdir():
        if not folder.is_dir() or not folder.name.endswith("_labels"):
            continue

        sample_name = folder.name.replace("_labels", "")
        csv_path = folder / f"{sample_name}.csv"

        if not csv_path.exists():
            continue

        print(f"\nProcessing {sample_name}")
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue

        key = f"{sample_name}.png"
        if key not in sample_map:
            print("Original mapping missing")
            continue

        original_path = raw_dir / sample_map[key]
        if not original_path.exists():
            print("Original image missing")
            continue

        original_img = cv2.imread(str(original_path))
        if original_img is None:
            raise ValueError(f"Cannot read: {original_path}")
        h, w = original_img.shape[:2]

        parsed = [parse_row_col(name) for name in df["filename"]]
        rows, cols = zip(*parsed)
        rows = [int(r) for r in rows]
        cols = [int(c) for c in cols]

        grid_rows, grid_cols = max(rows) + 1, max(cols) + 1
        print("Patch grid:", grid_rows, grid_cols)

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

        print("Finished")


# Run visualization for both train and validation labeling outputs.
def visualize():
    root_dir = util.get_root_dir()

    train_raw_dir = root_dir / "data/raw/train_img"
    train_labels_dir = root_dir / "data/samples_labels"
    train_info_path = root_dir / "data/normalized/samples_info.csv"
    train_map = load_mapping(train_info_path)
    process_dataset(train_labels_dir, train_raw_dir, train_map)

    valid_raw_dir = root_dir / "data/raw/valid_img"
    valid_labels_dir = root_dir / "data/valid_samples_labels"
    valid_info_path = root_dir / "data/valid_normalized/valid_samples_info.csv"
    valid_map = load_mapping(valid_info_path)
    process_dataset(valid_labels_dir, valid_raw_dir, valid_map)


if __name__ == "__main__":
    visualize()
