from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from src.tools import util

# =============================
# 1. Functional modules
# =============================

def parse_row_col(filename: str) -> tuple[int, int]:
    """从 filename 中解析 patch 的行列索引"""
    name = filename.replace(".png", "")
    parts = name.split("_")
    return int(parts[-2]), int(parts[-1])


def load_mapping(csv_path: Path) -> dict:
    """读取 filename 与原始图片的映射"""
    df = pd.read_csv(csv_path)
    return {row["filename"]: row["original_filename"] for _, row in df.iterrows()}


def generate_label_heatmap(df: pd.DataFrame, grid_rows: int, grid_cols: int) -> np.ndarray:
    """生成 label 热图（单列 label）"""
    heatmap = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
    for _, row in df.iterrows():
        if row["label"] != 1:
            continue
        r, c = parse_row_col(row["filename"])
        heatmap[r, c] = 1
    return heatmap


def generate_score_heatmap(df: pd.DataFrame, grid_rows: int, grid_cols: int) -> np.ndarray:
    """生成连续 score 热图"""
    score_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for _, row in df.iterrows():
        r, c = parse_row_col(row["filename"])
        score_map[r, c] = row["score"]
    return score_map


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray,
                             color=(0, 0, 255), alpha=0.3, beta=0.7) -> np.ndarray:
    """将二值热图叠加到原图"""
    mask = np.zeros_like(image)
    mask[heatmap == 1] = color
    overlay = cv2.addWeighted(image, beta, mask, alpha, 0)
    return overlay


def overlay_score_map_on_image(image: np.ndarray, score_map: np.ndarray,
                               alpha=0.4, beta=0.6) -> np.ndarray:
    """将连续 score 热图叠加到原图"""
    score_map_norm = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, beta, heatmap_color, alpha, 0)
    return overlay


# =============================
# 2. Dataset Processing
# =============================

def process_dataset(input_dir: Path, raw_dir: Path, sample_map: dict):
    """处理整个数据集，生成 label 热图和 score 热图"""
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

        # -----------------------------
        # 1. Label heatmap
        # -----------------------------
        label_heatmap = generate_label_heatmap(df, grid_rows, grid_cols)
        label_heatmap_resized = cv2.resize(label_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
        label_overlay = overlay_heatmap_on_image(original_img, label_heatmap_resized)
        output_path = folder / f"{sample_name}_label_overlay.png"
        cv2.imwrite(str(output_path), label_overlay)

        # -----------------------------
        # 2. Score heatmap
        # -----------------------------
        score_map = generate_score_heatmap(df, grid_rows, grid_cols)
        score_map_resized = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_NEAREST)
        score_overlay = overlay_score_map_on_image(original_img, score_map_resized)
        output_path = folder / f"{sample_name}_score_overlay.png"
        cv2.imwrite(str(output_path), score_overlay)

        print("Finished")


# =============================
# 3. Interface
# =============================

def visualize():
    root_dir = util.get_root_dir()

    # ---- Train set ----
    train_raw_dir = root_dir / "data/raw/train_img"
    train_labels_dir = root_dir / "data/samples_labels"
    train_info_path = root_dir / "data/normalized/samples_info.csv"
    train_map = load_mapping(train_info_path)
    process_dataset(train_labels_dir, train_raw_dir, train_map)

    # ---- Valid set ----
    valid_raw_dir = root_dir / "data/raw/valid_img"
    valid_labels_dir = root_dir / "data/valid_samples_labels"
    valid_info_path = root_dir / "data/valid_normalized/valid_samples_info.csv"
    valid_map = load_mapping(valid_info_path)
    process_dataset(valid_labels_dir, valid_raw_dir, valid_map)


# =============================
# 4. Main
# =============================

if __name__ == "__main__":
    visualize()