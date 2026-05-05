from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BINARY_LABELS = ["Blur", "Sharp"]
TRICLASS_LABELS = ["Blur", "Sharp", "Intermediate"]


FIGURE_SPECS = [
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/Decision_Tree_sample_percentage.csv",
        "output_name": "cm_main_2c_decision_tree.png",
        "labels": BINARY_LABELS,
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/Random_Forest_sample_percentage.csv",
        "output_name": "cm_main_2c_random_forest.png",
        "labels": BINARY_LABELS,
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/SVM_sample_percentage.csv",
        "output_name": "cm_main_2c_svm.png",
        "labels": BINARY_LABELS,
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/CNN_sample_percentage.csv",
        "output_name": "cm_main_2c_cnn.png",
        "labels": BINARY_LABELS,
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/Decision_Tree_sample_percentage.csv",
        "output_name": "cm_main_3c_decision_tree.png",
        "labels": TRICLASS_LABELS,
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/Random_Forest_sample_percentage.csv",
        "output_name": "cm_main_3c_random_forest.png",
        "labels": TRICLASS_LABELS,
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/SVM_sample_percentage.csv",
        "output_name": "cm_main_3c_svm.png",
        "labels": TRICLASS_LABELS,
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/CNN_sample_percentage.csv",
        "output_name": "cm_main_3c_cnn.png",
        "labels": TRICLASS_LABELS,
    },
]


def load_confusion_matrix(csv_path: Path, sample_percentage: int = 100) -> np.ndarray:
    df = pd.read_csv(csv_path)
    row = df.loc[df["sample_percentage"] == sample_percentage]
    if row.empty:
        raise ValueError(f"No sample_percentage={sample_percentage} row found in {csv_path}")
    row = row.iloc[0]

    cm_size = int(row["cm_size"])
    matrix = np.zeros((cm_size, cm_size), dtype=int)
    for r_idx in range(cm_size):
        for c_idx in range(cm_size):
            matrix[r_idx, c_idx] = int(row[f"cm_r{r_idx}_c{c_idx}"])
    return matrix


def choose_style(cm_size: int) -> dict[str, float]:
    if cm_size == 2:
        return {
            "figsize": 6.4,
            "tick_fontsize": 17,
            "count_fontsize": 24,
            "pct_fontsize": 15,
            "axis_label_fontsize": 16,
            "x_tick_rotation": 0,
        }
    return {
        "figsize": 9.4,
        "tick_fontsize": 19,
        "count_fontsize": 22,
        "pct_fontsize": 15,
        "axis_label_fontsize": 18,
        "x_tick_rotation": 26,
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    style = choose_style(cm.shape[0])
    row_sums = cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(style["figsize"], style["figsize"]))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(
        axis="x",
        labelsize=style["tick_fontsize"],
        rotation=style["x_tick_rotation"],
    )
    ax.tick_params(axis="y", labelsize=style["tick_fontsize"])
    ax.set_xlabel("Predicted", fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("True", fontsize=style["axis_label_fontsize"])

    threshold = cm.max() * 0.55
    for r_idx in range(cm.shape[0]):
        for c_idx in range(cm.shape[1]):
            value = cm[r_idx, c_idx]
            row_total = int(row_sums[r_idx, 0])
            pct = 0.0 if row_total == 0 else value / row_total * 100.0
            text_color = "white" if value >= threshold else "#111111"
            ax.text(
                c_idx,
                r_idx - 0.08,
                f"{value:,}",
                ha="center",
                va="center_baseline",
                fontsize=style["count_fontsize"],
                fontweight="semibold",
                color=text_color,
            )
            ax.text(
                c_idx,
                r_idx + 0.20,
                f"{pct:.1f}%",
                ha="center",
                va="center_baseline",
                fontsize=style["pct_fontsize"],
                color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=max(style["tick_fontsize"] - 2, 12))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "Bachelor_Thesis" / "imagefocus" / "images"

    for spec in FIGURE_SPECS:
        csv_path = root_dir / spec["csv_path"]
        output_path = output_dir / spec["output_name"]
        cm = load_confusion_matrix(csv_path, sample_percentage=100)
        plot_confusion_matrix(cm, spec["labels"], output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
