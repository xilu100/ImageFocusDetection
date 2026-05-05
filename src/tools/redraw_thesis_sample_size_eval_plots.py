from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "semibold"
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams["axes.titleweight"] = "semibold"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


PLOT_SPECS = [
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/Decision_Tree_sample_percentage.csv",
        "output_name": "DT_EVA_2C.png",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/Random_Forest_sample_percentage.csv",
        "output_name": "RF_EVA_2C.png",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/SVM_sample_percentage.csv",
        "output_name": "SVM_EVA_2C.png",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/CNN_sample_percentage.csv",
        "output_name": "CNN_EVA_2C.png",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/Decision_Tree_sample_percentage.csv",
        "output_name": "DT_EVA_3C.png",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/Random_Forest_sample_percentage.csv",
        "output_name": "RF_EVA_3C.png",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/SVM_sample_percentage.csv",
        "output_name": "SVM_EVA_3C.png",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/CNN_sample_percentage.csv",
        "output_name": "CNN_EVA_3C.png",
    },
]


CLASS_LABELS = {
    0: "Blur",
    1: "Sharp",
    2: "Intermediate",
}

CLASS_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
    2: "#ff7f0e",
}

CLASS1_METRIC_STYLES = {
    "precision": {"marker": "o", "linestyle": "-", "color": "#2ca02c"},
    "recall": {"marker": "s", "linestyle": "--", "color": "#17becf"},
    "f1": {"marker": "D", "linestyle": "-.", "color": "#d62728"},
}

F1_CLASS_STYLES = {
    0: {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    1: {"color": "#d62728", "marker": "D", "linestyle": "-."},
    2: {"color": "#ff7f0e", "marker": "^", "linestyle": "--"},
}


def build_series_specs(df: pd.DataFrame) -> list[dict[str, str]]:
    series_specs: list[dict[str, str]] = []

    class_col_pattern = re.compile(r"^class(\d+)_(precision|recall|f1)$")
    class_ids = sorted(
        {
            int(match.group(1))
            for col in df.columns
            for match in [class_col_pattern.match(col)]
            if match is not None
        }
    )

    for metric_name in ("precision", "recall", "f1"):
        column = f"class1_{metric_name}"
        if column not in df.columns:
            continue
        style = CLASS1_METRIC_STYLES[metric_name]
        metric_label = {"precision": "Precision", "recall": "Recall", "f1": "F1"}[metric_name]
        series_specs.append(
            {
                "column": column,
                "label": f"Sharp {metric_label}",
                "color": style["color"],
                "marker": style["marker"],
                "linestyle": style["linestyle"],
            }
        )

    for class_id in class_ids:
        column = f"class{class_id}_f1"
        if column not in df.columns:
            continue
        if class_id == 1:
            continue
        class_label = CLASS_LABELS.get(class_id, f"Class {class_id}")
        style = F1_CLASS_STYLES.get(
            class_id,
            {"color": CLASS_COLORS.get(class_id, "#7f7f7f"), "marker": "o", "linestyle": "-"},
        )
        series_specs.append(
            {
                "column": column,
                "label": f"{class_label} F1",
                "color": style["color"],
                "marker": style["marker"],
                "linestyle": style["linestyle"],
            }
        )

    if "macro_f1" in df.columns:
        series_specs.append(
            {
                "column": "macro_f1",
                "label": "Macro F1",
                "color": "#9467bd",
                "marker": "^",
                "linestyle": ":",
            }
        )

    return series_specs


def plot_eval_curve(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path).sort_values("sample_percentage")
    x_vals = df["sample_percentage"].astype(float).to_numpy()
    series_specs = build_series_specs(df)

    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    plotted_y_vals: list[np.ndarray] = []

    for spec in series_specs:
        y_vals = pd.to_numeric(df[spec["column"]], errors="coerce").to_numpy(dtype=float)
        plotted_y_vals.append(y_vals[np.isfinite(y_vals)])
        ax.plot(
            x_vals,
            y_vals,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=2.6,
            markersize=9.6,
            markeredgewidth=0.5,
            label=spec["label"],
        )

    all_y_vals = np.concatenate([vals for vals in plotted_y_vals if vals.size > 0])
    y_data_min = float(np.min(all_y_vals))
    y_data_max = float(np.max(all_y_vals))
    y_margin = max(0.015, (y_data_max - y_data_min) * 0.18)
    y_min = max(0.0, y_data_min - y_margin)
    y_max = y_data_max + y_margin
    if y_data_max >= 0.985:
        y_max = max(y_max, 1.02)
    y_max = min(1.05, y_max)
    if y_max - y_min < 0.12:
        y_center = (y_min + y_max) / 2
        half_span = 0.06
        y_min = max(0.0, y_center - half_span)
        y_max = min(1.05, y_center + half_span)

    ax.set_xlim(18, 113)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_vals)
    y_tick_top = min(1.0, y_max)
    ax.set_yticks(np.round(np.linspace(y_min, y_tick_top, 5), 2))
    ax.set_xlabel("Sample (%)", fontsize=24, fontweight="semibold")
    ax.set_ylabel("Score", fontsize=24, fontweight="semibold")
    ax.tick_params(axis="both", labelsize=20, width=1.2, length=5.5)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("semibold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(axis="y", linestyle="--", linewidth=1.15, alpha=0.48)
    ax.grid(axis="x", alpha=0.12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        prop={"weight": "semibold", "family": "DejaVu Sans", "size": 22},
        columnspacing=1.6,
        handlelength=2.3,
        handletextpad=0.7,
        labelspacing=0.75,
        borderaxespad=0.5,
    )

    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.81))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "Bachelor_Thesis" / "imagefocus" / "images"
    for spec in PLOT_SPECS:
        csv_path = root_dir / spec["csv_path"]
        output_path = output_dir / spec["output_name"]
        plot_eval_curve(csv_path, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
