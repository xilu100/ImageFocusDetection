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
        "task": "2C",
        "model_prefix": "DT",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/Random_Forest_sample_percentage.csv",
        "task": "2C",
        "model_prefix": "RF",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/SVM_sample_percentage.csv",
        "task": "2C",
        "model_prefix": "SVM",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_180944/CNN_sample_percentage.csv",
        "task": "2C",
        "model_prefix": "CNN",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/Decision_Tree_sample_percentage.csv",
        "task": "3C",
        "model_prefix": "DT",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/Random_Forest_sample_percentage.csv",
        "task": "3C",
        "model_prefix": "RF",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/SVM_sample_percentage.csv",
        "task": "3C",
        "model_prefix": "SVM",
    },
    {
        "csv_path": "logs/TR_sample_percentage_2026_0427_193743/CNN_sample_percentage.csv",
        "task": "3C",
        "model_prefix": "CNN",
    },
]


CLASS_LABELS = {
    0: "Blur",
    1: "Sharp",
    2: "Intermediate",
}

PLOT_CLASS_LABELS = {
    0: "Blur",
    1: "Sharp",
    2: "Interm.",
}

CLASS_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
    2: "#ff7f0e",
}

METRIC_STYLES = {
    "precision": {"marker": "o", "linestyle": "-", "suffix": "Precision"},
    "recall": {"marker": "s", "linestyle": "--", "suffix": "Recall"},
    "f1": {"marker": "D", "linestyle": "-.", "suffix": "F1"},
}

OTHER_CLASS_LINESTYLES = {
    0: {"precision": "#1f77b4", "recall": "#4c97d9", "f1": "#0f4c81"},
    2: {"precision": "#ff7f0e", "recall": "#f4a340", "f1": "#c55a00"},
}


def get_class_ids(df: pd.DataFrame) -> list[int]:
    class_col_pattern = re.compile(r"^class(\d+)_(precision|recall|f1)$")
    return sorted(
        {
            int(match.group(1))
            for col in df.columns
            for match in [class_col_pattern.match(col)]
            if match is not None
        }
    )


def build_sharp_specs(df: pd.DataFrame) -> list[dict[str, str]]:
    series_specs: list[dict[str, str]] = []

    for metric_name in ("precision", "recall", "f1"):
        column = f"class1_{metric_name}"
        if column not in df.columns:
            continue
        style = METRIC_STYLES[metric_name]
        series_specs.append(
            {
                "column": column,
                "label": f"Sharp {style['suffix']}",
                "color": CLASS_COLORS[1],
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


def build_other_specs(df: pd.DataFrame) -> list[dict[str, str]]:
    series_specs: list[dict[str, str]] = []

    for class_id in get_class_ids(df):
        if class_id == 1:
            continue
        class_label = PLOT_CLASS_LABELS.get(class_id, f"Class {class_id}")
        color_map = OTHER_CLASS_LINESTYLES.get(class_id, {})
        for metric_name in ("precision", "recall", "f1"):
            column = f"class{class_id}_{metric_name}"
            if column not in df.columns:
                continue
            style = METRIC_STYLES[metric_name]
            series_specs.append(
                {
                    "column": column,
                    "label": f"{class_label} {style['suffix']}",
                    "color": color_map.get(metric_name, CLASS_COLORS.get(class_id, "#7f7f7f")),
                    "marker": style["marker"],
                    "linestyle": style["linestyle"],
                }
            )

    return series_specs


def compute_y_limits(plotted_y_vals: list[np.ndarray]) -> tuple[float, float]:
    all_y_vals = np.concatenate([vals for vals in plotted_y_vals if vals.size > 0])
    y_data_min = float(np.min(all_y_vals))
    y_data_max = float(np.max(all_y_vals))
    y_margin = max(0.015, (y_data_max - y_data_min) * 0.18)
    y_min = max(0.0, y_data_min - y_margin)
    y_max = min(1.05, y_data_max + y_margin)

    if y_data_max >= 0.985:
        y_max = max(y_max, 1.02)
    if y_max - y_min < 0.12:
        y_center = (y_min + y_max) / 2
        half_span = 0.06
        y_min = max(0.0, y_center - half_span)
        y_max = min(1.05, y_center + half_span)

    return y_min, y_max


def plot_eval_curve(csv_path: Path, output_path: Path, mode: str) -> None:
    df = pd.read_csv(csv_path).sort_values("sample_percentage")
    x_vals = df["sample_percentage"].astype(float).to_numpy()
    series_specs = build_sharp_specs(df) if mode == "sharp" else build_other_specs(df)
    if not series_specs:
        return

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
            linewidth=2.55,
            markersize=9.0,
            markeredgewidth=0.5,
            label=spec["label"],
        )

    y_min, y_max = compute_y_limits(plotted_y_vals)

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
    ncol = 2 if len(labels) <= 4 else 3
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=ncol,
        frameon=False,
        prop={"weight": "semibold", "family": "DejaVu Sans", "size": 22},
        columnspacing=1.35,
        handlelength=2.2,
        handletextpad=0.65,
        labelspacing=0.7,
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
        for mode in ("sharp", "other"):
            output_name = f"{spec['model_prefix']}_EVA_{spec['task']}_{mode.upper()}.png"
            output_path = output_dir / output_name
            plot_eval_curve(csv_path, output_path, mode=mode)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
