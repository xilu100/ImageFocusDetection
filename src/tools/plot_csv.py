import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tools.util import get_root_dir
except ModuleNotFoundError:
    from util import get_root_dir

DEFAULT_PLOT_CONFIG = {
    "modules": {
        "time": 1,
        "evaluate": 1,
    },
    "time_metrics": {
        # Data loading / preprocessing time
        "img_load_time": 1,
        "pca_time": 1,
        # CNN cache build time
        "lmdb_time": 1,
        # Model training time
        "train_time": 1,
    },
    "evaluate_metrics": {
        # class 0 row
        "class0_precision": 0,
        "class0_recall": 0,
        "class0_f1": 0,
        "class0_support": 0,
        # class 1 row
        "class1_precision": 1,
        "class1_recall": 1,
        "class1_f1": 1,
        "class1_support": 0,
        # accuracy row
        "accuracy": 0,
        "accuracy_support": 0,
        # macro avg row
        "macro_p": 0,
        "macro_r": 0,
        "macro_f1": 1,
        "macro_support": 0,
        # weighted avg row
        "weighted_precision": 0,
        "weighted_recall": 0,
        "weighted_f1": 0,
        "weighted_support": 0,
        # Confusion matrix entries
        "cm_tn": 0,
        "cm_fp": 0,
        "cm_fn": 0,
        "cm_tp": 0,
    },
}


def resolve_plot_config(plot_config: dict[str, dict[str, int]] | None = None) -> dict[str, dict[str, int]]:
    config = {
        "modules": dict(DEFAULT_PLOT_CONFIG["modules"]),
        "time_metrics": dict(DEFAULT_PLOT_CONFIG["time_metrics"]),
        "evaluate_metrics": dict(DEFAULT_PLOT_CONFIG["evaluate_metrics"]),
    }
    if not plot_config:
        return config

    modules = plot_config.get("modules")
    if isinstance(modules, dict):
        config["modules"].update({str(k): int(v) for k, v in modules.items()})
    time_metrics = plot_config.get("time_metrics")
    if isinstance(time_metrics, dict):
        config["time_metrics"].update({str(k): int(v) for k, v in time_metrics.items()})
    evaluate_metrics = plot_config.get("evaluate_metrics")
    if isinstance(evaluate_metrics, dict):
        config["evaluate_metrics"].update({str(k): int(v) for k, v in evaluate_metrics.items()})
    return config


def sanitize_filename_part(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")


def find_latest_sweep_dir(logs_root: Path) -> Path:
    candidates = [p for p in logs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No sweep directories found in: {logs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def infer_control_var(csv_path: Path, df: pd.DataFrame) -> str:
    stem = csv_path.stem
    model_prefix = sanitize_filename_part(str(df["model_name"].iloc[0])) if "model_name" in df.columns and not df.empty else ""
    if model_prefix and stem.startswith(model_prefix + "_"):
        return stem[len(model_prefix) + 1:]
    parts = stem.split("_")
    if len(parts) > 1:
        return "_".join(parts[1:])
    return "control"


def to_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def enabled_columns(metric_switches: dict[str, int]) -> list[str]:
    return [col for col, use in metric_switches.items() if use == 1]


def sorted_by_control(df: pd.DataFrame, control_col: str) -> pd.DataFrame:
    ctrl = pd.to_numeric(df[control_col], errors="coerce")
    if ctrl.notna().all():
        return df.assign(ctrl_num=ctrl).sort_values("ctrl_num").drop(columns=["ctrl_num"])
    return df.assign(ctrl_str=df[control_col].astype(str)).sort_values("ctrl_str").drop(columns=["ctrl_str"])


def split_columns_by_value_range(
        metric_df: pd.DataFrame,
        lower: float = 0.0,
        upper: float = 1.0,
        tol: float = 1e-9,
) -> tuple[list[str], list[str]]:
    in_range_cols: list[str] = []
    out_range_cols: list[str] = []
    for col in metric_df.columns:
        series = metric_df[col].dropna()
        if series.empty:
            continue
        col_min = float(series.min())
        col_max = float(series.max())
        if (lower - tol) <= col_min and col_max <= (upper + tol):
            in_range_cols.append(col)
        else:
            out_range_cols.append(col)
    return in_range_cols, out_range_cols


def plot_grouped_bars(
        plot_df: pd.DataFrame,
        x_labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        out_file: Path,
        y_lim: tuple[float, float] | None = None,
) -> Path | None:
    if plot_df.empty:
        return None

    x_pos = np.arange(len(x_labels), dtype=float)
    metric_cols = list(plot_df.columns)
    n_metrics = len(metric_cols)

    group_width = 0.82
    bar_width = max(0.08, min(0.28, group_width / max(n_metrics, 1)))
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * bar_width

    plt.figure(figsize=(9, 5))
    for i, col in enumerate(metric_cols):
        plt.bar(
            x_pos + offsets[i],
            plot_df[col].values,
            width=bar_width * 0.95,
            label=col,
            alpha=0.9,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_pos, x_labels)
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    return out_file


def plot_time(
        df: pd.DataFrame,
        model_name: str,
        control_col: str,
        out_file: Path,
        plot_modules: dict[str, int],
        time_metrics: dict[str, int],
) -> None:
    if plot_modules.get("time", 0) != 1:
        return

    time_cols = [c for c in enabled_columns(time_metrics) if c in df.columns]
    if not time_cols:
        return
    time_df = to_numeric_frame(df, time_cols).dropna(axis=1, how="all")
    if time_df.empty:
        return

    x_labels = df[control_col].astype(str).tolist()
    x_pos = np.arange(len(x_labels), dtype=float)
    metric_cols = list(time_df.columns)
    n_metrics = len(metric_cols)

    # Grouped bar width adapts to metric count.
    group_width = 0.82
    bar_width = max(0.08, min(0.28, group_width / max(n_metrics, 1)))
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * bar_width

    plt.figure(figsize=(9, 5))
    for i, col in enumerate(metric_cols):
        plt.bar(
            x_pos + offsets[i],
            time_df[col].values,
            width=bar_width * 0.95,
            label=col,
            alpha=0.9,
        )

    plt.title(f"{model_name} - {control_col} - Time")
    plt.xlabel(control_col)
    plt.ylabel("Seconds")
    plt.xticks(x_pos, x_labels)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_evaluate(
        df: pd.DataFrame,
        model_name: str,
        control_col: str,
        out_file: Path,
        plot_modules: dict[str, int],
        evaluate_metrics: dict[str, int],
) -> list[Path]:
    if plot_modules.get("evaluate", 0) != 1:
        return []

    eval_cols = [c for c in enabled_columns(evaluate_metrics) if c in df.columns]
    eval_df = to_numeric_frame(df, eval_cols).dropna(axis=1, how="all")
    if eval_df.empty:
        return []

    x_labels = df[control_col].astype(str).tolist()
    in_range_cols, out_range_cols = split_columns_by_value_range(eval_df, lower=0.0, upper=1.0)

    outputs: list[Path] = []
    both_groups = bool(in_range_cols) and bool(out_range_cols)

    if in_range_cols:
        score_out = out_file.with_name(f"{out_file.stem}_Range0to1{out_file.suffix}") if both_groups else out_file
        generated = plot_grouped_bars(
            eval_df[in_range_cols],
            x_labels=x_labels,
            title=f"{model_name} - {control_col} - Evaluate [0,1]",
            xlabel=control_col,
            ylabel="Score",
            out_file=score_out,
            y_lim=(0.0, 1.0),
        )
        if generated:
            outputs.append(generated)

    if out_range_cols:
        other_out = out_file.with_name(f"{out_file.stem}_OtherRange{out_file.suffix}") if both_groups else out_file
        generated = plot_grouped_bars(
            eval_df[out_range_cols],
            x_labels=x_labels,
            title=f"{model_name} - {control_col} - Evaluate (Other Range)",
            xlabel=control_col,
            ylabel="Value",
            out_file=other_out,
            y_lim=None,
        )
        if generated:
            outputs.append(generated)

    return outputs


def plot_model_csv(
        csv_path: Path,
        output_dir: Path,
        plot_config: dict[str, dict[str, int]] | None,
) -> list[Path]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    if "model_name" not in df.columns:
        return []

    config = resolve_plot_config(plot_config)
    plot_modules = config["modules"]
    time_metrics = config["time_metrics"]
    evaluate_metrics = config["evaluate_metrics"]

    model_name = str(df["model_name"].iloc[0])
    control_col = infer_control_var(csv_path, df)
    if control_col not in df.columns:
        excluded_metric_cols = set(enabled_columns(time_metrics) + enabled_columns(evaluate_metrics))
        numeric_candidates = [
            c
            for c in df.columns
            if c not in {"model_name"}
            and c not in excluded_metric_cols
            and c not in {"accuracy", "accuracy_support", "weighted_f1", "weighted_precision", "weighted_recall"}
        ]
        if numeric_candidates:
            control_col = numeric_candidates[0]
        else:
            return []

    df = sorted_by_control(df, control_col)
    base_name = f"{sanitize_filename_part(model_name)}_{sanitize_filename_part(control_col)}"

    output_dir.mkdir(parents=True, exist_ok=True)
    time_out = output_dir / f"{base_name}_Time.png"
    eval_out = output_dir / f"{base_name}_Evaluate.png"

    plot_time(df, model_name, control_col, time_out, plot_modules, time_metrics)
    eval_outputs = plot_evaluate(df, model_name, control_col, eval_out, plot_modules, evaluate_metrics)

    outputs = []
    if time_out.exists():
        outputs.append(time_out)
    outputs.extend([p for p in eval_outputs if p.exists()])
    return outputs


def plot_directory(
        sweep_dir: Path,
        output_dir: Path | None = None,
        plot_config: dict[str, dict[str, int]] | None = None,
) -> list[Path]:
    csv_files = sorted(
        p
        for p in sweep_dir.glob("*.csv")
        if not p.name.startswith(".")
    )
    all_outputs: list[Path] = []
    if output_dir is None:
        output_dir = sweep_dir / "plots"
    for csv_path in csv_files:
        all_outputs.extend(plot_model_csv(csv_path, output_dir, plot_config))
    return all_outputs


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Time/Evaluate charts for each model CSV in a sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Sweep directory under logs/. If omitted, use latest directory in logs/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots. Default: <sweep-dir>/plots",
    )
    return parser


def run_plot_cli(
        sweep_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        plot_config: dict[str, dict[str, int]] | None = None,
) -> list[Path]:
    if sweep_dir is None and output_dir is None:
        args = build_cli_parser().parse_args()
        sweep_dir_arg = args.sweep_dir
        output_dir_arg = args.output_dir
    else:
        sweep_dir_arg = sweep_dir
        output_dir_arg = output_dir

    logs_root = get_root_dir() / "logs"
    resolved_sweep_dir = Path(sweep_dir_arg).resolve() if sweep_dir_arg else find_latest_sweep_dir(logs_root)
    resolved_output_dir = Path(output_dir_arg).resolve() if output_dir_arg else None

    outputs = plot_directory(resolved_sweep_dir, output_dir=resolved_output_dir, plot_config=plot_config)
    print(f"Generated {len(outputs)} plot(s)")
    for path in outputs:
        print(path)
    return outputs


if __name__ == "__main__":
    run_plot_cli()
