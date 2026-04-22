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
        # class 2 row
        "class2_precision": 1,
        "class2_recall": 1,
        "class2_f1": 1,
        "class2_support": 0,
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
        "cm_size": 0,
        "cm_r0_c0": 0,
        "cm_r0_c1": 0,
        "cm_r0_c2": 0,
        "cm_r1_c0": 0,
        "cm_r1_c1": 0,
        "cm_r1_c2": 0,
        "cm_r2_c0": 0,
        "cm_r2_c1": 0,
        "cm_r2_c2": 0,
    },
}


def resolve_plot_config(plot_config: dict[str, dict[str, int]] | None = None) -> dict[str, dict[str, int]]:
    config = {
        "time_metrics": dict(DEFAULT_PLOT_CONFIG["time_metrics"]),
        "evaluate_metrics": dict(DEFAULT_PLOT_CONFIG["evaluate_metrics"]),
    }
    if not plot_config:
        return config

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
    model_prefix = sanitize_filename_part(
        str(df["model_name"].iloc[0])) if "model_name" in df.columns and not df.empty else ""
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


def extract_loss_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^loss_e(\d+)$")
    pairs: list[tuple[int, str]] = []
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        pairs.append((int(match.group(1)), col))
    pairs.sort(key=lambda x: x[0])
    return [col for _, col in pairs]


def plot_grouped_bars(
        plot_df: pd.DataFrame,
        x_labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        out_file: Path,
        y_lim: tuple[float, float] | None = None,
        y_ref_lines: list[float] | None = None,
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
    if y_ref_lines:
        for y_ref in y_ref_lines:
            plt.axhline(
                y=y_ref,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.45,
            )
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    return out_file


def plot_loss_curves(
        df: pd.DataFrame,
        model_name: str,
        control_col: str,
        output_dir: Path,
        base_name: str,
) -> list[Path]:
    loss_cols = extract_loss_columns(df)
    if not loss_cols:
        return []

    loss_df = to_numeric_frame(df, loss_cols).dropna(axis=1, how="all")
    if loss_df.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    epochs = [int(col.split("loss_e", 1)[1]) for col in loss_df.columns]

    out_all = output_dir / f"{base_name}_Loss_AllRuns.png"
    plt.figure(figsize=(9, 5))
    for _, row in loss_df.iterrows():
        run_label = str(df.loc[row.name, control_col])
        y_values = row.values.astype(float)
        valid_mask = ~np.isnan(y_values)
        if not valid_mask.any():
            continue
        x_vals = np.array(epochs)[valid_mask]
        plt.plot(x_vals, y_values[valid_mask], marker="o", linewidth=1.8, label=run_label)
    if plt.gca().has_data():
        plt.title(f"{model_name} - {control_col} - Loss (All Runs)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(axis="y", alpha=0.25)
        plt.legend(title=control_col)
        plt.tight_layout()
        plt.savefig(out_all, dpi=180)
        outputs.append(out_all)
    plt.close()

    for _, row in loss_df.iterrows():
        run_value = str(df.loc[row.name, control_col])
        out_one = output_dir / f"{base_name}_Loss_{sanitize_filename_part(run_value)}.png"
        y_values = row.values.astype(float)
        valid_mask = ~np.isnan(y_values)
        if not valid_mask.any():
            continue

        plt.figure(figsize=(9, 5))
        x_vals = np.array(epochs)[valid_mask]
        plt.plot(x_vals, y_values[valid_mask], marker="o", linewidth=2.0, color="#1f77b4")
        plt.title(f"{model_name} - {control_col}={run_value} - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_one, dpi=180)
        plt.close()
        outputs.append(out_one)

    return outputs


def plot_time(
        df: pd.DataFrame,
        model_name: str,
        control_col: str,
        out_file: Path,
        time_metrics: dict[str, int],
) -> None:
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
        evaluate_metrics: dict[str, int],
) -> list[Path]:
    evaluate_metrics = dict(evaluate_metrics)
    class2_cols = ["class2_precision", "class2_recall", "class2_f1", "class2_support"]
    has_class2_data = False
    for col in class2_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            has_class2_data = True
            break
    if not has_class2_data:
        for col in class2_cols:
            evaluate_metrics[col] = 0

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
            y_ref_lines=[0.6, 0.7, 0.8],
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

    plot_time(df, model_name, control_col, time_out, time_metrics)
    eval_outputs = plot_evaluate(df, model_name, control_col, eval_out, evaluate_metrics)
    loss_outputs = plot_loss_curves(df, model_name, control_col, output_dir, base_name)

    outputs = []
    if time_out.exists():
        outputs.append(time_out)
    outputs.extend([p for p in eval_outputs if p.exists()])
    outputs.extend([p for p in loss_outputs if p.exists()])
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
