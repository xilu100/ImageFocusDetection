import argparse
import ast
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ALLOWED_MODES = {
    "=== Mode: Process Once + Train + Evaluate Sweep ===",
    "=== Mode: Train + Evaluate Only ===",
}

MODE_LINE_RE = re.compile(r"(=== Mode: .+ ===)")
RUN_LINE_RE = re.compile(
    r"(?:^|\s)=== Run (\d+)/(\d+) \| ([A-Za-z0-9_.]+)=(.+?) ==="
)
RUN_LINE_SIMPLE_RE = re.compile(r"(?:^|\s)=== Run (\d+)/(\d+) ===")
MODEL_BLOCK_RE = re.compile(r"---- (.+?) ----")
FINAL_MODEL_PARAMS_RE = re.compile(r"final_model_params = (.+)$")
TIMESTAMP_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
EXPERIMENT_CFG_RE = re.compile(r"experiment_cfg = (.+)$")
MODEL_PREFIX_RE = re.compile(r"^\[(.+?)\]\s+(.+)$")
CPU_CORES_RE = re.compile(r"Using (\d+) CPU cores for image loading\.")
IMAGE_LOADING_DONE_RE = re.compile(r"Image loading done, time: ([0-9.]+)s")
X_SHAPE_BEFORE_RE = re.compile(r"X shape before PCA: (.+)$")
PCA_DONE_RE = re.compile(r"PCA preprocessing done, time: ([0-9.]+)s")
X_SHAPE_AFTER_RE = re.compile(r"X shape after PCA: (.+)$")
TRAINING_DONE_RE = re.compile(r"Training done, time: ([0-9.]+)s")
CNN_TRAINING_TIME_RE = re.compile(r"Training time: ([0-9.]+) seconds")
LMDB_BUILD_DONE_RE = re.compile(r"^\[LMDB\] Build done in ([0-9.]+)s$")
EPOCH_LOSS_RE = re.compile(r"^Epoch (\d+)/(\d+), Loss: ([0-9.]+)$")
DEVICE_AMP_RE = re.compile(r"^Using device: ([^,]+), AMP enabled: (True|False)$")
BATCH_WORKER_RE = re.compile(r"^Batch size: (\d+), num_workers: (\d+)$")
EVAL_MODEL_RE = re.compile(r"^={7,}\s*(.+?)\s*={7,}$")
ACCURACY_RE = re.compile(r"^Accuracy:\s*([0-9.]+)$")
CLASS_REPORT_ROW_RE = re.compile(
    r"^\s*(0|1|macro avg|weighted avg)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s*$"
)
ACCURACY_ROW_RE = re.compile(r"^\s*accuracy\s+([0-9.]+)\s+([0-9]+)\s*$")
MODEL_KEY_TO_DISPLAY = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "cnn": "CNN",
}
MODEL_STATS_COLUMNS = [
    "device",
    "amp",
    "batch",
    "worker",
    "lmdb_s",
    "cpu_core",
    "img_load_s",
    "x_before",
    "pca_s",
    "x_after",
    "train_s",
]
EVAL_COLUMNS = [
    "accuracy",
    "accuracy_support",
    "class0_precision",
    "class0_recall",
    "class0_f1",
    "class0_support",
    "class1_precision",
    "class1_recall",
    "class1_f1",
    "class1_support",
    "macro_p",
    "macro_r",
    "macro_f1",
    "macro_support",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "weighted_support",
    "cm_tn",
    "cm_fp",
    "cm_fn",
    "cm_tp",
]


@dataclass
class RunRecord:
    run_idx: int
    run_total: int
    control_path: str
    control_name: str
    control_value: str
    model_params: dict[str, object] = field(default_factory=dict)
    model_stats: dict[str, dict[str, object]] = field(default_factory=dict)
    model_eval: dict[str, dict[str, object]] = field(default_factory=dict)


def _extract_payload(line: str) -> str:
    marker = " [INFO] "
    idx = line.find(marker)
    if idx >= 0:
        return line[idx + len(marker):].strip()
    return line.strip()


def _safe_parse_dict(raw: str) -> dict[str, object]:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _sanitize_filename_part(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _control_prefix(control_path: str) -> str:
    if control_path.startswith("training."):
        return "TR"
    if control_path.startswith("models.decision_tree."):
        return "DT"
    if control_path.startswith("models.random_forest."):
        return "RF"
    if control_path.startswith("models.svm."):
        return "SVM"
    if control_path.startswith("models.cnn."):
        return "CNN"
    return "CTRL"


def _flatten_plain(data: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        flat[str(key)] = value
    return flat


def _pick_mode(lines: list[str]) -> str:
    for line in lines:
        payload = _extract_payload(line)
        match = MODE_LINE_RE.search(payload)
        if not match:
            continue
        mode_text = match.group(1).strip()
        if mode_text in ALLOWED_MODES:
            return mode_text
    raise ValueError(
        "Unsupported mode in log. Only these modes are parsed: "
        f"{sorted(ALLOWED_MODES)}"
    )


def _extract_experiment_cfg(lines: list[str]) -> dict[str, object]:
    for line in lines:
        payload = _extract_payload(line)
        match = EXPERIMENT_CFG_RE.search(payload)
        if not match:
            continue
        parsed = _safe_parse_dict(match.group(1).strip())
        if parsed:
            return parsed
    return {}


def _extract_runs(lines: list[str]) -> tuple[list[RunRecord], datetime | None]:
    runs: list[RunRecord] = []
    first_run_time: datetime | None = None
    current_run: RunRecord | None = None
    current_model_name: str | None = None
    current_eval_model: str | None = None
    parsing_class_report = False
    parsing_confusion_matrix = False
    confusion_rows: list[list[int]] = []

    for line in lines:
        payload = _extract_payload(line)

        run_match = RUN_LINE_RE.search(payload)
        if run_match:
            run_idx = int(run_match.group(1))
            run_total = int(run_match.group(2))
            control_path = run_match.group(3).strip()
            control_name = control_path.split(".")[-1]
            control_value = run_match.group(4).strip()
            current_run = RunRecord(
                run_idx=run_idx,
                run_total=run_total,
                control_path=control_path,
                control_name=control_name,
                control_value=control_value,
            )
            runs.append(current_run)
            current_model_name = None
            current_eval_model = None
            parsing_class_report = False
            parsing_confusion_matrix = False
            confusion_rows = []

            if first_run_time is None:
                ts_match = TIMESTAMP_PREFIX_RE.match(line.strip())
                if ts_match:
                    first_run_time = datetime.strptime(
                        ts_match.group(1), "%Y-%m-%d %H:%M:%S"
                    )
            continue

        simple_run_match = RUN_LINE_SIMPLE_RE.search(payload)
        if simple_run_match:
            run_idx = int(simple_run_match.group(1))
            run_total = int(simple_run_match.group(2))
            current_run = RunRecord(
                run_idx=run_idx,
                run_total=run_total,
                control_path="__normal__",
                control_name="normal",
                control_value="normal",
            )
            runs.append(current_run)
            current_model_name = None
            current_eval_model = None
            parsing_class_report = False
            parsing_confusion_matrix = False
            confusion_rows = []

            if first_run_time is None:
                ts_match = TIMESTAMP_PREFIX_RE.match(line.strip())
                if ts_match:
                    first_run_time = datetime.strptime(
                        ts_match.group(1), "%Y-%m-%d %H:%M:%S"
                    )
            continue

        if current_run is None:
            continue

        model_match = MODEL_BLOCK_RE.search(payload)
        if model_match:
            current_model_name = model_match.group(1).strip()
            continue

        params_match = FINAL_MODEL_PARAMS_RE.search(payload)
        if params_match and current_model_name is not None:
            current_run.model_params[current_model_name] = _safe_parse_dict(
                params_match.group(1).strip()
            )
            continue

        eval_model_match = EVAL_MODEL_RE.search(payload)
        if eval_model_match:
            current_eval_model = eval_model_match.group(1).strip()
            current_run.model_eval.setdefault(current_eval_model, {})
            parsing_class_report = False
            parsing_confusion_matrix = False
            confusion_rows = []
            continue

        if current_eval_model is not None:
            eval_metrics = current_run.model_eval.setdefault(current_eval_model, {})

            accuracy_match = ACCURACY_RE.search(payload)
            if accuracy_match:
                eval_metrics["accuracy"] = float(accuracy_match.group(1))
                continue

            if payload.strip() == "Classification Report:":
                parsing_class_report = True
                parsing_confusion_matrix = False
                confusion_rows = []
                continue

            if payload.strip() == "Confusion Matrix:":
                parsing_confusion_matrix = True
                parsing_class_report = False
                confusion_rows = []
                continue

            if parsing_class_report:
                class_row_match = CLASS_REPORT_ROW_RE.search(payload)
                if class_row_match:
                    label = class_row_match.group(1)
                    p = float(class_row_match.group(2))
                    r = float(class_row_match.group(3))
                    f1 = float(class_row_match.group(4))
                    s = int(class_row_match.group(5))
                    if label == "0":
                        eval_metrics["class0_precision"] = p
                        eval_metrics["class0_recall"] = r
                        eval_metrics["class0_f1"] = f1
                        eval_metrics["class0_support"] = s
                    elif label == "1":
                        eval_metrics["class1_precision"] = p
                        eval_metrics["class1_recall"] = r
                        eval_metrics["class1_f1"] = f1
                        eval_metrics["class1_support"] = s
                    elif label == "macro avg":
                        eval_metrics["macro_p"] = p
                        eval_metrics["macro_r"] = r
                        eval_metrics["macro_f1"] = f1
                        eval_metrics["macro_support"] = s
                    elif label == "weighted avg":
                        eval_metrics["weighted_precision"] = p
                        eval_metrics["weighted_recall"] = r
                        eval_metrics["weighted_f1"] = f1
                        eval_metrics["weighted_support"] = s
                    continue

                accuracy_row_match = ACCURACY_ROW_RE.search(payload)
                if accuracy_row_match:
                    eval_metrics["accuracy"] = float(accuracy_row_match.group(1))
                    eval_metrics["accuracy_support"] = int(accuracy_row_match.group(2))
                    continue

            if parsing_confusion_matrix:
                numbers = [int(x) for x in re.findall(r"\d+", payload)]
                if len(numbers) >= 2:
                    confusion_rows.append(numbers[:2])
                    if len(confusion_rows) == 2:
                        eval_metrics["cm_tn"] = confusion_rows[0][0]
                        eval_metrics["cm_fp"] = confusion_rows[0][1]
                        eval_metrics["cm_fn"] = confusion_rows[1][0]
                        eval_metrics["cm_tp"] = confusion_rows[1][1]
                        parsing_confusion_matrix = False
                        confusion_rows = []
                    continue

        lmdb_match = LMDB_BUILD_DONE_RE.search(payload)
        if lmdb_match:
            cnn_stats = current_run.model_stats.setdefault("CNN", {})
            cnn_stats["lmdb_s"] = float(lmdb_match.group(1))
            continue

        epoch_match = EPOCH_LOSS_RE.search(payload)
        if epoch_match:
            epoch_idx = int(epoch_match.group(1))
            epoch_total = int(epoch_match.group(2))
            loss_value = float(epoch_match.group(3))
            cnn_stats = current_run.model_stats.setdefault("CNN", {})
            seq = cnn_stats.setdefault("_epoch_seq", [])
            if isinstance(seq, list):
                seq.append((epoch_idx, loss_value))
            cnn_stats["epoch_total"] = epoch_total
            continue

        device_amp_match = DEVICE_AMP_RE.search(payload)
        if device_amp_match:
            cnn_stats = current_run.model_stats.setdefault("CNN", {})
            cnn_stats["device"] = device_amp_match.group(1).strip()
            cnn_stats["amp"] = device_amp_match.group(2).strip()
            continue

        batch_worker_match = BATCH_WORKER_RE.search(payload)
        if batch_worker_match:
            cnn_stats = current_run.model_stats.setdefault("CNN", {})
            cnn_stats["batch"] = int(batch_worker_match.group(1))
            cnn_stats["worker"] = int(batch_worker_match.group(2))
            continue

        model_prefix_match = MODEL_PREFIX_RE.search(payload)
        if not model_prefix_match:
            continue

        model_name = model_prefix_match.group(1).strip()
        message = model_prefix_match.group(2).strip()
        stats = current_run.model_stats.setdefault(model_name, {})

        cpu_match = CPU_CORES_RE.search(message)
        if cpu_match:
            stats["cpu_core"] = int(cpu_match.group(1))
            continue

        load_done_match = IMAGE_LOADING_DONE_RE.search(message)
        if load_done_match:
            stats["img_load_s"] = float(load_done_match.group(1))
            continue

        x_before_match = X_SHAPE_BEFORE_RE.search(message)
        if x_before_match:
            stats["x_before"] = x_before_match.group(1).strip()
            continue

        pca_done_match = PCA_DONE_RE.search(message)
        if pca_done_match:
            stats["pca_s"] = float(pca_done_match.group(1))
            continue

        x_after_match = X_SHAPE_AFTER_RE.search(message)
        if x_after_match:
            stats["x_after"] = x_after_match.group(1).strip()
            continue

        training_done_match = TRAINING_DONE_RE.search(message)
        if training_done_match:
            stats["train_s"] = float(training_done_match.group(1))
            continue

        cnn_training_time_match = CNN_TRAINING_TIME_RE.search(message)
        if cnn_training_time_match:
            stats["train_s"] = float(cnn_training_time_match.group(1))

    if not runs:
        raise ValueError("No run section found in log.")

    return runs, first_run_time


def log_to_model_csv(log_path: str | Path, output_root: str | Path | None = None) -> Path:
    """
    Parse a run log and export per-model CSV files.

    CSV row includes:
    - model_name
    - flattened training params columns
    - flattened model params columns
    """
    src_path = Path(log_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Log file not found: {src_path}")

    lines = src_path.read_text(encoding="utf-8").splitlines()
    _pick_mode(lines)

    experiment_cfg = _extract_experiment_cfg(lines)
    training_params = {}
    if isinstance(experiment_cfg.get("training"), dict):
        training_params = experiment_cfg["training"]
    models_cfg = {}
    if isinstance(experiment_cfg.get("models"), dict):
        models_cfg = experiment_cfg["models"]

    fallback_model_params_by_name: dict[str, dict[str, object]] = {}
    for model_key, params in models_cfg.items():
        if not isinstance(params, dict):
            continue
        display_name = MODEL_KEY_TO_DISPLAY.get(model_key, model_key)
        fallback_model_params_by_name[display_name] = params

    runs, first_run_time = _extract_runs(lines)
    control_name = runs[0].control_name
    control_path = runs[0].control_path
    for run in runs[1:]:
        if run.control_path != control_path:
            raise ValueError(
                f"Multiple sweep params found: {control_path} and {run.control_path}"
            )

    stamp = (first_run_time or datetime.now()).strftime("%Y_%m%d_%H%M%S")
    if control_path == "__normal__":
        folder_name = f"NM_{stamp}"
    else:
        prefix = _control_prefix(control_path)
        folder_name = f"{prefix}_{_sanitize_filename_part(control_name)}_{stamp}"
    if output_root:
        base_dir = Path(output_root).resolve()
    else:
        project_root = _project_root()
        base_dir = project_root / "logs"
        base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = base_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names: set[str] = set()
    for run in runs:
        model_names.update(run.model_params.keys())
    model_names.update(fallback_model_params_by_name.keys())

    for model_name in sorted(model_names):
        file_name = (
            f"{_sanitize_filename_part(model_name)}_"
            f"{_sanitize_filename_part(control_name)}.csv"
        )
        csv_path = out_dir / file_name
        rows: list[dict[str, object]] = []
        training_columns_order: list[str] = []
        model_columns_order: list[str] = []
        stats_columns_present: list[str] = []
        eval_columns_present: list[str] = []
        dynamic_stats_columns_present: list[str] = []

        for run in runs:
            model_params = run.model_params.get(model_name)
            if model_params is None:
                model_params = fallback_model_params_by_name.get(model_name)
            if model_params is None:
                continue

            # Per-run training params: replace sweep target with current run value.
            run_training = dict(training_params)
            if run.control_path != "__normal__":
                run_training[control_name] = run.control_value

            flat_training = _flatten_plain(run_training)
            flat_model = _flatten_plain(model_params)
            if model_name == "CNN":
                # Keep LMDB timing metric, drop config flag about whether to build LMDB.
                flat_model.pop("build_lmdb_if_missing", None)
                flat_model.pop("assume_fixed_size", None)

            if not training_columns_order:
                training_columns_order = list(flat_training.keys())
            if not model_columns_order:
                model_columns_order = [k for k in flat_model.keys() if k not in {"model_name"}]

            row = {"model_name": model_name}
            row.update(flat_training)
            row.update(flat_model)
            model_stats = dict(run.model_stats.get(model_name, {}))
            if model_name == "CNN":
                epoch_seq = model_stats.pop("_epoch_seq", None)
                if isinstance(epoch_seq, list) and epoch_seq:
                    epoch_seq_sorted = sorted(epoch_seq, key=lambda x: x[0])
                    for idx, loss in epoch_seq_sorted:
                        model_stats[f"loss_e{idx}"] = float(f"{loss:.4f}")
                model_stats.pop("epoch_total", None)
            row.update(model_stats)
            row.update(run.model_eval.get(model_name, {}))
            rows.append(row)
            for stat_col in MODEL_STATS_COLUMNS:
                value = row.get(stat_col)
                if value not in (None, "", []) and stat_col not in stats_columns_present:
                    stats_columns_present.append(stat_col)
            for eval_col in EVAL_COLUMNS:
                value = row.get(eval_col)
                if value not in (None, "", []) and eval_col not in eval_columns_present:
                    eval_columns_present.append(eval_col)
            for stat_col in model_stats.keys():
                if stat_col in MODEL_STATS_COLUMNS:
                    continue
                value = row.get(stat_col)
                if value not in (None, "", []) and stat_col not in dynamic_stats_columns_present:
                    dynamic_stats_columns_present.append(stat_col)

        # Keep epoch loss columns ordered by epoch index (loss_e1, loss_e2, ...).
        epoch_loss_columns = sorted(
            (c for c in dynamic_stats_columns_present if c.startswith("loss_e")),
            key=lambda c: int(c.split("loss_e", 1)[1]),
        )
        other_dynamic_stats_columns = [
            c for c in dynamic_stats_columns_present if not c.startswith("loss_e")
        ]

        stats_columns_without_train = [c for c in stats_columns_present if c != "train_s"]
        train_s_columns = [c for c in stats_columns_present if c == "train_s"]

        ordered_columns: list[str] = []
        for col in (
                ["model_name"]
                + training_columns_order
                + model_columns_order
                + stats_columns_without_train
                + epoch_loss_columns
                + other_dynamic_stats_columns
                + train_s_columns
                + eval_columns_present
        ):
            if col not in ordered_columns:
                ordered_columns.append(col)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=ordered_columns,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return out_dir


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert training log to per-model CSV files.")
    parser.add_argument(
        "log_path",
        nargs="?",
        default=None,
        help="Optional log path. If omitted, uses the latest .log file in project logs/.",
    )
    return parser


def _find_latest_log_file() -> Path:
    project_root = _project_root()
    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    candidates = [
        p
        for p in logs_dir.rglob("*.log")
        if p.is_file() and not p.name.endswith("_complete.log")
    ]
    if not candidates:
        raise FileNotFoundError(f"No .log files found in: {logs_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    selected_log = args.log_path if args.log_path else _find_latest_log_file()
    output_dir = log_to_model_csv(selected_log)
    print(output_dir)
