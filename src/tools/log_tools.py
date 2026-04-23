import argparse
import ast
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, cast

try:
    from tools.util import get_root_dir
except ModuleNotFoundError:
    from util import get_root_dir

ALLOWED_MODES = {
    "=== Mode: Train + Evaluate ===",
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
MODEL_PREFIX_RE = re.compile(r"^\[(.+?)]\s+(.+)$")
CPU_CORES_RE = re.compile(r"Using (\d+) CPU cores for image loading\.")
IMAGE_LOADING_DONE_RE = re.compile(r"Image loading done, time: ([0-9.]+)s")
X_SHAPE_BEFORE_RE = re.compile(r"X shape before PCA: (.+)$")
PCA_DONE_RE = re.compile(r"PCA preprocessing done, time: ([0-9.]+)s")
X_SHAPE_AFTER_RE = re.compile(r"X shape after PCA: (.+)$")
TRAINING_DONE_RE = re.compile(r"Training done, time: ([0-9.]+)s")
CNN_TRAINING_TIME_RE = re.compile(r"Training time: ([0-9.]+) seconds")
LMDB_BUILD_DONE_RE = re.compile(r"^\[LMDB] Build done in ([0-9.]+)s$")
EPOCH_LOSS_RE = re.compile(r"^Epoch (\d+)/(\d+), Loss: ([0-9.]+)$")
DEVICE_AMP_RE = re.compile(r"^Using device: ([^,]+), AMP enabled: (True|False)$")
BATCH_WORKER_RE = re.compile(r"^Batch size: (\d+), num_workers: (\d+)$")
EVAL_MODEL_RE = re.compile(r"^={7,}\s*(.+?)\s*={7,}$")
ACCURACY_RE = re.compile(r"^Accuracy:\s*([0-9.]+)$")
CLASS_REPORT_ROW_RE = re.compile(
    r"^\s*(\d+|macro avg|weighted avg)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s*$"
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
    "lmdb_time",
    "cpu_core",
    "img_load_time",
    "x_before",
    "pca_time",
    "x_after",
    "train_time",
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
    "class2_precision",
    "class2_recall",
    "class2_f1",
    "class2_support",
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
    "cm_size",
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


def extract_payload(line: str) -> str:
    marker = " [INFO] "
    idx = line.find(marker)
    if idx >= 0:
        return line[idx + len(marker):].strip()
    return line.strip()


def safe_parse_dict(raw: str) -> dict[str, object]:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(value, dict):
        return value
    return {}


def sanitize_filename_part(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def control_prefix(control_path: str) -> str:
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


def flatten_plain(data: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        flat[str(key)] = value
    return flat


def parse_control_value(raw: str) -> object:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def to_float_or_none(value: object) -> float | None:
    try:
        candidate = cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value)
        return float(candidate)
    except (TypeError, ValueError):
        return None


def is_binary_labeling_mode(training_params: dict[str, object]) -> bool:
    sharp = to_float_or_none(training_params.get("sharp_threshold"))
    blur = to_float_or_none(training_params.get("blur_threshold"))
    if sharp is None or blur is None:
        return False
    return abs(sharp - blur) <= 1e-8


def pick_mode(lines: list[str]) -> str:
    for line in lines:
        payload = extract_payload(line)
        match = MODE_LINE_RE.search(payload)
        if not match:
            continue
        mode_text = match.group(1).strip()
        if mode_text in ALLOWED_MODES:
            return mode_text
    raise ValueError(
        f"Unsupported mode in log. Only these modes are parsed: {sorted(ALLOWED_MODES)}"
    )


def extract_experiment_cfg(lines: list[str]) -> dict[str, object]:
    for line in lines:
        payload = extract_payload(line)
        match = EXPERIMENT_CFG_RE.search(payload)
        if not match:
            continue
        parsed = safe_parse_dict(match.group(1).strip())
        if parsed:
            return parsed
    return {}


def extract_runs(lines: list[str]) -> tuple[list[RunRecord], datetime | None]:
    runs: list[RunRecord] = []
    first_run_time: datetime | None = None
    current_run: RunRecord | None = None
    current_model_name: str | None = None
    current_eval_model: str | None = None
    parsing_class_report = False
    parsing_confusion_matrix = False
    confusion_rows: list[list[int]] = []

    for line in lines:
        payload = extract_payload(line)

        run_match = RUN_LINE_RE.search(payload)
        if run_match:
            run_idx = int(run_match.group(1))
            run_total = int(run_match.group(2))
            control_path = run_match.group(3).strip()
            control_name = control_path.split(".")[-1]
            control_value = run_match.group(4).strip()
            new_run = RunRecord(
                run_idx=run_idx,
                run_total=run_total,
                control_path=control_path,
                control_name=control_name,
                control_value=control_value,
            )
            current_run = new_run
            runs.append(new_run)
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
            new_run = RunRecord(
                run_idx=run_idx,
                run_total=run_total,
                control_path="__normal__",
                control_name="normal",
                control_value="normal",
            )
            current_run = new_run
            runs.append(new_run)
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
            current_run.model_params[current_model_name] = safe_parse_dict(
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
                    if label.isdigit():
                        eval_metrics[f"class{label}_precision"] = p
                        eval_metrics[f"class{label}_recall"] = r
                        eval_metrics[f"class{label}_f1"] = f1
                        eval_metrics[f"class{label}_support"] = s
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
                    confusion_rows.append(numbers)
                    row_len = len(confusion_rows[0])
                    if all(len(row) == row_len for row in confusion_rows) and len(confusion_rows) == row_len:
                        cm_size = row_len
                        eval_metrics["cm_size"] = cm_size
                        for r_idx, row in enumerate(confusion_rows):
                            for c_idx, value in enumerate(row):
                                eval_metrics[f"cm_r{r_idx}_c{c_idx}"] = value

                        # Backward compatibility for binary confusion-matrix keys.
                        if cm_size == 2:
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
            cnn_stats["lmdb_time"] = float(lmdb_match.group(1))
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
            stats["img_load_time"] = float(load_done_match.group(1))
            continue

        x_before_match = X_SHAPE_BEFORE_RE.search(message)
        if x_before_match:
            stats["x_before"] = x_before_match.group(1).strip()
            continue

        pca_done_match = PCA_DONE_RE.search(message)
        if pca_done_match:
            stats["pca_time"] = float(pca_done_match.group(1))
            continue

        x_after_match = X_SHAPE_AFTER_RE.search(message)
        if x_after_match:
            stats["x_after"] = x_after_match.group(1).strip()
            continue

        training_done_match = TRAINING_DONE_RE.search(message)
        if training_done_match:
            stats["train_time"] = float(training_done_match.group(1))
            continue

        cnn_training_time_match = CNN_TRAINING_TIME_RE.search(message)
        if cnn_training_time_match:
            stats["train_time"] = float(cnn_training_time_match.group(1))

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
    pick_mode(lines)

    experiment_cfg = extract_experiment_cfg(lines)
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

    runs, first_run_time = extract_runs(lines)
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
        prefix = control_prefix(control_path)
        folder_name = f"{prefix}_{sanitize_filename_part(control_name)}_{stamp}"
    if output_root:
        base_dir = Path(output_root).resolve()
    else:
        base_dir = get_root_dir() / "logs"
        base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = base_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names: set[str] = set()
    for run in runs:
        model_names.update(run.model_params.keys())
    model_names.update(fallback_model_params_by_name.keys())

    for model_name in sorted(model_names):
        file_name = (
            f"{sanitize_filename_part(model_name)}_"
            f"{sanitize_filename_part(control_name)}.csv"
        )
        csv_path = out_dir / file_name
        rows: list[dict[str, object]] = []
        training_columns_order: list[str] = []
        model_columns_order: list[str] = []
        stats_columns_present: list[str] = []
        eval_columns_present: list[str] = []
        dynamic_stats_columns_present: list[str] = []
        dynamic_eval_columns_present: list[str] = []

        for run in runs:
            model_params = run.model_params.get(model_name)
            if model_params is None:
                model_params = fallback_model_params_by_name.get(model_name)
            if model_params is None:
                continue

            # Per-run training params: replace sweep target with current run value.
            run_training = dict(training_params)
            parsed_control_value = parse_control_value(run.control_value)
            if run.control_path != "__normal__":
                run_training[control_name] = parsed_control_value

            flat_training = flatten_plain(run_training)
            flat_model = flatten_plain(model_params)
            if run.control_path != "__normal__":
                target_model = ""
                if run.control_path.startswith("models."):
                    target_model = run.control_path.split(".")[1]
                target_display = MODEL_KEY_TO_DISPLAY.get(target_model, target_model)

                # Keep sweep axis column stable across all model CSVs.
                if run.control_name in flat_model and model_name != target_display:
                    flat_model[f"model_{run.control_name}"] = flat_model[run.control_name]
                    flat_model.pop(run.control_name, None)

            if run.control_path.startswith("models.") and run.control_name in flat_model:
                target_model = run.control_path.split(".")[1]
                target_display = MODEL_KEY_TO_DISPLAY.get(target_model, target_model)
                if model_name == target_display:
                    flat_model[run.control_name] = parsed_control_value
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
            model_eval = dict(run.model_eval.get(model_name, {}))
            if is_binary_labeling_mode(run_training):
                for col in ("class2_precision", "class2_recall", "class2_f1", "class2_support"):
                    model_eval.pop(col, None)
            row.update(model_eval)
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
            for eval_col in model_eval.keys():
                if eval_col in EVAL_COLUMNS:
                    continue
                value = row.get(eval_col)
                if value not in (None, "", []) and eval_col not in dynamic_eval_columns_present:
                    dynamic_eval_columns_present.append(eval_col)

        # Keep epoch loss columns ordered by epoch index (loss_e1, loss_e2, ...).
        epoch_loss_columns = sorted(
            (c for c in dynamic_stats_columns_present if c.startswith("loss_e")),
            key=lambda c: int(c.split("loss_e", 1)[1]),
        )
        other_dynamic_stats_columns = [
            c for c in dynamic_stats_columns_present if not c.startswith("loss_e")
        ]

        stats_columns_without_train = [c for c in stats_columns_present if c != "train_time"]
        train_time_columns = [c for c in stats_columns_present if c == "train_time"]

        ordered_columns: list[str] = []
        for col in (
                ["model_name"]
                + training_columns_order
                + model_columns_order
                + stats_columns_without_train
                + epoch_loss_columns
                + other_dynamic_stats_columns
                + train_time_columns
                + eval_columns_present
                + dynamic_eval_columns_present
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


def run_log_tools(log_path: str | Path, output_root: str | Path | None = None) -> Path:
    return log_to_model_csv(log_path, output_root=output_root)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert training log to per-model CSV files.")
    parser.add_argument(
        "log_path",
        nargs="?",
        default=None,
        help="Optional log path. If omitted, uses the latest .log file in project logs/.",
    )
    return parser


def find_latest_log_file() -> Path:
    logs_dir = get_root_dir() / "logs"
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
    args = build_cli_parser().parse_args()
    selected_log = args.log_path if args.log_path else find_latest_log_file()
    output_dir = run_log_tools(selected_log)
    print(output_dir)
