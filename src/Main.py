import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, TypedDict

from evaluate import evaluate_all as evaluate_runner
from preprocessing import normalize_raw, segment_nor_img, label_patches
from tools.log import print_and_save, save, flush_logs, get_current_log_paths, close_logs
from tools.log_tools import run_log_tools
from tools.plot_csv import run_plot_cli
from training import train_all


def get_experiment_config():
    return {
        "training": {
            # Patch size (px). Options: [16, 32, 64, 128], default: 32
            "patch_size": 32,
            # Sharp score threshold. Range: [0.0, 1.0], default: 0.75
            "sharp_threshold": 0.75,
            # Blur score threshold. Range: [0.0, 1.0], default: 0.40
            "blur_threshold": 0.40,
            # PCA components. Options: [-1 (off), 0.9, 0.95, 0.99], default: 0.95
            "PCA_components": 0.95,
            # Train sample ratio (%). Range: (0, 100], default: 100
            "sample_percentage": 100,
        },
        "models": {
            # Decision Tree params
            "decision_tree": {
                # Max depth. Options: [8, 12, 16, 24], default: 16
                "max_depth": 16,
                # Min split samples. Range: int >= 2 or float in (0, 1], default: 50
                "min_samples_split": 50,
                # Min leaf samples. Range: int >= 1 or float in (0, 0.5], default: 20
                "min_samples_leaf": 20,
                # Class weight. Options: {"balanced", None}, default: "balanced"
                "class_weight": "balanced",
                # Random seed. Options: [42, 123, 2024, 3407], default: 42
                "random_state": 42,
            },
            # Random Forest params
            "random_forest": {
                # Trees count. Options: [50, 100, 200, 300], default: 100
                "n_estimators": 100,
                # Max depth. Options: [8, 10, 16, None], default: 16
                "max_depth": 16,
                # Random seed. Options: [42, 123, 2024, 3407], default: 42
                "random_state": 42,
                # Class weight. Options: {"balanced_subsample", "balanced", None}, default: "balanced_subsample"
                "class_weight": "balanced_subsample",
                # Parallel workers. Options: [-1, 1, 2, 4], default: -1
                "n_jobs": -1,
            },
            # SVM + Nystroem params
            "svm": {
                # Nystroem components. Options: [100, 200, 300, 500], default: 300
                "nystroem_components": 300,
                # Nystroem kernel. Options: {"rbf", "cosine", "poly", "sigmoid"}, default: "rbf"
                "nystroem_kernel": "rbf",
                # Gamma. Options: [None, 1e-4, 1e-3, 1e-2], default: None
                "nystroem_gamma": None,
                # Random seed. Options: [42, 123, 2024, 3407], default: 42
                "random_state": 42,
                # C. Range: float > 0, default: 1.0
                "svc_c": 1.0,
                # Class weight. Options: {"balanced", None}, default: "balanced"
                "class_weight": "balanced",
                # Max iterations. Options: [2000, 5000, 10000, 20000], default: 5000
                "max_iter": 5000,
            },
            # CNN params
            "cnn": {
                # Epochs. Options: [5, 10, 15, 20], default: 10
                "epochs": 10,
                # Base batch size. Options: [32, 64, 128, 256], default: 128
                "batch_base": 128,
                # Random seed. Options: [42, 123, 2024, 3407], default: 42
                "seed": 42,
                # Learning rate. Range: float > 0, default: 1e-3
                "learning_rate": 1e-3,
                # Build LMDB if missing. Options: {True, False}, default: True
                "build_lmdb_if_missing": True,
                # Enforce fixed patch size. Options: {True, False}, default: True
                "assume_fixed_size": True,
            },
        },
    }


def get_control_config():
    return {
        "pipeline": {
            # Preprocessing switch. Options: {0, 1}, default: 1
            "preprocess": 1,
            # Bound train+evaluate switch. Options: {0, 1}, default: 1
            "train_evaluate": 1,
        },
        "models": {
            # Per-model switches. Options: {0, 1}, default: 1
            "decision_tree": 1,
            "random_forest": 1,
            "svm": 1,
            "cnn": 1,
        },
    }


def get_plot_config():
    return {
        "time_metrics": {
            # Time metrics switches. Options: {0, 1}
            "img_load_time": 1,
            "pca_time": 1,
            "lmdb_time": 1,
            "train_time": 1,
        },
        "evaluate_metrics": {
            # class 0 row
            "class0_precision": 1,
            "class0_recall": 1,
            "class0_f1": 1,
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

            # confusion matrix entries
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


def main():
    experiment_cfg = get_experiment_config()
    control_cfg = get_control_config()
    plot_cfg = get_plot_config()

    save(experiment_cfg)
    save(control_cfg)
    save("\n")

    training_cfg = experiment_cfg["training"]
    pipeline_cfg = control_cfg.get("pipeline", {})
    preprocess_enabled = bool(pipeline_cfg.get("preprocess", 1))
    train_evaluate_enabled = bool(pipeline_cfg.get("train_evaluate", 1))
    enabled_models = get_enabled_models(control_cfg)
    adapt_plot_config_for_label_mode(training_cfg, plot_cfg)
    sweep_targets = find_sweep_targets(experiment_cfg)

    if len(sweep_targets) > 1:
        target_names = [target["path_str"] for target in sweep_targets]
        raise ValueError(
            "Only one parameter can be a list at a time. "
            f"Found: {target_names}"
        )

    target: SweepTarget | None
    if len(sweep_targets) == 1:
        selected_target = sweep_targets[0]
        target = selected_target
        run_values = list(selected_target["values"])
        print_and_save(f"Sweep target: {selected_target['path_str']} -> {run_values}")
    else:
        target = None
        run_values = [None]
        print_and_save("No sweep target found. Running once with current config.")

    if not (preprocess_enabled or train_evaluate_enabled):
        print_and_save("All pipeline switches are disabled. Nothing to run.")
        return

    if preprocess_enabled:
        print_and_save("=== Step 1: Preprocessing (Run Once) ===")
        run_pipeline_once(
            training_cfg=training_cfg,
            experiment_cfg=experiment_cfg,
            preprocess=1,
            train_models=0,
            evaluate=0,
            enabled_models=enabled_models,
            run_tag=None,
        )

    if not train_evaluate_enabled:
        print_and_save("Train + Evaluate is disabled. Pipeline finished.")
        return

    print_and_save("=== Mode: Train + Evaluate ===")

    created_evaluate_dirs: set[Path] = set()

    for run_idx, run_value in enumerate(run_values, start=1):
        if target is not None:
            set_nested_value(experiment_cfg, target["path"], run_value)
            print_and_save(
                f"=== Run {run_idx}/{len(run_values)} | {target['path_str']}={run_value} ==="
            )
        else:
            print_and_save("=== Run 1/1 ===")

        run_tag = build_run_tag(target, run_value)
        if run_tag:
            print_and_save(f"Evaluate output tag: {run_tag}")

        run_evaluate_dirs = run_pipeline_once(
            training_cfg=training_cfg,
            experiment_cfg=experiment_cfg,
            preprocess=0,
            train_models=train_evaluate_enabled,
            evaluate=train_evaluate_enabled,
            enabled_models=enabled_models,
            run_tag=run_tag,
        )
        created_evaluate_dirs.update(run_evaluate_dirs)

    out_dir, log_path, complete_log_path = prepare_run_outputs()
    try:
        collect_evaluate_outputs(out_dir, created_evaluate_dirs)
        plot_paths = run_plot_cli(sweep_dir=out_dir, plot_config=plot_cfg)
        print_and_save(f"Generated plots: {len(plot_paths)}")
    finally:
        finalize_packaged_logs(out_dir, log_path, complete_log_path)


def adapt_plot_config_for_label_mode(training_cfg: dict, plot_cfg: dict):
    sharp = training_cfg.get("sharp_threshold")
    blur = training_cfg.get("blur_threshold")
    try:
        sharp_f = float(sharp)
        blur_f = float(blur)
    except (TypeError, ValueError):
        return

    if abs(sharp_f - blur_f) > 1e-8:
        return

    eval_metrics = plot_cfg.get("evaluate_metrics", {})
    for col in ("class2_precision", "class2_recall", "class2_f1", "class2_support"):
        if col in eval_metrics:
            eval_metrics[col] = 0


class SweepTarget(TypedDict):
    path: tuple[str, ...]
    path_str: str
    values: list[Any]


def run_pipeline_once(
        training_cfg,
        experiment_cfg,
        preprocess,
        train_models,
        evaluate,
        enabled_models: dict[str, bool],
        run_tag: str | None = None,
):
    patch_size = training_cfg["patch_size"]

    if preprocess:
        print("=== Step 1: Preprocessing ===")
        process_start = time.perf_counter()
        delete_folder()

        normalize_raw.normalize_images(patch_size)
        segment_nor_img.segment_images(patch_size)
        label_patches.label(training_cfg["sharp_threshold"], training_cfg["blur_threshold"])
        from preprocessing import visualize_labels
        visualize_labels.visualize()
        process_elapsed = time.perf_counter() - process_start
        print_and_save(f"Preprocessing total time: {process_elapsed:.2f}s")

    if train_models or evaluate:
        enabled_model_names = [name for name, enabled in enabled_models.items() if enabled]
        if not enabled_model_names:
            print_and_save("No model is enabled. Skip training and evaluate.")
            return []

        print_and_save(f"Enabled models: {', '.join(enabled_model_names)}")
        if train_models:
            print_and_save("=== Step 2: Training ===")
            train_all.train_models(
                patch_size=patch_size,
                PCA_components=training_cfg["PCA_components"],
                sample_percentage=training_cfg["sample_percentage"],
                config=experiment_cfg,
                enabled_models=enabled_models,
            )
            log_model_artifacts_snapshot()

        if evaluate:
            print_and_save("=== Step 3: Evaluate ===")
            return evaluate_runner.evaluate_valid_set(
                patch_size,
                run_tag=run_tag,
                enabled_models=enabled_models,
            )
    return []


def get_enabled_models(control_cfg: dict) -> dict[str, bool]:
    models_cfg = control_cfg.get("models", {})
    return {
        "decision_tree": bool(models_cfg.get("decision_tree", 1)),
        "random_forest": bool(models_cfg.get("random_forest", 1)),
        "svm": bool(models_cfg.get("svm", 1)),
        "cnn": bool(models_cfg.get("cnn", 1)),
    }


def log_model_artifacts_snapshot():
    model_dir = Path(__file__).resolve().parent / "training" / "model_save"
    print(f"Model artifacts directory: {model_dir}")
    if not model_dir.exists():
        print("Model artifacts directory does not exist yet.")
        return

    model_files = sorted(p for p in model_dir.iterdir() if p.is_file())
    if not model_files:
        print("No model artifact files found.")
        return

    for file_path in model_files:
        modified_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_path.stat().st_mtime))
        print(f"Artifact: {file_path.name} | modified_at={modified_at}")


def sanitize_tag_part(value: object) -> str:
    text = str(value)
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return sanitized if sanitized else "value"


def build_run_tag(target: "SweepTarget | None", run_value: object) -> str | None:
    if target is None:
        return None
    path_str = target["path_str"]
    if path_str.startswith("models.decision_tree."):
        prefix = "DT"
    elif path_str.startswith("models.random_forest."):
        prefix = "RF"
    elif path_str.startswith("models.svm."):
        prefix = "SVM"
    elif path_str.startswith("models.cnn."):
        prefix = "CNN"
    elif path_str.startswith("training."):
        prefix = "TR"
    else:
        prefix = "CTRL"
    control_name = target["path"][-1] if target["path"] else target["path_str"]
    return f"{prefix}_{sanitize_tag_part(control_name)}_{sanitize_tag_part(run_value)}"


def collect_evaluate_outputs(out_dir: Path, created_evaluate_dirs: set[Path]):
    evaluate_out_dir = out_dir / "predict_images"
    copied = 0
    parent_dirs: set[Path] = set()
    for src_dir in sorted(created_evaluate_dirs):
        if not src_dir.exists() or not src_dir.is_dir():
            print_and_save(f"Skip missing evaluate folder: {src_dir}")
            continue
        dst_dir = evaluate_out_dir / src_dir.name
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        parent_dirs.add(src_dir.parent)
        copied += 1

    # Cleanup temporary evaluate cache folders after packaging outputs.
    for src_dir in sorted(created_evaluate_dirs, reverse=True):
        if src_dir.exists() and src_dir.is_dir():
            shutil.rmtree(src_dir, ignore_errors=True)

    # Remove empty cache root(s), e.g. logs/_evaluate_cache
    for parent_dir in sorted(parent_dirs, reverse=True):
        if parent_dir.exists() and parent_dir.is_dir():
            try:
                parent_dir.rmdir()
            except OSError:
                # Directory not empty (or in use); keep it.
                pass

    print_and_save(f"Collected {copied} predict image folder(s) to: {evaluate_out_dir}")


def find_sweep_targets(
        node: Any, path: tuple[str, ...] = ()
) -> list[SweepTarget]:
    targets: list[SweepTarget] = []
    if isinstance(node, dict):
        for key, value in node.items():
            targets.extend(find_sweep_targets(value, path + (key,)))
    elif isinstance(node, (list, tuple)):
        if len(path) > 0:
            targets.append(
                {
                    "path": path,
                    "path_str": ".".join(path),
                    "values": list(node),
                }
            )
    return targets


def set_nested_value(node, path, value):
    current = node
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def delete_folder():
    """
    Remove intermediate data folders to ensure a clean preprocess run.
    """
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent

    samples_dir = parent_dir / "data" / "samples"
    samples_label_dir = parent_dir / "data" / "samples_labels"

    valid_samples_dir = parent_dir / "data" / "valid_samples"
    valid_samples_label_dir = parent_dir / "data" / "valid_samples_labels"

    folders_to_remove = [
        samples_label_dir,
        samples_dir,
        valid_samples_label_dir,
        valid_samples_dir,
    ]

    for folder in folders_to_remove:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")
        else:
            print(f"Folder does not exist: {folder}")


def prepare_run_outputs() -> tuple[Path, Path, Path]:
    flush_logs()
    log_path, complete_log_path = get_current_log_paths()
    if log_path is None or complete_log_path is None:
        raise RuntimeError("Current log paths are unavailable.")

    out_dir = run_log_tools(log_path)
    print_and_save(f"Packaged outputs: {out_dir}")
    return out_dir, log_path, complete_log_path


def finalize_packaged_logs(out_dir: Path, log_path: Path, complete_log_path: Path):
    flush_logs()
    close_logs()

    target_log = out_dir / log_path.name
    target_complete_log = out_dir / complete_log_path.name
    if target_log.exists():
        target_log.unlink()
    if target_complete_log.exists():
        target_complete_log.unlink()
    shutil.move(str(log_path), str(target_log))
    shutil.move(str(complete_log_path), str(target_complete_log))

    # Keep folder timestamp aligned with log timestamp.
    log_mtime = target_log.stat().st_mtime
    os.utime(out_dir, (log_mtime, log_mtime))


if __name__ == "__main__":
    main()
