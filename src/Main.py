import os
import shutil
import time
from pathlib import Path
from typing import Any, TypedDict

from evaluate import evaluate_all
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
            # Sharp threshold (%). Range: [0, 100], default: 75
            "top_percent": 75,
            # Discard threshold (%): unidentifiable or extremely blurred. Range: [0, 100], default: 10
            "low_percent": 10,
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


def get_plot_config():
    return {
        "modules": {
            # Plot time chart. Options: {0, 1}, default: 1
            "time": 1,
            # Plot evaluation chart. Options: {0, 1}, default: 1
            "evaluate": 1,
        },
        "time_metrics": {
            # Time metrics switches. Options: {0, 1}
            "img_load_time": 1,
            "pca_time": 1,
            "lmdb_time": 1,
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

            # confusion matrix entries
            "cm_tn": 0,
            "cm_fp": 0,
            "cm_fn": 0,
            "cm_tp": 0,
        },
    }


def main():
    # Preprocessing switch. Options: {0, 1}, default: 1
    process = 1
    # Train + Evaluation switch. Options: {0, 1}, default: 1
    train_and_evaluate = 1

    experiment_cfg = get_experiment_config()
    plot_cfg = get_plot_config()

    save(experiment_cfg)
    save("\n")

    training_cfg = experiment_cfg["training"]
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

    if process and not train_and_evaluate:
        print_and_save("=== Mode: Process Only ===")
        run_pipeline_once(
            training_cfg=training_cfg,
            experiment_cfg=experiment_cfg,
            process=1,
            train_and_evaluate=0,
        )
        return

    if train_and_evaluate and not process:
        print_and_save("=== Mode: Train + Evaluate Only ===")
    elif process and train_and_evaluate:
        print_and_save("=== Mode: Process Once + Train + Evaluate Sweep ===")
        run_pipeline_once(
            training_cfg=training_cfg,
            experiment_cfg=experiment_cfg,
            process=1,
            train_and_evaluate=0,
        )
    else:
        print_and_save("Both process and train_and_evaluate are disabled. Nothing to run.")
        return

    for run_idx, run_value in enumerate(run_values, start=1):
        if target is not None:
            set_nested_value(experiment_cfg, target["path"], run_value)
            print_and_save(
                f"=== Run {run_idx}/{len(run_values)} | {target['path_str']}={run_value} ==="
            )
        else:
            print_and_save("=== Run 1/1 ===")

        run_pipeline_once(
            training_cfg=training_cfg,
            experiment_cfg=experiment_cfg,
            process=0,
            train_and_evaluate=1,
        )

    out_dir = package_run_outputs()
    plot_paths = run_plot_cli(sweep_dir=out_dir, plot_config=plot_cfg)
    print(f"Generated plots: {len(plot_paths)}")


class SweepTarget(TypedDict):
    path: tuple[str, ...]
    path_str: str
    values: list[Any]


def run_pipeline_once(
        training_cfg,
        experiment_cfg,
        process,
        train_and_evaluate,
):
    patch_size = training_cfg["patch_size"]

    if process:
        print_and_save("=== Step 1: Preprocessing ===")
        process_start = time.perf_counter()
        delete_folder()

        normalize_raw.normalize_images(patch_size)
        segment_nor_img.segment_images(patch_size)
        label_patches.label(training_cfg["top_percent"], training_cfg["low_percent"])
        from preprocessing import visualize_labels
        visualize_labels.visualize()
        process_elapsed = time.perf_counter() - process_start
        print_and_save(f"Preprocessing total time: {process_elapsed:.2f}s")

    if train_and_evaluate:
        print_and_save("=== Step 2: Training ===")
        train_all.train_models(
            patch_size=patch_size,
            PCA_components=training_cfg["PCA_components"],
            sample_percentage=training_cfg["sample_percentage"],
            config=experiment_cfg
        )

        print_and_save("=== Step 3: Evaluation ===")
        evaluate_all.evaluate_valid_set(patch_size)


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


def package_run_outputs():
    flush_logs()
    log_path, complete_log_path = get_current_log_paths()
    if log_path is None or complete_log_path is None:
        raise RuntimeError("Current log paths are unavailable.")

    out_dir = run_log_tools(log_path)
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

    print(f"Packaged outputs: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
