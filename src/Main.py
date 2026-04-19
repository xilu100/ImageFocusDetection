import os
import shutil
from pathlib import Path
from typing import Any, TypedDict

from evaluate import evaluate_all
from preprocessing import normalize_raw, segment_nor_img, label_patches
from tools.log import print_and_save, save, flush_logs, get_current_log_paths, close_logs
from tools.log_tools import log_to_model_csv
from training import train_all


def get_experiment_config():
    return {
        "training": {
            # Patch size (px). Options: [16, 32, 64, 128], default: 32
            "patch_size": 32,
            # Sharp threshold (%). Range: [0, 100], default: 75
            "top_percent": 75,
            # Blur threshold (%). Range: [0, 100], default: 10
            "low_percent": 10,
            # PCA components (-1 disables PCA). Options: [-1, 64, 100, 256], default: 100
            "PCA_components": 100,
            # Train sample ratio (%). Range: (0, 100], default: 80
            "sample_percentage": 80,
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
                # Trees count. Options: [50, 100, 200, 300], default: 50
                "n_estimators": 50,
                # Max depth. Options: [8, 10, 16, None], default: 10
                "max_depth": 10,
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
                # C. Range: float > 0, default: 2.0
                "svc_c": 2.0,
                # Class weight. Options: {"balanced", None}, default: "balanced"
                "class_weight": "balanced",
                # Max iterations. Options: [2000, 5000, 10000, 20000], default: 5000
                "max_iter": 5000,
            },
            # CNN params
            "cnn": {
                # Epochs. Options: [5, 10, 15, 20], default: 5
                "epochs": 5,
                # Base batch size. Options: [32, 64, 128, 256], default: 64
                "batch_base": 64,
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


def main():
    # Preprocessing switch. Options: {0, 1}, default: 1
    process = 1
    # Train + Evaluation switch. Options: {0, 1}, default: 1
    train_and_evaluate = 1

    experiment_cfg = get_experiment_config()

    save(experiment_cfg)
    save("\n")

    training_cfg = experiment_cfg["training"]
    sweep_targets = _find_sweep_targets(experiment_cfg)

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
            _set_nested_value(experiment_cfg, target["path"], run_value)
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

    _package_run_outputs()


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
        delete_folder()

        normalize_raw.normalize_images(patch_size)
        segment_nor_img.segment_images(patch_size)
        label_patches.label(training_cfg["top_percent"], training_cfg["low_percent"])
        from preprocessing import visualize_labels
        visualize_labels.visualize()

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


def _find_sweep_targets(
        node: Any, path: tuple[str, ...] = ()
) -> list[SweepTarget]:
    targets: list[SweepTarget] = []
    if isinstance(node, dict):
        for key, value in node.items():
            targets.extend(_find_sweep_targets(value, path + (key,)))
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


def _set_nested_value(node, path, value):
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


def _package_run_outputs():
    flush_logs()
    log_path, complete_log_path = get_current_log_paths()
    if log_path is None or complete_log_path is None:
        raise RuntimeError("Current log paths are unavailable.")

    out_dir = log_to_model_csv(log_path)
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


if __name__ == "__main__":
    main()
