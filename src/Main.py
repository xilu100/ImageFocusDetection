import shutil
from pathlib import Path

from evaluate import evaluate_all
from preprocessing import normalize_raw, segment_nor_img, label_patches
from tools.log import print_and_save, save
from training import train_all


def main():
    # =========================
    # Hyperparameters
    # =========================
    save("Main Hyperparameters")
    # Patch size; recommended values: [16, 32, 64, 128]; must be a multiple of 16
    patch_size = 32
    save(patch_size)

    # Scoring thresholds; range: [0, 100]
    top_percent = 75
    low_percent = 10
    save(top_percent)
    save(low_percent)

    # Number of PCA components; use -1 to disable PCA; otherwise range: [1, patch_size**2]
    PCA_components = 100
    save(PCA_components)

    # Sampling percentage; range: [0, 100]
    samples_percentage = 80
    save(samples_percentage)
    save("\n")
    # =========================
    # Pipeline switches
    # =========================

    process = 1
    train = 1
    evaluate = 1

    if process:
        print_and_save("=== Step 1: Preprocessing ===")
        delete_folder()

        normalize_raw.normalize_images(patch_size)
        segment_nor_img.segment_images(patch_size)
        label_patches.label(top_percent, low_percent)
        from preprocessing import visualize_labels
        visualize_labels.visualize()

    if train:
        print_and_save("=== Step 2: Training ===")
        train_all.train_models(patch_size, PCA_components, samples_percentage)

    if evaluate:
        print_and_save("=== Step 3: Evaluation ===")
        evaluate_all.evaluate_valid_set(patch_size)


def delete_folder():
    """
    Remove intermediate data folders and saved models to ensure a clean run.
    """
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent

    samples_dir = parent_dir / "data" / "samples"
    samples_label_dir = parent_dir / "data" / "samples_labels"

    valid_samples_dir = parent_dir / "data" / "valid_samples"
    valid_samples_label_dir = parent_dir / "data" / "valid_samples_labels"

    model_dir = parent_dir / "src" / "training" / "model_save"

    folders_to_remove = [
        samples_label_dir,
        samples_dir,
        valid_samples_label_dir,
        valid_samples_dir,
        model_dir,
    ]

    for folder in folders_to_remove:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")
        else:
            print(f"Folder does not exist: {folder}")


if __name__ == "__main__":
    main()
