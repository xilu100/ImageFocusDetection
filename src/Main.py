import shutil
from pathlib import Path

from src.evaluate import evaluate_all
from src.preprocessing import normalize_raw, segment_nor_img, label_patches, visualize_labels
from src.training import train_all


def main():
    patch_size = 32
    process = 0
    train = 1
    evaluate = 1

    if process:
        print("=== Step 1: Preprocessing ===")
        delete_folder()

        normalize_raw.normalize_images(patch_size)
        segment_nor_img.segment_images(patch_size)
        label_patches.label(top_percent=75)
        visualize_labels.visualize()

    if train:
        print("=== Step 2: Training ===")
        train_all.train_models(patch_size)

    if evaluate:
        print("=== Step 3: Evaluation ===")
        evaluate_all.evaluate_valid_set()


def delete_folder():
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
            print(f"The folder does not exist: {folder}")


if __name__ == "__main__":
    main()
