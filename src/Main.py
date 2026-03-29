import shutil
from pathlib import Path

from src.evaluate import evaluate_all
from src.preprocessing import normalize_raw, segment_nor_img, label_patches, visualize_labels
from src.training import train_all


def main():
    patch_size = 32
    delete_folder()
    normalize_raw.normalize_images(patch_size)
    segment_nor_img.segment_images(patch_size)
    label_patches.label(thresholds=[200, 210, 220, 230])
    visualize_labels.visualize()
    train_all.train_models(patch_size)
    evaluate_all.evaluate_valid_set()


def delete_folder():
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent

    # 原来的 samples 目录
    samples_dir = parent_dir / "data" / "samples"
    samples_label_dir = parent_dir / "data" / "samples_labels"

    # 新增 valid_samples 目录
    valid_samples_dir = parent_dir / "data" / "valid_samples"
    valid_samples_label_dir = parent_dir / "data" / "valid_samples_labels"

    model_dir = parent_dir / "src" / "training" / "model_save"

    print(samples_dir)
    print(samples_label_dir)
    print(valid_samples_dir)
    print(valid_samples_label_dir)
    print(model_dir)

    # 所有需要删除的文件夹
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
