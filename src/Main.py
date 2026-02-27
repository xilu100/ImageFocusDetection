import os
import shutil

from src.preprocessing import normalize_raw, segment_nor_img, label_patches, visualize_labels


def main():
    delete_folder()
    normalize_raw.normalize_images()
    segment_nor_img.segment_images(estimated_patches=2000)
    label_patches.label(thresholds=[200, 210, 220, 230])
    visualize_labels.visualize()


def delete_folder():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    samples_dir = os.path.join(parent_dir, 'data/samples')
    samples_label_dir = os.path.join(parent_dir, 'data/samples_labels')
    print(samples_dir)
    print(samples_label_dir)
    folders_to_remove = [
        samples_label_dir,
        samples_dir,
    ]

    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted folder : {folder}")
        else:
            print(f"The folder does not exist.: {folder}")


if __name__ == "__main__":
    main()
