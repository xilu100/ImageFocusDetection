import csv
import os

import cv2


def compute_laplacian_score(gray_image):
    """
    Compute Laplacian variance score for a grayscale image.
    Higher value means sharper image.
    """
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def label(thresholds=None):
    # ---- Define thresholds internally ----

    if thresholds is None:
        thresholds = [200, 210, 220, 230]
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)

    input_dir = os.path.join(root_dir, 'data', 'samples')
    output_root = os.path.join(root_dir, 'data', 'samples_labels')

    os.makedirs(output_root, exist_ok=True)

    for sample_folder in os.listdir(input_dir):
        sample_path = os.path.join(input_dir, sample_folder)

        if not os.path.isdir(sample_path):
            continue

        sample_output_dir = os.path.join(output_root, f"{sample_folder}_labels")
        os.makedirs(sample_output_dir, exist_ok=True)

        csv_path = os.path.join(
            sample_output_dir,
            f"{sample_folder}_laplacian.csv"
        )

        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Header
            header = ['filename', 'laplacian_score']
            for i in range(len(thresholds)):
                header.append(f'y_thresh{i + 1}')
            writer.writerow(header)

            for file_name in os.listdir(sample_path):
                file_path = os.path.join(sample_path, file_name)

                if not os.path.isfile(file_path):
                    continue

                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                score = compute_laplacian_score(image)

                row = [file_name, score]

                for thresh in thresholds:
                    row.append(1 if score >= thresh else 0)

                writer.writerow(row)

        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    label(thresholds=[200, 210, 220, 230])
