import os

import cv2
import pandas as pd
def features_labels(laplacian_threshold=215):
    # ========================
    # Path settings
    # ========================
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)

    input_dir = os.path.join(parent_dir, "samples")
    picture_dir = os.path.join(parent_dir, "pictures")
    output_dir = os.path.join(parent_dir, "samples_labels")

    os.makedirs(output_dir, exist_ok=True)

    # ========================
    # Parameters
    # ========================
    #laplacian_threshold = 215
    alpha = 0.4  # overlay transparency

    # BGR colors
    color_label_1 = (0, 0, 255)  # red
    color_label_0 = (255, 0, 0)  # blue


    # ========================
    # Utils
    # ========================
    def laplacian_score(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()


    def parse_patch_name(name):
        """
        sample_1_0_6.jpg -> row=0, col=6
        """
        base = os.path.splitext(name)[0]
        parts = base.split("_")
        return int(parts[-2]), int(parts[-1])


    # ========================
    # Main
    # ========================
    for sample_folder in sorted(os.listdir(input_dir)):
        sample_path = os.path.join(input_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue

        print(f"Processing {sample_folder}")

        sample_id = int(sample_folder.split("_")[1])
        original_img_name = f"{sample_id:04d}.png"
        original_img_path = os.path.join(picture_dir, original_img_name)

        if not os.path.exists(original_img_path):
            print(f"Missing original image: {original_img_name}")
            continue

        original_img = cv2.imread(original_img_path)
        overlay = original_img.copy()

        records = []

        patch_files = sorted([
            f for f in os.listdir(sample_path)
            if f.lower().endswith((".jpg", ".png"))
        ])

        if len(patch_files) == 0:
            continue

        # get patch size
        sample_patch = cv2.imread(os.path.join(sample_path, patch_files[0]))
        ph, pw = sample_patch.shape[:2]

        for patch_name in patch_files:
            patch_path = os.path.join(sample_path, patch_name)
            patch_img = cv2.imread(patch_path)

            score = laplacian_score(patch_img)
            label = 1 if score >= laplacian_threshold else 0

            row, col = parse_patch_name(patch_name)

            y1, y2 = row * ph, (row + 1) * ph
            x1, x2 = col * pw, (col + 1) * pw

            color = color_label_1 if label == 1 else color_label_0
            overlay[y1:y2, x1:x2] = color

            records.append([
                patch_name,
                round(score, 2),
                label
            ])

        # ========================
        # Save CSV
        # ========================
        csv_path = os.path.join(
            output_dir, f"{sample_folder}_label.csv"
        )
        pd.DataFrame(
            records,
            columns=["patch_name", "laplacian_score", "label"]
        ).to_csv(csv_path, index=False)

        # ========================
        # Blend & save visualization
        # ========================
        blended = cv2.addWeighted(
            original_img, 1 - alpha,
            overlay, alpha, 0
        )

        vis_path = os.path.join(
            output_dir, f"{sample_folder}_binary_map.jpg"
        )
        cv2.imwrite(vis_path, blended)

        print(f"Saved: {csv_path}")
        print(f"Saved: {vis_path}")

    print("Done.")
