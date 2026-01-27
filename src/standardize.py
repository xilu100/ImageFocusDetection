import json
import os

import cv2


def standardize():
    # Paths
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    input_dir = os.path.join(parent_dir, "pictures")
    output_dir = os.path.join(parent_dir, "normalised")
    os.makedirs(output_dir, exist_ok=True)

    # Standard aspect ratios
    ratios = {
        "1_1": 1 / 1,
        "3_2": 3 / 2,
        "2_3": 2 / 3,
        "4_3": 4 / 3,
        "3_4": 3 / 4,
        "16_9": 16 / 9,
        "9_16": 9 / 16
    }

    # JSON file to record processed images
    json_path = os.path.join(output_dir, "records.json")

    # Load existing records to avoid processing duplicates
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = []

    processed_files = set(r["original_name"] for r in records)
    sample_counter = len(records) + 1

    # Get all image files and sort them to ensure order consistency
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
    image_files = sorted(image_files)  # 确保 macOS 和 Windows 顺序一致

    for img_name in image_files:
        if img_name in processed_files:
            print(f"Skipping already processed file: {img_name}")
            continue

        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read file: {img_name}, skipping")
            continue

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orig_h, orig_w = img_gray.shape
        orig_ratio = orig_w / orig_h

        # Find the closest standard aspect ratio
        closest_ratio_name = min(ratios, key=lambda r: abs(ratios[r] - orig_ratio))
        target_ratio = ratios[closest_ratio_name]

        # Determine target dimensions, keep long edge
        if target_ratio >= 1:
            target_w = orig_w
            target_h = int(target_w / target_ratio)
        else:
            target_h = orig_h
            target_w = int(target_h * target_ratio)

        # Resize (stretch) image
        img_resized = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Save output
        out_name = f"sample_{sample_counter}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), img_resized)

        # Record information
        records.append({
            "sample": f"sample_{sample_counter}",
            "original_name": img_name,
            "original_width": orig_w,
            "original_height": orig_h,
            "normalized_width": target_w,
            "normalized_height": target_h,
            "ratio": closest_ratio_name
        })

        sample_counter += 1
        processed_files.add(img_name)

    # Save JSON record
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. {sample_counter - 1} images generated. Records saved to {json_path}")


if __name__ == "__main__":
    standardize()
