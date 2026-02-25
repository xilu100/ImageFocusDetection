import csv
import os

import cv2

# Standard aspect ratios
IMAGE_RATIOS = {
    '1:1': 1 / 1,
    '3:2': 3 / 2,
    '4:3': 4 / 3,
    '16:9': 16 / 9,
    '2:3': 2 / 3,
    '3:4': 3 / 4,
    '9:16': 9 / 16,
}


def find_closest_ratio(width, height):
    """Find the closest standard aspect ratio."""
    current_ratio = width / height
    min_diff = float("inf")
    closest_key = None
    closest_ratio = None

    for key, ratio in IMAGE_RATIOS.items():
        diff = abs(current_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            closest_key = key
            closest_ratio = ratio

    return closest_key, closest_ratio


def resize_image(image):
    """Resize image to the closest standard aspect ratio."""
    height, width = image.shape[:2]
    _, target_ratio = find_closest_ratio(width, height)

    new_width = width
    new_height = int(round(width / target_ratio))

    if abs(new_height - height) > abs(new_width - width):
        new_height = height
        new_width = int(round(height * target_ratio))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized, (new_width, new_height)


def normalize_images():
    """Normalize all images: resize, convert to grayscale, save PNG, generate CSV."""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)

    raw_image_dir = os.path.join(root_dir, 'data/raw/train_img')
    output_image_dir = os.path.join(root_dir, 'data/normalized')
    os.makedirs(output_image_dir, exist_ok=True)

    csv_path = os.path.join(output_image_dir, "samples_info.csv")
    items = os.listdir(raw_image_dir)

    # Determine the starting counter based on existing files
    existing_files = [f for f in os.listdir(output_image_dir) if f.startswith("sample") and f.endswith(".png")]
    if existing_files:
        existing_numbers = [int(f.replace("sample", "").replace(".png", "")) for f in existing_files]
        counter = max(existing_numbers) + 1
    else:
        counter = 1

    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # If CSV is empty, write header
        if os.stat(csv_path).st_size == 0:
            writer.writerow(['filename', 'original_size', 'current_size', 'aspect_ratio', 'original_filename'])

        for item in items:
            img_path = os.path.join(raw_image_dir, item)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Warning: Could not read image {item}")
                continue

            # Check if this image has already been processed by matching original filename in CSV
            skip = False
            if os.path.exists(csv_path):
                with open(csv_path, mode='r', encoding='utf-8') as f:
                    if item in f.read():
                        print(f"Skipped (already processed): {item}")
                        skip = True

            if skip:
                continue

            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            orig_height, orig_width = gray_image.shape[:2]
            resized_image, (new_width, new_height) = resize_image(gray_image)
            closest_ratio_key, _ = find_closest_ratio(orig_width, orig_height)

            output_filename = f"sample{counter}.png"
            output_path = os.path.join(output_image_dir, output_filename)
            cv2.imwrite(output_path, resized_image)

            writer.writerow([
                output_filename,
                f"{orig_width}x{orig_height}",
                f"{new_width}x{new_height}",
                closest_ratio_key,
                item  # original file name only
            ])
            counter += 1
            print(f"Saved: {output_filename}")


if __name__ == "__main__":
    normalize_images()
