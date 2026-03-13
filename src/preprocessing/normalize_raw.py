import csv
from pathlib import Path

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


def resize_image(image, base):
    """
    Resize image so that width and height are multiples of `base`.
    """
    height, width = image.shape[:2]
    _, target_ratio = find_closest_ratio(width, height)

    # 按目标比例计算新尺寸
    new_width = width
    new_height = int(round(width / target_ratio))

    if abs(new_height - height) > abs(new_width - width):
        new_height = height
        new_width = int(round(height * target_ratio))

    # 调整到最接近的 base 倍数
    new_width = max(base, round(new_width / base) * base)
    new_height = max(base, round(new_height / base) * base)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized, (new_width, new_height)


def normalize_images(patch_size=64):
    """
    Normalize images to multiples of patch_size.
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    raw_image_dir = root_dir / "data/raw/train_img"
    output_image_dir = root_dir / "data/normalized"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_image_dir / "samples_info.csv"
    items = list(raw_image_dir.iterdir())

    existing_files = [
        f.name for f in output_image_dir.iterdir()
        if f.name.startswith("sample") and f.suffix == ".png"
    ]

    counter = max(
        [int(f.replace("sample", "").replace(".png", "")) for f in existing_files],
        default=0
    ) + 1

    with csv_path.open(mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if not csv_path.exists() or csv_path.stat().st_size == 0:
            writer.writerow([
                "filename",
                "original_size",
                "current_size",
                "aspect_ratio",
                "original_filename"
            ])

        for item in items:

            img_path = item
            image = cv2.imread(str(img_path))

            if image is None:
                print(f"Warning: Could not read image {item.name}")
                continue

            # Skip if already processed
            skip = False
            if csv_path.exists():
                with csv_path.open("r", encoding="utf-8") as f:
                    if item.name in f.read():
                        print(f"Skipped (already processed): {item.name}")
                        skip = True

            if skip:
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            orig_height, orig_width = gray_image.shape[:2]

            resized_image, (new_width, new_height) = resize_image(
                gray_image,
                patch_size
            )

            closest_ratio_key, _ = find_closest_ratio(orig_width, orig_height)

            output_filename = f"sample{counter}.png"
            output_path = output_image_dir / output_filename

            cv2.imwrite(str(output_path), resized_image)

            writer.writerow([
                output_filename,
                f"{orig_width}x{orig_height}",
                f"{new_width}x{new_height}",
                closest_ratio_key,
                item.name
            ])

            counter += 1
            print(f"Saved: {output_filename}")


if __name__ == "__main__":
    normalize_images(patch_size=32)
