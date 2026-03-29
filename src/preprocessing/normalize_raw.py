import csv
from pathlib import Path

import cv2

# 支持的宽高比
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
    height, width = image.shape[:2]
    _, target_ratio = find_closest_ratio(width, height)

    new_width = width
    new_height = int(round(width / target_ratio))

    if abs(new_height - height) > abs(new_width - width):
        new_height = height
        new_width = int(round(height * target_ratio))

    new_width = max(base, round(new_width / base) * base)
    new_height = max(base, round(new_height / base) * base)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized, (new_width, new_height)


def _load_processed_files(csv_path):
    """
    从 CSV 中读取已处理的 original_filename（第5列）
    """
    processed = set()

    if not csv_path.exists():
        return processed

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) >= 5:
                processed.add(row[4])

    return processed


def _get_start_counter(output_dir, prefix):
    """
    根据已有文件确定 counter 起始值
    """
    existing_files = [
        f.name for f in output_dir.iterdir()
        if f.is_file() and f.name.startswith(prefix) and f.suffix == ".png"
    ]

    return max(
        [int(f.replace(prefix, "").replace(".png", "")) for f in existing_files],
        default=0
    ) + 1


def _process_dataset(root_dir, input_dir, output_dir, patch_size, prefix, csv_name="samples_info.csv"):
    """
    核心处理函数
    """
    raw_image_dir = root_dir / input_dir
    output_image_dir = root_dir / output_dir
    output_image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_image_dir / csv_name

    items = [p for p in raw_image_dir.iterdir() if p.is_file()]

    processed_files = _load_processed_files(csv_path)
    counter = _get_start_counter(output_image_dir, prefix)

    file_exists = csv_path.exists()

    with csv_path.open(mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # 写表头（仅一次）
        if not file_exists or csv_path.stat().st_size == 0:
            writer.writerow([
                "filename",
                "original_size",
                "current_size",
                "aspect_ratio",
                "original_filename"
            ])

        for item in items:
            # 跳过已处理
            if item.name in processed_files:
                print(f"Skipped: {item.name}")
                continue

            image = cv2.imread(str(item))
            if image is None:
                print(f"Warning: Could not read image {item.name}")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            orig_height, orig_width = gray_image.shape[:2]

            resized_image, (new_width, new_height) = resize_image(gray_image, patch_size)
            closest_ratio_key, _ = find_closest_ratio(orig_width, orig_height)

            output_filename = f"{prefix}{counter}.png"
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


def normalize_images(patch_size=64):
    """
    对外接口（可直接调用）
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    # 数据集配置：input_dir, output_dir, prefix, csv_name
    datasets = [
        ("data/raw/train_img", "data/normalized", "sample", "samples_info.csv"),
        ("data/raw/valid_img", "data/valid_normalized", "valid_sample", "valid_samples_info.csv"),
    ]

    for input_dir, output_dir, prefix, csv_name in datasets:
        _process_dataset(root_dir, input_dir, output_dir, patch_size, prefix, csv_name)


if __name__ == "__main__":
    normalize_images(patch_size=32)
