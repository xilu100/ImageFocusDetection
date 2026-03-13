import cv2
import numpy as np

def convert(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # 保证尺寸一致
    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0
    return img.flatten()