from pathlib import Path

import cv2
import numpy as np


def get_root_dir():
    root_path = Path(__file__).resolve().parent.parent.parent
    return root_path


def img_to_X(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Can not read image: {img_path}")

    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0

    return img.flatten()
