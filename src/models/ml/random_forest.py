import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models import img_to_X



def train_random_forest(img_paths, y, patch_size, n_estimators=100, max_depth=None):
    X = np.array([img_to_X.convert(p, patch_size) for p in img_paths])
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1  # 多线程训练加速
    )

    model.fit(X, y)
    return model