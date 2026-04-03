import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.tools import util


def train_random_forest(img_paths, y, patch_size, n_estimators=100, max_depth=None):
    X = []
    for p in img_paths:
        X.append(util.img_to_X(p, patch_size))
    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    print("[Random Forest] Start training ...")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    print(f"[Random Forest] Training time: {end_time - start_time:.4f} seconds")

    return model
