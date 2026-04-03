import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.tools import util


def train_tree(img_paths, y, patch_size):
    X = []
    for p in img_paths:
        X.append(util.img_to_X(p, patch_size))
    X = np.array(X)
    y = np.array(y)

    model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    print("[Decision Tree] Start training ...")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    print(f"[Decision Tree] Training time: {end_time - start_time:.4f} seconds")

    return model
