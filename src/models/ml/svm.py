import time

import numpy as np
from sklearn.svm import SVC, LinearSVC

from src.tools import util


def train_svm(img_paths, y, patch_size):
    X = []
    for p in img_paths:
        X.append(util.img_to_X(p, patch_size))
    X = np.array(X)
    y = np.array(y)

    model = LinearSVC()

    print("[SVM] Start training ...")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    print(f"[SVM]Training time: {end_time - start_time:.4f} seconds")

    return model
