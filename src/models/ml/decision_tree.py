import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from src.tools import util, pca


def train_tree(img_paths, y, patch_size, n_components=100, n_jobs=None):
    y = np.array(y)

    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count // 2)
    print(f"[Decision Tree] Using {n_jobs} CPU cores for image loading.")

    print("[Decision Tree] Start loading images ...")
    start_load = time.time()
    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)
    end_load = time.time()
    print(f"[Decision Tree] Image loading done, time: {end_load - start_load:.2f}s")
    print(f"X shape before PCA: {X.shape}")

    print("[Decision Tree] Start PCA preprocessing ...")
    start_pca = time.time()
    X_reduced, pca_model = pca.reduce_dimensions(X, n_components)
    end_pca = time.time()
    print(f"[Decision Tree] PCA preprocessing done, time: {end_pca - start_pca:.2f}s")
    print(f"X shape after PCA: {X_reduced.shape}")

    model = DecisionTreeClassifier(
        max_depth=16,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    print("[Decision Tree] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print(f"[Decision Tree] Training done, time: {end_train - start_train:.2f}s")

    return model, pca_model
