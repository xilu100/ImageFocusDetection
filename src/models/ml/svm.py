import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.tools import util, pca


def train_svm(
        img_paths,
        y,
        patch_size,
        n_components=100,
        nystroem_components=300,
        n_jobs=None
):
    y = np.array(y)

    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count // 2)

    print(f"[SVM] Using {n_jobs} CPU cores for image loading.")

    print("[SVM] Start loading images ...")
    start_load = time.time()

    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)

    end_load = time.time()
    print(f"[SVM] Image loading done, time: {end_load - start_load:.2f}s")
    print(f"X shape before PCA: {X.shape}")

    print("[SVM] Start PCA preprocessing ...")
    start_pca = time.time()
    X_reduced, pca_model = pca.reduce_dimensions(X, n_components)
    end_pca = time.time()
    print(f"[SVM] PCA preprocessing done, time: {end_pca - start_pca:.2f}s")
    print(f"X shape after PCA: {X_reduced.shape}")

    model = make_pipeline(
        StandardScaler(),
        Nystroem(
            kernel='rbf',
            gamma=None,
            n_components=nystroem_components,
            random_state=42
        ),
        LinearSVC(
            C=2.0,
            class_weight='balanced',
            max_iter=5000
        ),
    )
    print("[SVM] Start training ...")

    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()

    print(f"[SVM] Training done, time: {end_train - start_train:.2f}s")

    return model, pca_model
