import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from tools import util, pca
from tools.log import print_and_save, save


def train_random_forest(
        img_paths,
        y,
        patch_size,
        n_components=0.95,
        n_jobs=None,
        model_params=None
):
    # Prepare labels and worker count.
    y = np.array(y)

    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count // 2)
    print_and_save(f"[Random Forest] Using {n_jobs} CPU cores for image loading.")

    # Load image features in parallel.
    print("[Random Forest] Start loading images ...")
    start_load = time.time()
    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)
    end_load = time.time()
    print_and_save(f"[Random Forest] Image loading done, time: {end_load - start_load:.2f}s")
    print_and_save(f"[Random Forest] X shape before PCA: {X.shape}")

    # Reduce feature dimensions with PCA.
    print("[Random Forest] Start PCA preprocessing ...")
    start_pca = time.time()
    X_reduced, pca_model = pca.reduce_dimensions(X, n_components)
    end_pca = time.time()
    print_and_save(f"[Random Forest] PCA preprocessing done, time: {end_pca - start_pca:.2f}s")
    print_and_save(f"[Random Forest] X shape after PCA: {X_reduced.shape}")

    # Define model hyperparameters and train.
    default_model_params = {
        "n_estimators": 50,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    }
    model_params = model_params or {}
    final_model_params = {**default_model_params, **model_params}

    save("---- Random Forest ----")
    save(final_model_params)

    model = RandomForestClassifier(**final_model_params)
    print("[Random Forest] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print_and_save(f"[Random Forest] Training done, time: {end_train - start_train:.2f}s")
    save("---- Random Forest ----\n")
    return model, pca_model
