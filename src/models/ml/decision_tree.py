import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from tools import util, pca
from tools.log import print_and_save, save


def train_decision_tree(
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
    print_and_save(f"[Decision Tree] Using {n_jobs} CPU cores for image loading.")

    # Load image features in parallel.
    print("[Decision Tree] Start loading images ...")
    start_load = time.time()
    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)
    end_load = time.time()
    print_and_save(f"[Decision Tree] Image loading done, time: {end_load - start_load:.2f}s")
    print_and_save(f"[Decision Tree] X shape before PCA: {X.shape}")

    # Reduce feature dimensions with PCA.
    print("[Decision Tree] Start PCA preprocessing ...")
    start_pca = time.time()
    X_reduced, pca_model = pca.reduce_dimensions(X, n_components)
    end_pca = time.time()
    print_and_save(f"[Decision Tree] PCA preprocessing done, time: {end_pca - start_pca:.2f}s")
    print_and_save(f"[Decision Tree] X shape after PCA: {X_reduced.shape}")

    # Define model hyperparameters and train.
    default_model_params = {
        "max_depth": 16,
        "min_samples_split": 50,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "random_state": 42,
    }
    model_params = model_params or {}
    final_model_params = {**default_model_params, **model_params}

    save("---- Decision Tree ----")
    save(final_model_params)

    model = DecisionTreeClassifier(**final_model_params)
    print("[Decision Tree] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print_and_save(f"[Decision Tree] Training done, time: {end_train - start_train:.2f}s")
    save("---- Decision Tree ----\n")
    return model, pca_model
