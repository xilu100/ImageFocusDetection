import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tools import util, pca
from tools.log import print_and_save, save


def train_svm(
        img_paths,
        y,
        patch_size,
        n_components=0.95,
        n_jobs=None,
        model_params=None
):
    # Prepare labels and worker count.
    y = np.array(y)
    unique_labels = np.unique(y).tolist()
    print_and_save(f"[SVM] Detected classes: {unique_labels}")

    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count // 2)
    print_and_save(f"[SVM] Using {n_jobs} CPU cores for image loading.")

    # Load image features in parallel.
    print("[SVM] Start loading images ...")
    start_load = time.time()
    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)
    end_load = time.time()
    print_and_save(f"[SVM] Image loading done, time: {end_load - start_load:.2f}s")
    print_and_save(f"[SVM] X shape before PCA: {X.shape}")

    # Reduce feature dimensions with PCA.
    print("[SVM] Start PCA preprocessing ...")
    start_pca = time.time()
    X_reduced, pca_model = pca.reduce_dimensions(X, n_components)
    end_pca = time.time()
    print_and_save(f"[SVM] PCA preprocessing done, time: {end_pca - start_pca:.2f}s")
    print_and_save(f"[SVM] X shape after PCA: {X_reduced.shape}")

    # Define model hyperparameters and train.
    default_model_params = {
        "nystroem_components": 300,
        "nystroem_kernel": "rbf",
        "nystroem_gamma": None,
        "random_state": 42,
        "svc_c": 2.0,
        "class_weight": "balanced",
        "max_iter": 5000,
    }
    model_params = model_params or {}
    final_model_params = {**default_model_params, **model_params}

    save(final_model_params)

    model = make_pipeline(
        StandardScaler(),
        Nystroem(
            kernel=final_model_params["nystroem_kernel"],
            gamma=final_model_params["nystroem_gamma"],
            n_components=final_model_params["nystroem_components"],
            random_state=final_model_params["random_state"]
        ),
        LinearSVC(
            C=final_model_params["svc_c"],
            class_weight=final_model_params["class_weight"],
            max_iter=final_model_params["max_iter"]
        ),
    )
    print("[SVM] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print_and_save(f"[SVM] Training done, time: {end_train - start_train:.2f}s")
    save("\n")
    return model, pca_model
