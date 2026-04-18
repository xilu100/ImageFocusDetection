import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tools import util, pca


def train_svm(img_paths, y, patch_size, n_components=100, n_jobs=None):
    # Prepare labels and worker count.
    y = np.array(y)

    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count // 2)
    print(f"[SVM] Using {n_jobs} CPU cores for image loading.")

    # Load image features in parallel.
    print("[SVM] Start loading images ...")
    start_load = time.time()
    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)
    end_load = time.time()
    print(f"[SVM] Image loading done, time: {end_load - start_load:.2f}s")
    print(f"X shape before PCA: {X.shape}")

    # Reduce feature dimensions with PCA.
    print("[SVM] Start PCA preprocessing ...")
    start_pca = time.time()
    X_reduced, pca_model = pca.reduce_dimensions(X, n_components)
    end_pca = time.time()
    print(f"[SVM] PCA preprocessing done, time: {end_pca - start_pca:.2f}s")
    print(f"X shape after PCA: {X_reduced.shape}")

    # Define model hyperparameters and train.
    nystroem_components = 300  # Nystroem映射后的特征维度，越大越能近似核方法但开销更高。
    nystroem_kernel = 'rbf'  # 核近似所用核函数类型。
    nystroem_gamma = None  # RBF核系数；None表示按sklearn默认策略自动设置。
    random_state = 42  # 固定随机种子，保证特征映射可复现。
    svc_c = 2.0  # 线性SVM正则化系数，越大越强调训练集拟合。
    class_weight = 'balanced'  # 按类别频率自动加权，缓解类别不平衡。
    max_iter = 5000  # 求解器最大迭代次数，防止未收敛时无限迭代。

    model = make_pipeline(
        StandardScaler(),
        Nystroem(
            kernel=nystroem_kernel,
            gamma=nystroem_gamma,
            n_components=nystroem_components,
            random_state=random_state
        ),
        LinearSVC(
            C=svc_c,
            class_weight=class_weight,
            max_iter=max_iter
        ),
    )
    print("[SVM] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print(f"[SVM] Training done, time: {end_train - start_train:.2f}s")

    return model, pca_model
