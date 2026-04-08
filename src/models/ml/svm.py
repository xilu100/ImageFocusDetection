import os
import time
import numpy as np
from joblib import Parallel, delayed

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from src.tools import util, pca


def train_svm(
    img_paths,
    y,
    patch_size,
    use_sgd=False,
    n_components=100,
    batch_size=10000,
    n_epochs=20,
    n_jobs=None
):
    y = np.array(y)

    # =========================
    # CPU 核心数
    # =========================
    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count // 2)

    print(f"[SVM] Using {n_jobs} CPU cores for image loading.")

    # =========================
    # 加载图片
    # =========================
    print("[SVM] Start loading images ...")
    start_load = time.time()

    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)

    end_load = time.time()
    print(f"[SVM] Image loading done, time: {end_load - start_load:.2f}s")
    print(f"X shape before PCA: {X.shape}")

    # =========================
    # PCA（保持和你原来一致）
    # =========================
    if n_components is not None:
        print(f"[SVM] Start PCA to {n_components} ...")
        start_pca = time.time()

        X_reduced, pca_model = pca.reduce_dimensions(X, n_components)

        end_pca = time.time()
        print(f"[SVM] PCA done, time: {end_pca - start_pca:.2f}s")
        print(f"X shape after PCA: {X_reduced.shape}")
    else:
        X_reduced = X
        pca_model = None

    classes = np.unique(y)

    # =========================
    # 模型训练
    # =========================
    if use_sgd:
        print("[SGD] Start training ...")

        model = SGDClassifier(
            loss='hinge',
            max_iter=1,
            tol=None,
            class_weight='balanced',
            learning_rate='optimal'
        )

        start_train = time.time()

        for epoch in range(n_epochs):
            perm = np.random.permutation(len(X_reduced))
            X_shuffled = X_reduced[perm]
            y_shuffled = y[perm]

            for i in range(0, len(X_reduced), batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                model.partial_fit(batch_X, batch_y, classes=classes)

        end_train = time.time()
        print(f"[SGD] Training done, time: {end_train - start_train:.2f}s")

    else:
        print("[LinearSVC] Start training ...")

        model = LinearSVC(
            dual=False,
            max_iter=5000,
            tol=1e-4,
            class_weight='balanced',
            C=1.0
        )

        start_train = time.time()
        model.fit(X_reduced, y)
        end_train = time.time()

        print(f"[LinearSVC] Training done, time: {end_train - start_train:.2f}s")

    return model, pca_model