import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from tools import util, pca
from tools.log import print_and_save, save


def train_decision_tree(img_paths, y, patch_size, n_components=100, n_jobs=None):
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
    max_depth = 16  # 树的最大深度，限制模型复杂度，降低过拟合风险。
    min_samples_split = 50  # 一个节点继续分裂所需的最小样本数。
    min_samples_leaf = 20  # 叶子节点最少样本数，避免叶子过小导致不稳定。
    class_weight = 'balanced'  # 按类别频率自动加权，缓解类别不平衡。
    random_state = 42  # 固定随机种子，保证训练结果可复现。
    save("---- Decision Tree ----")
    save(max_depth)
    save(min_samples_split)
    save(min_samples_leaf)
    save(class_weight)
    save(random_state)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
    )
    print("[Decision Tree] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print_and_save(f"[Decision Tree] Training done, time: {end_train - start_train:.2f}s")
    save("---- Decision Tree ----\n")
    return model, pca_model
