import os
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from src.tools import util, pca

def train_random_forest(img_paths, y, patch_size, n_estimators=50, max_depth=10, n_components=100, n_jobs=None):
    y = np.array(y)

    # ===== 自适应 CPU 核心数 =====
    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count // 2)
    print(f"[Random Forest] Using {n_jobs} CPU cores for image loading.")

    # ===== 加载图片 =====
    print("[Random Forest] Start loading images ...")
    start_load = time.time()
    X = Parallel(n_jobs=n_jobs)(
        delayed(util.img_to_X)(p, patch_size) for p in img_paths
    )
    X = np.array(X, dtype=np.float32)
    end_load = time.time()
    print(f"[Random Forest] Image loading done, time: {end_load - start_load:.2f}s")
    print(f"X shape before PCA: {X.shape}")

    # ===== PCA =====
    print("[Random Forest] Start PCA preprocessing ...")
    start_pca = time.time()
    X_reduced, pca_model = pca.reduce_dimensions(X, n_components)
    end_pca = time.time()
    print(f"[Random Forest] PCA preprocessing done, time: {end_pca - start_pca:.2f}s")
    print(f"X shape after PCA: {X_reduced.shape}")

    # ===== 模型训练 =====
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    print("[Random Forest] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print(f"[Random Forest] Training done, time: {end_train - start_train:.2f}s")

    return model, pca_model