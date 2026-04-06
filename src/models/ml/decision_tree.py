import time
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from src.tools import util


def train_tree(img_paths, y, patch_size, n_components=100, batch_size=20000, n_jobs=2):
    """
    内存优化版 Decision Tree 训练
    - 使用 IncrementalPCA 分批降维
    - 分批加载图像特征，避免一次性占用内存
    - n_jobs 限制并行进程，减少内存峰值
    """
    num_samples = len(img_paths)
    y = np.array(y)

    print("[Decision Tree] Start preprocessing and PCA in batches ...")
    start_time = time.time()

    # 生成器：分批加载图像并 flatten
    def batch_generator(paths, batch_size):
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            X_batch = Parallel(n_jobs=n_jobs)(
                delayed(util.img_to_X)(p, patch_size) for p in batch_paths
            )
            yield np.array(X_batch, dtype=np.float32)

    # Incremental PCA 分批 fit
    ipca = IncrementalPCA(n_components=n_components)
    for X_batch in batch_generator(img_paths, batch_size):
        ipca.partial_fit(X_batch)

    # 再次分批 transform 并累积结果（内存只占一批大小）
    X_reduced_list = []
    for X_batch in batch_generator(img_paths, batch_size):
        X_reduced_list.append(ipca.transform(X_batch))
    X_reduced = np.vstack(X_reduced_list)

    end_pca_time = time.time()
    print(f"[Decision Tree] PCA preprocessing done, time: {end_pca_time - start_time:.2f} s")

    # 决策树训练
    model = DecisionTreeClassifier(
        max_depth=16,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )

    print("[Decision Tree] Start training ...")
    start_train = time.time()
    model.fit(X_reduced, y)
    end_train = time.time()
    print(f"[Decision Tree] Training done, time: {end_train - start_train:.2f} s")

    return model