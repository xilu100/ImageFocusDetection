import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from src.tools import util

def train_svm(img_paths, y, patch_size, use_sgd=False, pca_dim=None, batch_size=10000, n_epochs=20):
    """
    训练 SVM
    :param img_paths: 图片路径列表
    :param y: 标签列表
    :param patch_size: 每个 patch 大小
    :param use_sgd: 是否使用增量训练（SGDClassifier）
    :param pca_dim: 如果不为 None，则先降维到 pca_dim 维
    :param batch_size: SGD 每批大小
    :param n_epochs: SGD 训练轮数
    :return: 训练好的模型
    """

    # 1️⃣ 准备数据
    X = [util.img_to_X(p, patch_size) for p in img_paths]
    X = np.array(X)
    y = np.array(y)

    # 2️⃣ 可选 PCA 降维
    if pca_dim is not None:
        print(f"[SVM] Applying PCA to reduce dimension to {pca_dim} ...")
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(X)
        print("[SVM] PCA finished.")

    classes = np.unique(y)

    # 3️⃣ 初始化模型
    if use_sgd:
        # 使用增量训练 SGDClassifier
        model = SGDClassifier(loss='hinge', max_iter=1, tol=None)
        print(f"[SGD] Start incremental training: batch_size={batch_size}, n_epochs={n_epochs} ...")
        start_time = time.time()

        for epoch in range(n_epochs):
            # 可以打乱样本顺序，提高训练效果
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                model.partial_fit(batch_X, batch_y, classes=classes)

    else:
        # 使用 LinearSVC
        model = LinearSVC(dual=False, max_iter=2000, tol=1e-3)
        print("[LinearSVC] Start training ...")
        start_time = time.time()
        model.fit(X, y)

    end_time = time.time()
    print(f"[SVM] Training time: {end_time - start_time:.4f} seconds")

    return model