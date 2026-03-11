from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_samples_and_labels(image_size=(64, 64)):
    current_file_path = Path(__file__).resolve()
    src_path = current_file_path.parent.parent.parent
    root_path = src_path.parent

    samples_path = root_path / "data" / "samples"
    labels_csv_path = root_path / "data" / "samples_labels" / "merged_all.csv"

    df = pd.read_csv(labels_csv_path)

    X_list = []
    y_list = []

    for idx, row in df.iterrows():
        file_name = row['filename']
        folder_name = file_name.split("_")[0]
        file_path = samples_path / folder_name / file_name

        if file_path.exists():
            img = Image.open(file_path).convert("L")
            img = img.resize(image_size)
            img_array = np.array(img).flatten()
            X_list.append(img_array)
            y_list.append(row['y_thresh2'])
        else:
            print(f"警告: 文件不存在 {file_path}")

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def train_svm_pca_classifier_and_evaluate(variance_ratio=1, image_size=(64,64)):
    # 加载数据
    X, y = load_samples_and_labels(image_size=image_size)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 降维，保留指定方差比例
    pca = PCA(n_components=variance_ratio)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA 后特征形状: {X_pca.shape}, 保留方差: {np.sum(pca.explained_variance_ratio_):.4f}")

    # SVM 分类器训练
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_pca, y)
    print("SVM 分类器训练完成")

    # 在训练集上预测
    y_pred = svm_model.predict(X_pca)

    # 训练集评估
    acc = accuracy_score(y, y_pred)
    print(f"训练集准确率: {acc:.4f}")

    print("分类报告:")
    print(classification_report(y, y_pred))

    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))

    return svm_model, pca, scaler, y_pred

if __name__ == "__main__":
    model, pca_model, scaler_model, y_pred = train_svm_pca_classifier_and_evaluate(
        variance_ratio=0.95, image_size=(64,64)
    )