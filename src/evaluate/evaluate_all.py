from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.dl.cnn import SimpleCNN
from src.tools import util


# =========================
# 1. 合并验证集 CSV（优化：避免重复执行）
# =========================
def merge_valid_samples_labels():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    labels_dir = root_dir / 'data/valid_samples_labels'
    samples_dir = root_dir / 'data/valid_samples'
    output_file = labels_dir / 'merged_valid_samples_labels.csv'

    if output_file.exists():
        print(f"[INFO] using existing merged CSV: {output_file}")
        return

    all_dfs = []
    for subfolder in labels_dir.iterdir():
        if not subfolder.is_dir():
            continue

        sample_name = subfolder.name.replace('_labels', '')
        source_folder_path = samples_dir / sample_name

        for csv_file in subfolder.glob('*.csv'):
            df = pd.read_csv(csv_file)
            df['source_folder'] = str(source_folder_path)
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No validation CSV files found.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"[INFO] merged validation csv: {output_file}")


# =========================
# 2. 加载验证数据（增加字段校验）
# =========================
def load_valid_data():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    csv_file = root_dir / 'data/valid_samples_labels/merged_valid_samples_labels.csv'

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)

    required_columns = ['filename', 'label', 'source_folder']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    img_paths = [str(Path(row['source_folder']) / row['filename']) for _, row in df.iterrows()]
    y = df['label'].tolist()

    return img_paths, y


# =========================
# 3. 构建 ML 特征
# =========================
def build_X(img_paths, patch_size):
    X = [util.img_to_X(p, patch_size) for p in img_paths]
    return np.array(X, dtype=np.float32)


# =========================
# 4. CNN 输入
# =========================
def load_img_for_cnn(path, patch_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size))

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)

    return img


# =========================
# 5. CNN 批量预测（性能优化）
# =========================
def cnn_predict_batch(model, img_paths, patch_size, device, batch_size=64):
    model.eval()
    preds = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]

        batch_imgs = [load_img_for_cnn(p, patch_size) for p in batch_paths]
        batch_imgs = np.stack(batch_imgs, axis=0)  # (N,1,H,W)

        batch_tensor = torch.from_numpy(batch_imgs).to(device)

        with torch.no_grad():
            out = model(batch_tensor)
            batch_preds = out.argmax(dim=1).cpu().numpy()

        preds.extend(batch_preds)

    return preds


# =========================
# 6. 评估函数
# =========================
def evaluate(name, y_true, y_pred):
    print(f"\n========== {name} ==========")

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# =========================
# 7. 主函数（完整修复）
# =========================
def evaluate_valid_set(patch_size=32):
    merge_valid_samples_labels()

    img_paths, y_true = load_valid_data()
    X = build_X(img_paths, patch_size)

    current_file = Path(__file__).resolve()
    model_dir = current_file.parents[1] / 'training/model_save'

    # =========================
    # ML 模型（关键修复：加入 PCA）
    # =========================
    decision_tree_model = joblib.load(model_dir / 'decision_tree_model.joblib')
    decision_tree_pca = joblib.load(model_dir / 'decision_tree_pca.joblib')

    random_forest_model = joblib.load(model_dir / 'random_forest_model.joblib')
    random_forest_pca = joblib.load(model_dir / 'random_forest_pca.joblib')

    svm_model = joblib.load(model_dir / 'svm_model.joblib')
    svm_pca = joblib.load(model_dir / 'svm_pca.joblib')

    # ⚠️ 必须 transform（否则结果是错的）
    X_dt = decision_tree_pca.transform(X)
    X_rf = random_forest_pca.transform(X)
    X_svm = svm_pca.transform(X)

    y_pred_dt = decision_tree_model.predict(X_dt)
    evaluate("Decision Tree", y_true, y_pred_dt)

    y_pred_rf = random_forest_model.predict(X_rf)
    evaluate("Random Forest", y_true, y_pred_rf)

    y_pred_svm = svm_model.predict(X_svm)
    evaluate("SVM", y_true, y_pred_svm)

    # =========================
    # CNN 模型（修复：正确加载参数）
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_dir / 'cnn_model.pth', map_location=device)

    cnn_model = SimpleCNN(
        patch_size=checkpoint['patch_size'],
        num_classes=checkpoint['num_classes']
    )

    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    cnn_model.to(device)
    cnn_model.eval()

    y_pred_cnn = cnn_predict_batch(
        cnn_model,
        img_paths,
        checkpoint['patch_size'],
        device
    )

    evaluate("CNN", y_true, y_pred_cnn)


# =========================
if __name__ == "__main__":
    evaluate_valid_set(patch_size=32)