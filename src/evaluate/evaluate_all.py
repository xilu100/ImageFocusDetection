from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.dl.cnn import SimpleCNN  # 你的CNN类

from src.tools import util


# =========================
# 1. 合并验证集 CSV
# =========================
def merge_valid_samples_labels():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]

    labels_dir = root_dir / 'data/valid_samples_labels'
    samples_dir = root_dir / 'data/valid_samples'
    output_file = labels_dir / 'merged_valid_samples_labels.csv'

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
# 2. 加载验证数据
# =========================
def load_valid_data():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    csv_file = root_dir / 'data/valid_samples_labels/merged_valid_samples_labels.csv'
    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)
    img_paths = [str(Path(row['source_folder']) / row['filename']) for _, row in df.iterrows()]
    y = df['y_thresh2'].tolist()
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
        raise ValueError(f"Can not read image: {path}")
    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1,H,W)
    return img


def cnn_predict(model, img_paths, patch_size, device):
    model.eval()
    preds = []
    for p in img_paths:
        img = load_img_for_cnn(p, patch_size)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,1,H,W)
        with torch.no_grad():
            out = model(img_tensor)
            pred = out.argmax(dim=1).item()
        preds.append(pred)
    return preds


# =========================
# 5. 评估函数
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
# 6. 主函数
# =========================
def evaluate_valid_set(patch_size=32):
    merge_valid_samples_labels()
    img_paths, y_true = load_valid_data()
    X = build_X(img_paths, patch_size)

    current_file = Path(__file__).resolve()
    model_dir = current_file.parents[1] / 'training/model_save'

    # =========================
    # ML 模型
    # =========================
    decision_tree_model = joblib.load(model_dir / 'decision_tree_model.joblib')
    random_forest_model = joblib.load(model_dir / 'random_forest_model.joblib')
    svm_model = joblib.load(model_dir / 'svm_model.joblib')

    y_pred_dt = decision_tree_model.predict(X)
    evaluate("Decision Tree", y_true, y_pred_dt)

    y_pred_rf = random_forest_model.predict(X)
    evaluate("Random Forest", y_true, y_pred_rf)

    y_pred_svm = svm_model.predict(X)
    evaluate("SVM", y_true, y_pred_svm)

    # =========================
    # CNN 模型
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_dir / 'cnn_model.pth', map_location=device)

    cnn_model = SimpleCNN(checkpoint['patch_size'])
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    cnn_model.to(device)
    cnn_model.eval()

    y_pred_cnn = cnn_predict(cnn_model, img_paths, checkpoint['patch_size'], device)
    evaluate("CNN", y_true, y_pred_cnn)


# =========================
if __name__ == "__main__":
    evaluate_valid_set(patch_size=32)
