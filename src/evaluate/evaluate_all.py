from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.dl import cnn
from tools import util
from tools.log import print_and_save


def merge_valid_samples_labels(source_subfolder: str = None):
    root_dir = util.get_root_dir()

    labels_dir = root_dir / 'data/valid_samples_labels'
    samples_dir = root_dir / 'data/valid_samples'
    output_file = labels_dir / 'merged_valid_samples_labels.csv'

    if output_file.exists():
        output_file.unlink()
        print(f"Deleted existing merged CSV: {output_file}")

    all_dfs: list[pd.DataFrame] = []

    if source_subfolder:
        folders_to_process = [labels_dir / f"{source_subfolder}_labels"]
    else:
        folders_to_process = [f for f in labels_dir.iterdir() if f.is_dir()]

    for subfolder in folders_to_process:
        if not subfolder.exists():
            print(f"Subfolder does not exist: {subfolder}")
            continue

        sample_name = subfolder.name.replace('_labels', '')
        source_folder_path = samples_dir / sample_name

        for csv_file in subfolder.glob('*.csv'):
            df = pd.read_csv(csv_file)
            before_count = len(df)
            if 'label' in df.columns:
                df = df[df['label'] != -1].copy()
            removed_count = before_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} rows with label=-1 from: {csv_file}")
            df['source_folder'] = str(source_folder_path)
            all_dfs.append(df)

    if all_dfs:
        combined_df: pd.DataFrame = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved: {output_file}")
    else:
        print("No CSV files found to merge.")


def load_valid_data():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    csv_file = root_dir / 'data/valid_samples_labels/merged_valid_samples_labels.csv'

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    img_paths = [str(Path(row['source_folder']) / row['filename']) for _, row in df.iterrows()]
    y = df['label'].tolist()

    return img_paths, y


def build_X(img_paths, patch_size: int = 32):
    X = [util.img_to_X(p, patch_size) for p in img_paths]
    return np.array(X, dtype=np.float32)


def load_img_for_cnn(path, patch_size: int = 32):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size))

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def cnn_predict_batch(model, img_paths, patch_size: int, device, batch_size: int = 64):
    model.eval()
    preds = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]

        batch_imgs = [load_img_for_cnn(p, patch_size) for p in batch_paths]
        batch_imgs = np.stack(batch_imgs, axis=0)

        batch_tensor = torch.from_numpy(batch_imgs).to(device)

        with torch.no_grad():
            out = model(batch_tensor)
            batch_preds = out.argmax(dim=1).cpu().numpy()

        preds.extend(batch_preds)

    return preds


def evaluate(name, y, y_pred):
    print_and_save(f"========== {name} ==========")

    acc = accuracy_score(y, y_pred)
    print_and_save(f"Accuracy: {acc:.4f}")

    print_and_save("Classification Report:")
    print_and_save("\n", classification_report(y, y_pred))

    print_and_save("Confusion Matrix:")
    print_and_save("\n", confusion_matrix(y, y_pred))


def evaluate_valid_set(patch_size: int = 32):
    merge_valid_samples_labels()

    img_paths, y = load_valid_data()
    X = build_X(img_paths, patch_size)

    current_file = Path(__file__).resolve()
    model_dir = current_file.parents[1] / 'training/model_save'

    decision_tree_model = joblib.load(model_dir / 'decision_tree_model.joblib')
    decision_tree_pca = joblib.load(model_dir / 'decision_tree_pca.joblib')

    random_forest_model = joblib.load(model_dir / 'random_forest_model.joblib')
    random_forest_pca = joblib.load(model_dir / 'random_forest_pca.joblib')

    svm_model = joblib.load(model_dir / 'svm_model.joblib')
    svm_pca = joblib.load(model_dir / 'svm_pca.joblib')

    if decision_tree_pca is None:
        print("[Decision Tree] No PCA model found, evaluate with original features.")
    if random_forest_pca is None:
        print("[Random Forest] No PCA model found, evaluate with original features.")
    if svm_pca is None:
        print("[SVM] No PCA model found, evaluate with original features.")

    X_dt = X if decision_tree_pca is None else decision_tree_pca.transform(X)
    X_rf = X if random_forest_pca is None else random_forest_pca.transform(X)
    X_svm = X if svm_pca is None else svm_pca.transform(X)

    y_pred_dt = decision_tree_model.predict(X_dt)
    evaluate("Decision Tree", y, y_pred_dt)

    y_pred_rf = random_forest_model.predict(X_rf)
    evaluate("Random Forest", y, y_pred_rf)

    y_pred_svm = svm_model.predict(X_svm)
    evaluate("SVM", y, y_pred_svm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = model_dir / 'cnn_model.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility for older PyTorch versions without `weights_only`.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    cnn_model = cnn.SimpleCNN(
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

    evaluate("CNN", y, y_pred_cnn)


if __name__ == "__main__":
    evaluate_valid_set(patch_size=32)
