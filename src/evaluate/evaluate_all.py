from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.dl import cnn
from preprocessing import visualize_labels
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

    return img_paths, y, df


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


def safe_model_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def model_name_suffix(name: str) -> str:
    return {
        "Decision Tree": "RT",
        "Random Forest": "RF",
        "SVM": "SVM",
        "CNN": "CNN",
    }.get(name, safe_model_name(name).upper())


def save_prediction_outputs(
        model_name: str,
        valid_df: pd.DataFrame,
        y_pred,
):
    root_dir = util.get_root_dir()
    model_suffix = model_name_suffix(model_name)
    labels_dir = root_dir / "data/valid_samples_labels"
    raw_dir = root_dir / "data/raw/valid_img"
    sample_map = visualize_labels.load_mapping(root_dir / "data/valid_normalized/valid_samples_info.csv")

    pred_df = valid_df.copy()
    pred_df["predicted_label"] = np.asarray(y_pred, dtype=np.int16)

    for source_folder, group in pred_df.groupby("source_folder", sort=True):
        sample_name = Path(source_folder).name
        label_folder = labels_dir / f"{sample_name}_labels"
        original_csv = label_folder / f"{sample_name}.csv"

        if original_csv.exists():
            output_df = pd.read_csv(original_csv)
            output_df["original_label"] = output_df["label"]
            pred_map = dict(zip(group["filename"], group["predicted_label"]))
            output_df["predicted_label"] = output_df["filename"].map(pred_map).astype("Int64")
            output_df["label"] = output_df["predicted_label"].fillna(output_df["original_label"]).astype(np.int16)
        else:
            output_df = group.copy()
            output_df["original_label"] = output_df["label"]
            output_df["label"] = output_df["predicted_label"].astype(np.int16)

        sample_output_dir = label_folder / f"{sample_name}_labels_predict"
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        key = f"{sample_name}.png"
        if key not in sample_map:
            print_and_save(f"[{model_name}] Prediction overlay skipped, mapping missing: {sample_name}")
            continue

        original_path = raw_dir / sample_map[key]
        original_img = cv2.imread(str(original_path))
        if original_img is None:
            print_and_save(f"[{model_name}] Prediction overlay skipped, cannot read original: {original_path}")
            continue

        if output_df.empty:
            print_and_save(f"[{model_name}] Prediction overlay skipped, empty output: {sample_name}")
            continue

        parsed = [visualize_labels.parse_row_col(name) for name in output_df["filename"]]
        rows, cols = zip(*parsed)
        grid_rows, grid_cols = max(rows) + 1, max(cols) + 1

        label_heatmap = visualize_labels.generate_label_heatmap(output_df, grid_rows, grid_cols)
        h, w = original_img.shape[:2]
        label_heatmap_resized = cv2.resize(label_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
        label_overlay = visualize_labels.overlay_heatmap_on_image(original_img, label_heatmap_resized)
        label_overlay_path = sample_output_dir / f"{sample_name}_label_overlay_{model_suffix}.png"
        cv2.imwrite(str(label_overlay_path), label_overlay)

    print_and_save(f"[{model_name}] Prediction overlays saved under: {labels_dir}")


def evaluate_valid_set(patch_size: int = 32):
    merge_valid_samples_labels()

    img_paths, y, valid_df = load_valid_data()
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
    save_prediction_outputs("Decision Tree", valid_df, y_pred_dt)

    y_pred_rf = random_forest_model.predict(X_rf)
    evaluate("Random Forest", y, y_pred_rf)
    save_prediction_outputs("Random Forest", valid_df, y_pred_rf)

    y_pred_svm = svm_model.predict(X_svm)
    evaluate("SVM", y, y_pred_svm)
    save_prediction_outputs("SVM", valid_df, y_pred_svm)

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
    save_prediction_outputs("CNN", valid_df, y_pred_cnn)


if __name__ == "__main__":
    evaluate_valid_set(patch_size=32)
