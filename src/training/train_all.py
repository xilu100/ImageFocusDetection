from pathlib import Path
import joblib
import pandas as pd
import torch

from src.models.dl import cnn
from src.models.ml import decision_tree, random_forest, svm
from src.tools import util


def merge_samples_labels(source_subfolder: str = None):
    """
    Merge all CSV label files into a single CSV file.

    Args:
        source_subfolder (str, optional): If provided, only process this subfolder.
    """
    root_dir = util.get_root_dir()
    samples_labels_dir = root_dir / 'data/samples_labels'
    samples_dir = root_dir / 'data/samples'
    output_file = samples_labels_dir / 'merged_samples_labels.csv'

    # Remove existing merged CSV if present
    if output_file.exists():
        output_file.unlink()
        print(f"Deleted existing merged CSV: {output_file}")

    all_dfs = []

    # Select subfolders to process
    if source_subfolder:
        folders_to_process = [samples_labels_dir / f"{source_subfolder}_labels"]
    else:
        folders_to_process = [f for f in samples_labels_dir.iterdir() if f.is_dir()]

    for subfolder in folders_to_process:
        if not subfolder.exists():
            print(f"Subfolder does not exist: {subfolder}")
            continue

        # Map labels folder to original sample folder
        sample_name = subfolder.name.replace('_labels', '') if subfolder.name.endswith('_labels') else subfolder.name
        source_folder_path = samples_dir / sample_name

        # Read all CSV files in subfolder
        for csv_file in subfolder.glob('*.csv'):
            df = pd.read_csv(csv_file)
            df['source_folder'] = str(source_folder_path)
            all_dfs.append(df)

    # Concatenate and save merged CSV
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved: {output_file}")
    else:
        print("No CSV files found to merge.")


def load_csv_data(patch_size: int = 16):
    """
    Load image paths and labels from merged CSV.

    Args:
        patch_size (int): Size of image patches (currently unused here).

    Returns:
        img_paths (list[str]): List of image file paths.
        y (list[int]): Corresponding labels.
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    csv_file = root_dir / 'data/samples_labels/merged_samples_labels.csv'

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    img_paths = [str(Path(row['source_folder']) / row['filename']) for _, row in df.iterrows()]
    y = df['label'].tolist()

    return img_paths, y


def train_models(patch_size: int = 32):
    """
    Train multiple models (Decision Tree, Random Forest, SVM, CNN) on dataset.

    Args:
        patch_size (int): Size of image patches.
    """
    # Merge label CSVs and load data
    merge_samples_labels()
    img_paths, y = load_csv_data(patch_size)

    # Prepare model saving directory
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent / 'model_save'
    model_dir.mkdir(exist_ok=True)

    # ================= Decision Tree =================
    decision_tree_model, decision_tree_pca = decision_tree.train_tree(img_paths, y, patch_size)
    joblib.dump(decision_tree_model, model_dir / 'decision_tree_model.joblib')
    joblib.dump(decision_tree_pca, model_dir / 'decision_tree_pca.joblib')

    # ================= Random Forest =================
    random_forest_model, random_forest_pca = random_forest.train_random_forest(img_paths, y, patch_size)
    joblib.dump(random_forest_model, model_dir / 'random_forest_model.joblib')
    joblib.dump(random_forest_pca, model_dir / 'random_forest_pca.joblib')

    # ================= SVM =================
    svm_model, svm_pca = svm.train_svm(img_paths, y, patch_size)
    joblib.dump(svm_model, model_dir / 'svm_model.joblib')
    joblib.dump(svm_pca, model_dir / 'svm_pca.joblib')

    # ================= CNN =================
    cnn_model = cnn.train_cnn(img_paths, y, patch_size)
    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'num_classes': cnn_model.num_classes,
        'patch_size': patch_size
    }, model_dir / 'cnn_model.pth')

    print(f"All models saved to: {model_dir}")


if __name__ == "__main__":
    train_models(patch_size=32)