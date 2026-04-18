from pathlib import Path

import joblib
import pandas as pd
import torch

from models.dl import cnn
from models.ml import decision_tree, random_forest, svm
from tools import util


def merge_samples_labels(
        source_subfolder: str = None,
        sample_percentage: float = 100.0
):
    root_dir = util.get_root_dir()
    samples_labels_dir = root_dir / 'data/samples_labels'
    samples_dir = root_dir / 'data/samples'
    output_file = samples_labels_dir / 'merged_samples_labels.csv'

    if output_file.exists():
        output_file.unlink()
        print(f"Deleted existing merged CSV: {output_file}")

    all_dfs: list[pd.DataFrame] = []

    if source_subfolder:
        folders_to_process = [samples_labels_dir / f"{source_subfolder}_labels"]
    else:
        folders_to_process = [f for f in samples_labels_dir.iterdir() if f.is_dir()]

    for subfolder in folders_to_process:
        if not subfolder.exists():
            print(f"Subfolder does not exist: {subfolder}")
            continue

        sample_name = subfolder.name.replace('_labels', '') if subfolder.name.endswith('_labels') else subfolder.name
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

    if not (0 < sample_percentage <= 100):
        raise ValueError(f"sample_percentage must be in (0, 100], got {sample_percentage}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        if sample_percentage < 100:
            seed = 42
            combined_df = combined_df.sample(
                frac=sample_percentage / 100,
                random_state=seed
            ).reset_index(drop=True)
            print(f"Randomly kept {sample_percentage}% samples. Remaining rows: {len(combined_df)}")
        combined_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved: {output_file}")
    else:
        print("No CSV files found to merge.")


def load_csv_data():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    csv_file = root_dir / 'data/samples_labels/merged_samples_labels.csv'

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    img_paths = [str(Path(row['source_folder']) / row['filename']) for _, row in df.iterrows()]
    y = df['label'].tolist()

    return img_paths, y


def train_models(
        patch_size: int = 32,
        PCA_components=100,
        sample_percentage: float = 100.0
):
    merge_samples_labels(sample_percentage=sample_percentage)
    img_paths, y = load_csv_data()
    root_dir = util.get_root_dir()

    current_file = Path(__file__).resolve()
    model_dir = current_file.parent / 'model_save'
    model_dir.mkdir(exist_ok=True)

    decision_tree_model, decision_tree_pca = decision_tree.train_decision_tree(img_paths, y, patch_size,PCA_components)
    joblib.dump(decision_tree_model, model_dir / 'decision_tree_model.joblib')
    joblib.dump(decision_tree_pca, model_dir / 'decision_tree_pca.joblib')

    random_forest_model, random_forest_pca = random_forest.train_random_forest(img_paths, y, patch_size,PCA_components)
    joblib.dump(random_forest_model, model_dir / 'random_forest_model.joblib')
    joblib.dump(random_forest_pca, model_dir / 'random_forest_pca.joblib')

    svm_model, svm_pca = svm.train_svm(img_paths, y, patch_size,PCA_components)
    joblib.dump(svm_model, model_dir / 'svm_model.joblib')
    joblib.dump(svm_pca, model_dir / 'svm_pca.joblib')

    lmdb_path = root_dir / f"data/samples_labels/patches_ps{patch_size}.lmdb"
    cnn_model = cnn.train_cnn(
        img_paths,
        y,
        patch_size,
        lmdb_path=str(lmdb_path),
        build_lmdb_if_missing=True,
        assume_fixed_size=True
    )
    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'num_classes': cnn_model.num_classes,
        'patch_size': patch_size
    }, model_dir / 'cnn_model.pth')

    print(f"All models saved to: {model_dir}")


if __name__ == "__main__":
    train_models(patch_size=32, PCA_components=100, sample_percentage=100.0)

