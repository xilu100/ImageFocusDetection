from pathlib import Path

import joblib
import pandas as pd
import torch

from src.models.dl import cnn
from src.models.ml import decision_tree, random_forest, svm
from src.tools import util

def merge_samples_labels(source_subfolder: str = None):
    root_dir = util.get_root_dir()
    samples_labels_dir = root_dir / 'data/samples_labels'
    samples_dir = root_dir / 'data/samples'
    output_file = samples_labels_dir / 'merged_samples_labels.csv'

    all_dfs = []

    # Determine which subfolders to process
    if source_subfolder:
        folders_to_process = [samples_labels_dir / f"{source_subfolder}_labels"]
    else:
        folders_to_process = [f for f in samples_labels_dir.iterdir() if f.is_dir()]

    for subfolder in folders_to_process:
        if not subfolder.exists():
            print(f"Subfolder does not exist: {subfolder}")
            continue

        # Corresponding original data folder in samples
        sample_name = subfolder.name
        if sample_name.endswith('_labels'):
            sample_name = sample_name.replace('_labels', '')
        source_folder_path = samples_dir / sample_name

        # Iterate through CSV files
        for csv_file in subfolder.glob('*.csv'):
            df = pd.read_csv(csv_file)
            # Store the path to the samples folder
            df['source_folder'] = str(source_folder_path)
            all_dfs.append(df)

    # Merge all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Merged CSV file generated: {output_file}")
    else:
        print("No CSV files found.")


def load_csv_data(patch_size=16):
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    csv_file = root_dir / 'data/samples_labels/merged_samples_labels.csv'

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_file}")

    df = pd.read_csv(csv_file)

    img_paths = [str(Path(row['source_folder']) / row['filename']) for _, row in df.iterrows()]
    y = df['y_thresh1'].tolist()

    return img_paths, y


def train_models(patch_size=32):
    # Merge CSV files and load data
    merge_samples_labels()
    img_paths, y = load_csv_data(patch_size)

    # Train models
    decision_tree_model = decision_tree.train_tree(img_paths, y, patch_size)
    random_forest_model = random_forest.train_random_forest(img_paths, y, patch_size)
    svm_model = svm.train_svm(img_paths, y, patch_size)
    cnn_model = cnn.train_cnn(img_paths, y, patch_size)

    # model_save folder in the current directory
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent / 'model_save'
    model_dir.mkdir(exist_ok=True)

    # Save three models separately
    joblib.dump(decision_tree_model, model_dir / 'decision_tree_model.joblib')
    joblib.dump(random_forest_model, model_dir / 'random_forest_model.joblib')
    joblib.dump(svm_model, model_dir / 'svm_model.joblib')
    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'num_classes': cnn_model.num_classes,  # 或你自己传入的类别数
        'patch_size': patch_size
    }, model_dir / 'cnn_model.pth')

    print(f"Models saved to: {model_dir}")


if __name__ == "__main__":
    train_models(patch_size=32)
