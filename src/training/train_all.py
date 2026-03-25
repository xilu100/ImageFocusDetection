from pathlib import Path

import joblib
import pandas as pd

from src.models.ml import decision_tree

from pathlib import Path
import pandas as pd


def merge_samples_labels(source_subfolder: str = None):
    """
    将 samples_labels 下指定子文件夹或全部 CSV 文件合并到一个总 CSV 文件，
    输出到 samples_labels/merged_samples_labels.csv。

    参数:
        source_subfolder (str, optional): 指定的子文件夹名称，如 'sample1'。
                                           如果为 None，则合并全部子文件夹。
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]  # 根目录
    samples_labels_dir = root_dir / 'data/samples_labels'
    samples_dir = root_dir / 'data/samples'
    output_file = samples_labels_dir / 'merged_samples_labels.csv'

    all_dfs = []

    # 确定需要处理的子文件夹
    if source_subfolder:
        folders_to_process = [samples_labels_dir / f"{source_subfolder}_labels"]
    else:
        folders_to_process = [f for f in samples_labels_dir.iterdir() if f.is_dir()]

    for subfolder in folders_to_process:
        if not subfolder.exists():
            print(f"子文件夹不存在: {subfolder}")
            continue

        # 对应 samples 下的原始数据文件夹
        sample_name = subfolder.name
        if sample_name.endswith('_labels'):
            sample_name = sample_name.replace('_labels', '')
        source_folder_path = samples_dir / sample_name

        # 遍历 CSV 文件
        for csv_file in subfolder.glob('*.csv'):
            df = pd.read_csv(csv_file)
            # 存储 samples 下的路径
            df['source_folder'] = str(source_folder_path)
            all_dfs.append(df)

    # 合并所有 DataFrame
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"已生成总 CSV 文件: {output_file}")
    else:
        print("没有找到 CSV 文件。")

def load_csv_data(patch_size=16):
    """
    读取合并后的 CSV 文件，返回 img_paths 和 y。

    参数:
        patch_size (int): 图像 patch 大小，占位参数，可在后续扩展中使用。

    返回:
        img_paths (list[str]): 所有 source_folder 列表
        y (list[float]): 对应的 y_thresh2 值
    """
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    csv_file = root_dir / 'data/samples_labels/merged_samples_labels.csv'

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_file}")

    df = pd.read_csv(csv_file)

    img_paths = [str(Path(row['source_folder']) / row['filename']) for _, row in df.iterrows()]
    y = df['y_thresh2'].tolist()

    return img_paths, y
def train_models(patch_size=16):
    # 合并 CSV 并加载数据
    merge_samples_labels()
    img_paths, y = load_csv_data(patch_size)

    # 训练模型
    model = decision_tree.train_tree(img_paths, y, patch_size)

    # 当前文件夹下的 model_save 文件夹
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent / 'model_save'
    model_dir.mkdir(exist_ok=True)  # 如果文件夹不存在就创建

    # 模型保存路径
    model_file = model_dir / 'decision_tree_model.joblib'

    # 保存模型
    joblib.dump(model, model_file)
    print(f"模型已保存到: {model_file}")


if __name__ == "__main__":
    train_models(patch_size=16)