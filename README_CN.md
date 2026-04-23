# ImageFocusDetection

语言: [English](README.md) | [中文](README_CN.md) | [Deutsch](README_DE.md)

一个基于 Patch 的图像对焦区域检测与清晰度分类项目。  
项目将整图切成固定大小 patch，对每个 patch 自动打标签并训练多个模型（Decision Tree / Random Forest / SVM / CNN），再将预测结果回投到原图生成可视化热力图。

## 1. 项目功能

- 原图标准化预处理：
  - 自动读取训练/验证原图，统一转灰度。
  - 自动把图像尺寸对齐到可被 `patch_size` 整除，保证后续**网格切分稳定性**。
  - 记录原图与归一化图映射关系（`samples_info.csv` / `valid_samples_info.csv`）。
- Patch 切分与网格化表达：
  - 将每张图按固定 patch 大小切分为不重叠网格。
  - patch 命名显式携带 `row/col` 坐标，支持回投到原图。
- 自动伪标签生成（无需人工逐 patch 标注）：
  - 融合 Laplacian、Sobel、FFT 三类清晰度线索计算 `total_score`。
  - 支持二分类与三分类两种标签模式，且对低纹理 patch 标记 `-1` 并在训练时过滤。
- 多模型训练与统一实验配置：
  - 传统机器学习：Decision Tree、Random Forest、SVM（含 Nystroem 核近似）。
  - 深度学习：轻量 CNN（支持**类别不平衡加权**、设备自适应、**LMDB 缓存**）。
  - 通过 `Main.py` 统一配置训练参数、模型开关和 **Sweep（单参数扫描）**。
- 评估与可解释可视化：
  - 在验证集输出 **Accuracy**、**Classification Report**、**Confusion Matrix**。
  - 生成按模型区分的预测叠加图（pred overlay），直观看到聚焦区域定位效果。
  - 预处理阶段同步生成标签叠加图、分数热力图、样本级 PCA 2D/3D 分布图。
- 日志解析、结果打包与自动绘图：
  - 自动收集结构化日志与完整日志，归档到单次运行目录。
  - 将日志解析为模型级 CSV，沉淀可比较的参数-指标表。
  - 自动生成 Time/Evaluate/Loss 图表，便于横向比较不同参数与模型表现。

## 2. 项目结构

```text
ImageFocusDetection/
├─ data/
│  ├─ raw/
│  │  ├─ train_img/                 # 训练原图
│  │  └─ valid_img/                 # 验证原图
│  ├─ normalized/                   # 训练集归一化图 + samples_info.csv
│  ├─ valid_normalized/             # 验证集归一化图 + valid_samples_info.csv
│  ├─ samples/                      # 训练集 patch
│  ├─ valid_samples/                # 验证集 patch
│  ├─ samples_labels/               # 训练集 patch 标签 CSV
│  └─ valid_samples_labels/         # 验证集 patch 标签 CSV
├─ logs/                            # 运行日志与打包输出
├─ src/
│  ├─ Main.py                       # 统一入口（推荐从这里启动）
│  ├─ preprocessing/                # 归一化、切分、自动标注、可视化
│  ├─ training/                     # 训练调度与模型保存
│  ├─ evaluate/                     # 验证集评估与预测可视化输出
│  ├─ models/
│  │  ├─ ml/                        # 决策树 / 随机森林 / SVM
│  │  └─ dl/                        # CNN
│  └─ tools/                        # 日志解析、绘图、PCA、工具函数
├─ requirements.txt
└─ README.md
```

## 3. 环境要求

- Python：建议 `3.10+`
- 系统：Windows / Linux / macOS 均可（路径示例以 Windows 为主）
- 硬件（CNN 训练建议）：
  - 建议优先使用 **NVIDIA GPU** 并启用 **CUDA** 加速训练。
  - 若使用 **Apple Silicon**（M 系列芯片），可启用 **Metal（PyTorch MPS）** 加速 CNN 训练。
  - 无 GPU 时会自动回退到 CPU（训练速度会明显下降）。

安装依赖：

```bash
pip install -r requirements.txt
```

## 4. 数据准备

将训练与验证原图放到以下目录（文件名尽量唯一，支持 `jpg/jpeg/png`）：

```text
data/raw/train_img
data/raw/valid_img
```

重要说明：必须手动提前创建 `data/raw`、`data/raw/train_img`、`data/raw/valid_img`。  
这三个目录只要缺少任意一个，流程都将无法运行。

本项目使用的原始图片来源注明（BibTeX）：

```bibtex
@inproceedings{abuolaim2020defocus,
  title={Defocus deblurring using dual-pixel data},
  author={Abuolaim, Abdullah and Brown, Michael S},
  booktitle={European Conference on Computer Vision},
  pages={111--126},
  year={2020},
  organization={Springer}
}
```

图片下载及原论文项目链接：  
`https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel`

默认复现实验的数据选择说明：  
使用 [All images used for training/testing](https://ln2.sync.com/dl/c45358c50/view/default/10770664840008?sync_id=0#r7kpybwk-xw8hhszh-qkj249ap-y8k2344d) 下载后的 `dd_dp_dataset_png` 文件夹下的 `train_l` 文件夹和 `val_l` 文件夹。  
当然也可以替换为其他数据来源，这里仅为方便复现。

### 4.1 `data/` 目录详细说明

```text
data/
├─ raw/
│  ├─ train_img/                           # 输入训练原图（你需要提供）
│  └─ valid_img/                           # 输入验证原图（你需要提供）
├─ normalized/
│  ├─ sample1.png ...                      # 训练原图归一化后（灰度 + 尺寸对齐）
│  └─ samples_info.csv                     # 训练归一化映射信息
├─ valid_normalized/
│  ├─ valid_sample1.png ...                # 验证原图归一化后
│  └─ valid_samples_info.csv               # 验证归一化映射信息
├─ samples/
│  ├─ sample1/
│  │  ├─ sample1_0_0.png                   # patch 文件，命名含 row/col
│  │  └─ ...
│  └─ sampleN/
├─ valid_samples/
│  ├─ valid_sample1/
│  │  ├─ valid_sample1_0_0.png
│  │  └─ ...
│  └─ valid_sampleN/
├─ samples_labels/
│  ├─ sample1_labels/
│  │  ├─ sample1.csv                       # patch 标签与分数
│  │  ├─ sample1_label_overlay.png         # 标签叠加图
│  │  ├─ sample1_score_overlay.png         # 分数热力图叠加
│  │  ├─ sample1_pca_2d_distribution.png
│  │  └─ sample1_pca_3d_distribution.png
│  ├─ sample2_labels/
│  └─ merged_samples_labels.csv            # 训练总表（训练前自动生成）
└─ valid_samples_labels/
   ├─ valid_sample1_labels/
   │  ├─ valid_sample1.csv
   │  ├─ valid_sample1_label_overlay.png
   │  ├─ valid_sample1_score_overlay.png
   │  ├─ valid_sample1_pca_2d_distribution.png
   │  └─ valid_sample1_pca_3d_distribution.png
   ├─ valid_sample2_labels/
   └─ merged_valid_samples_labels.csv      # 验证总表（评估前自动生成）
```

### 4.2 关键 CSV 字段

`normalized/samples_info.csv` 与 `valid_normalized/valid_samples_info.csv`：

- `filename`：归一化后文件名（如 `sample1.png`）
- `original_size`：原图尺寸（`WxH`）
- `current_size`：归一化后尺寸（能被 `patch_size` 整除）
- `original_filename`：原始文件名（来自 `data/raw/...`）

`samples_labels/*/*.csv` 与 `valid_samples_labels/*/*.csv`：

- `filename`：patch 文件名（包含网格坐标）
- `lap_score`：Laplacian 分数
- `sobel_score`：Sobel 分数
- `fft_score`：FFT 高频比例分数
- `total_score`：融合后的聚焦分数
- `label`：标签（`-1/0/1/2`，含义见 6.4）

### 4.3 数据流转关系（从输入到训练）

1. `data/raw/*` 原图 -> `normalize_raw.py` -> `normalized/valid_normalized`
2. 归一化图 -> `segment_nor_img.py` -> `samples/valid_samples`
3. patch -> `label_patches.py` -> `samples_labels/valid_samples_labels`
4. 标签 CSV 合并 -> `merged_samples_labels.csv` / `merged_valid_samples_labels.csv`
5. 训练读取 `merged_samples_labels.csv`，评估读取 `merged_valid_samples_labels.csv`

## 5. 运行方式

在项目根目录执行：

```bash
python src/Main.py
```

`Main.py` 默认按以下流程执行：

1. 预处理（归一化 -> 切 patch -> 自动标注 -> 标签可视化）
2. 训练（按开关启用模型）
3. 评估（在验证集输出指标和预测可视化）
4. 打包（日志转 CSV、自动画图、汇总输出）

## 6. 配置说明（`src/Main.py`）

### 6.1 实验参数 `get_experiment_config()`

- `training.patch_size`：patch 边长，常用 `16/32/64/128`。
- `training.sharp_threshold` / `training.blur_threshold`：**标签阈值**（0~1）。
- `training.PCA_components`：PCA 维度参数，`-1` 表示关闭 PCA。
- `training.sample_percentage`：训练样本抽样比例（0~100]。
- `models.*`：各模型超参数（DT/RF/SVM/CNN）。

### 6.1.1 模型参数详细说明（`models`）

#### A. Decision Tree（`models.decision_tree`）

- `max_depth`：树最大深度。越大越容易拟合训练集，也更容易过拟合。
- `min_samples_split`：节点继续分裂所需最小样本数。增大可抑制过拟合。
- `min_samples_leaf`：叶子节点最少样本数。增大可让决策边界更平滑。
- `class_weight`：类别权重（如 `balanced`）。类别不平衡时建议开启。
- `random_state`：随机种子，保证结果可复现。

#### B. Random Forest（`models.random_forest`）

- `n_estimators`：树数量。更多树通常更稳但训练更慢。
- `max_depth`：每棵树最大深度。控制复杂度与过拟合风险。
- `random_state`：随机种子。
- `class_weight`：类别权重（常用 `balanced_subsample`）。
- `n_jobs`：并行线程数，`-1` 表示尽量使用所有 CPU。

#### C. SVM（`models.svm`，实现为 `StandardScaler + Nystroem + LinearSVC`）

- `nystroem_components`：核近似维度。越大表达能力越强，但更耗时耗内存。
- `nystroem_kernel`：核函数（`rbf/cosine/poly/sigmoid`）。
- `nystroem_gamma`：核参数（主要影响 `rbf/poly/sigmoid`）。
- `random_state`：随机种子（用于 Nystroem 采样）。
- `svc_c`：惩罚系数 C。大 C 倾向更低训练误差，小 C 泛化更稳。
- `class_weight`：类别权重（不平衡时建议 `balanced`）。
- `max_iter`：最大迭代次数，收敛困难时可适当增大。

#### D. CNN（`models.cnn`）

- `epochs`：训练轮数。
- `batch_base`：基础 batch size；实际 batch 会根据设备自动调整。
- `seed`：随机种子。
- `learning_rate`：Adam 学习率。
- `use_weighted_sampler`：是否启用加权采样（`auto/true/false`）。
- `sampler_weight_power`：加权采样权重指数，越大越偏向少数类。
- `loss_weight_power`：损失类别权重指数，越大越强调少数类。
- `build_lmdb_if_missing`：是否构建 **LMDB 缓存**（当前实现会重建）。
- `assume_fixed_size`：是否强制 patch 尺寸必须等于 `patch_size`。
- `noise_std`（可选）：训练时对输入 patch 加高斯噪声的数据增强强度。

### 6.1.2 CNN 训练实现要点（便于调参）

- 网络结构随 `patch_size` 自动变化：
  - `<=16`：1 个卷积块
  - `<=64`：2 个卷积块
  - `>64`：3 个卷积块
- 分类损失使用 `CrossEntropyLoss`，并根据类别频次自动构建 class weights。
- 设备自动选择 `CUDA > MPS > CPU`，并自动设置 AMP、`batch_size`、`num_workers`。
- 其中：
  - `CUDA` 对应 NVIDIA 显卡加速（推荐）。
  - `MPS` 对应 Apple Silicon 的 Metal 加速。
- 若启用 LMDB，会把 patch 缓存到 `data/samples_labels/patches_ps<patch_size>.lmdb`。

### 6.2 执行开关 `get_control_config()`

- `pipeline.preprocess`：是否执行预处理。
- `pipeline.train_evaluate`：是否执行训练 + 评估。
- `models.decision_tree/random_forest/svm/cnn`：按模型启停。

### 6.3 Sweep（参数扫描）

将某个参数改为列表即可触发 **Sweep**（例如 `patch_size: [16, 32, 64]`）。  
注意：当前版本只允许**一个参数**是列表；多个列表会报错。

另外需要注意：训练阶段每次都会清空 `src/training/model_save/` 再保存模型，  
因此在 sweep 中**只会保留最后一次 run 的模型权重**，前面 run 的权重会被覆盖。

### 6.4 标签模式说明

- 三分类模式：`sharp_threshold > blur_threshold`
  - `1`：清晰
  - `2`：中间带
  - `0`：模糊
  - `-1`：纹理不足（训练时会被过滤）
- 二分类模式：`sharp_threshold == blur_threshold`
  - `1`：清晰
  - `0`：模糊
  - `-1`：纹理不足（过滤）

## 7. 输出结果说明

### 7.1 模型权重

训练后保存在：

```text
src/training/model_save/
```

典型文件：

- `decision_tree_model.joblib` / `decision_tree_pca.joblib`
- `random_forest_model.joblib` / `random_forest_pca.joblib`
- `svm_model.joblib` / `svm_pca.joblib`
- `cnn_model.pth`

注意：由于训练入口会先删除并重建 `model_save`，若执行 sweep，以上文件最终对应的是**最后一次 sweep 取值**的结果。

### 7.2 日志与打包目录

每次运行会在 `logs/` 下生成一个打包目录，常见命名：

- `NM_<timestamp>`：普通单次运行
- `TR_<param>_<timestamp>` / `DT_...` / `RF_...` / `SVM_...` / `CNN_...`：参数扫描运行

典型结构如下：

```text
logs/
└─ NM_2026_0423_222346/                      # 一次完整运行的打包目录（示例）
   ├─ 20260423_2222.log                      # 结构化日志（print_and_save / save）
   ├─ 20260423_2222_complete.log             # 完整 stdout/stderr 日志
   ├─ Decision_Tree_<control>.csv            # 决策树指标汇总
   ├─ Random_Forest_<control>.csv            # 随机森林指标汇总
   ├─ SVM_<control>.csv                      # SVM 指标汇总
   ├─ CNN_<control>.csv                      # CNN 指标汇总
   ├─ plots/                                 # 基于 CSV 自动绘图
   │  ├─ <Model>_<control>_Time.png
   │  ├─ <Model>_<control>_Evaluate.png
   │  └─ CNN_<control>_Loss_*.png
   └─ predict_images/                        # 验证集预测叠加图
      └─ valid_sample*_predict_images[_tag]/
         └─ valid_sample*_pred_overlay_<DT|RF|SVM|CNN>.png
```

其中：

- `<control>` 是 **Sweep 控制变量名**（无 sweep 时一般为 `normal`）。
- `[_tag]` 仅在 **Sweep** 时出现，用于区分不同参数取值的预测结果目录。
- `logs/_evaluate_cache` 是评估阶段临时目录，打包完成后会自动清理。

目录内通常包含：

- 每个模型的汇总 CSV（从日志自动解析）
- `plots/` 图表（时间指标、评估指标、CNN loss 曲线）
- `predict_images/` 预测可视化结果（按样本目录汇总）
- 运行日志与完整日志

### 7.3 可视化结果

预处理阶段会生成：

- 标签叠加图（`*_label_overlay.png`）
- 分数热力图叠加（`*_score_overlay.png`）
- PCA 2D/3D 分布图（按样本）

评估阶段会生成：

- 验证集预测叠加图（按模型，如 `*_pred_overlay_DT.png`）

## 8. 常见问题

### 8.1 为什么每次预处理都会清空一些目录？

`Main.py` 的预处理阶段会删除并重建以下中间目录，确保流程干净可复现：

- `data/samples`
- `data/samples_labels`
- `data/valid_samples`
- `data/valid_samples_labels`

### 8.2 运行很慢怎么办？

- 先只开一个模型（在 `get_control_config()` 里关闭其余模型）。
- 降低 `sample_percentage`（如 30~50）。
- 减小 `patch_size` 或减少 sweep 取值数量。
- CNN 训练可减少 `epochs`。

### 8.3 只想画图/只想解析日志可以吗？

可以，相关脚本在：

- `src/tools/log_tools.py`
- `src/tools/plot_csv.py`

### 8.4 为什么会出现 `label=-1`？

`-1` 表示 patch **纹理不足**（低方差且低梯度），属于“不可可靠判别”区域。  
训练与评估合并 CSV 时会自动过滤掉 `label=-1`，这是预期行为。

### 8.5 二分类和三分类如何切换？

- 二分类：`sharp_threshold == blur_threshold`
- 三分类：`sharp_threshold > blur_threshold`

若设置为 `blur_threshold > sharp_threshold`，程序会直接报错并终止。

### 8.6 为什么 sweep 后只剩一份模型？

训练入口会先清空 `src/training/model_save/` 再保存新模型。  
因此 sweep 结束后，`model_save` 中只保留最后一次 run 的权重文件。

### 8.7 评估时提示某个模型被跳过（missing model files）怎么办？

说明对应模型权重文件不存在或被覆盖。请检查：

- 该模型是否在 `get_control_config().models` 中启用；
- 最近一次训练是否成功完成；
- sweep 是否已覆盖此前模型文件。

### 8.8 为什么 `predict_images` 里是按样本目录而不是按模型目录？

当前实现按“验证样本”聚合，每个样本目录中放置各模型叠加图。  
这样更方便比较同一张图在不同模型下的预测差异。

### 8.9 `logs/_evaluate_cache` 是什么？

这是评估阶段的临时缓存目录，用于先写预测叠加图。  
主流程打包到 `logs/<run_dir>/predict_images` 后会自动清理该缓存。

### 8.10 为什么没有测试集（test）流程？

当前主流程固定为 `train + valid evaluate`。  
如需 test 流程，可按 `data/raw/valid_img` 的处理方式扩展一套 `test_img` 管线。

### 8.11 `PCA_components` 该怎么设置？

- `-1`：关闭 PCA（使用原始像素特征）。
- `0~1`（如 `0.95`）：按**累计解释方差**保留主成分。
- 若设置值大于等于原始维度，代码会自动跳过 PCA。

### 8.12 SVM 训练慢或不收敛怎么办？

- 先降低 `nystroem_components`；
- 适当减小 `svc_c`；
- 增大 `max_iter`；
- 或减少 `sample_percentage` 做快速迭代。

### 8.13 CNN 显存不够怎么办？

- 减小 `models.cnn.batch_base`；
- 减小 `patch_size`；
- 减少 `epochs`；
- 必要时只保留 CNN 单模型运行，避免并行资源竞争。

### 8.14 为什么同样参数多次运行结果略有波动？

项目已设置随机种子，但不同硬件后端（CPU/CUDA/MPS）和底层并行实现仍可能引入微小差异。  
做严格对比时建议固定设备、固定数据与固定参数。

## 9. 依赖列表

见 `requirements.txt`：

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `opencv-python`
- `torch`
- `lmdb`
- `matplotlib`

## 10. 开发声明

本项目的部分代码构建使用了 CodeX 辅助完成，主要包括但不限于：

- 大规模正则表达式解析与日志结构化抽取（如 `src/tools/log_tools.py` 中的日志解析逻辑）。
- 输出日志的统一收集与打包流程（如 `src/tools/log.py` 的双日志流、`src/Main.py` 的日志归档与汇总）。
- 各类核心参数与中间数据结构采用严格类型约束，整体风格接近 Java / C / C++ 的数据类型定义模式（如显式 `int/float/bool` 语义、结构化类型与边界检查）。

## 11. 版权与第三方说明（统一标注）

- **CodeX**：CodeX 是 OpenAI 提供的代码生成与编程辅助工具。  
- **第三方开源依赖**：本项目使用的 `numpy`、`pandas`、`scikit-learn`、`joblib`、`opencv-python`、`torch`、`lmdb`、`matplotlib` 等库，其版权与许可证归各自社区/维护方所有，使用时请遵循对应许可证。  
- **数据集与图片素材**：`data/raw/*` 中的原始图像版权归其原作者或数据集提供方所有；使用、分发与发表请确保已获得合法授权。  
- **本项目图片来源（BibTeX）**：
  ```bibtex
  @inproceedings{abuolaim2020defocus,
    title={Defocus deblurring using dual-pixel data},
    author={Abuolaim, Abdullah and Brown, Michael S},
    booktitle={European Conference on Computer Vision},
    pages={111--126},
    year={2020},
    organization={Springer}
  }
  ```  
- **方法与术语引用**：文档和代码中出现的算法名/模型名（如 PCA、SVM、Random Forest、CNN、Nystroem）为学术与工程通用方法名称，相关论文与资料版权归原出版方及作者所有。  
