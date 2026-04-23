# ImageFocusDetection

Language: [English](README.md) | [中文](README_CN.md) | [Deutsch](README_DE.md)

A patch-based image focus region detection and sharpness classification project.  
The project splits full images into fixed-size patches, automatically labels each patch, trains multiple models (Decision Tree / Random Forest / SVM / CNN), and projects prediction results back onto original images to generate visual heatmaps.

## 1. Project Features

- Raw image standardization and preprocessing:
  - Automatically reads training/validation raw images and converts them to grayscale.
  - Automatically aligns image dimensions to be divisible by `patch_size`, ensuring **stable grid segmentation**.
  - Records mapping metadata between original and normalized images (`samples_info.csv` / `valid_samples_info.csv`).
- Patch segmentation and grid representation:
  - Splits each image into non-overlapping grids with fixed patch size.
  - Patch filenames explicitly include `row/col` coordinates for projection back to original images.
- Automatic pseudo-label generation (no manual patch-level annotation needed):
  - Fuses Laplacian, Sobel, and FFT sharpness cues to compute `total_score`.
  - Supports both binary and tri-class labeling modes, and marks low-texture patches as `-1` (filtered during training).
- Multi-model training and unified experiment configuration:
  - Traditional ML: Decision Tree, Random Forest, SVM (with Nystroem kernel approximation).
  - Deep learning: lightweight CNN (supports **class-imbalance weighting**, adaptive device selection, **LMDB cache**).
  - Unified control of training parameters, model switches, and **Sweep (single-parameter scan)** via `Main.py`.
- Evaluation and interpretable visualization:
  - Outputs **Accuracy**, **Classification Report**, and **Confusion Matrix** on validation set.
  - Generates per-model prediction overlays (pred overlay) for intuitive focus-region localization.
  - During preprocessing, also generates label overlays, score heatmaps, and per-sample PCA 2D/3D distribution plots.
- Log parsing, result packaging, and auto plotting:
  - Automatically collects structured logs and complete logs into one run directory.
  - Parses logs into per-model CSV files for parameter-metric comparison.
  - Automatically generates Time/Evaluate/Loss charts for cross-model and cross-parameter comparisons.

## 2. Project Structure

```text
ImageFocusDetection/
├─ data/
│  ├─ raw/
│  │  ├─ train_img/                 # training raw images
│  │  └─ valid_img/                 # validation raw images
│  ├─ normalized/                   # normalized training images + samples_info.csv
│  ├─ valid_normalized/             # normalized validation images + valid_samples_info.csv
│  ├─ samples/                      # training patches
│  ├─ valid_samples/                # validation patches
│  ├─ samples_labels/               # training patch label CSVs
│  └─ valid_samples_labels/         # validation patch label CSVs
├─ logs/                            # run logs and packaged outputs
├─ src/
│  ├─ Main.py                       # unified entrypoint (recommended)
│  ├─ preprocessing/                # normalization, segmentation, auto-labeling, visualization
│  ├─ training/                     # training orchestration and model saving
│  ├─ evaluate/                     # validation evaluation and prediction visualization
│  ├─ models/
│  │  ├─ ml/                        # Decision Tree / Random Forest / SVM
│  │  └─ dl/                        # CNN
│  └─ tools/                        # log parsing, plotting, PCA, utility functions
├─ requirements.txt
└─ README.md
```

## 3. Environment Requirements

- Python: recommended `3.10+`
- OS: Windows / Linux / macOS (path examples are mainly Windows-style)
- Hardware (recommended for CNN training):
  - Prefer **NVIDIA GPU** with **CUDA** acceleration.
  - If using **Apple Silicon** (M-series), **Metal (PyTorch MPS)** acceleration can be enabled for CNN training.
  - Without GPU, execution falls back to CPU automatically (training will be significantly slower).

Install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Data Preparation

Place training and validation raw images in the following directories (unique filenames recommended, supports `jpg/jpeg/png`):

```text
data/raw/train_img
data/raw/valid_img
```

Important: you must manually create `data/raw`, `data/raw/train_img`, and `data/raw/valid_img` in advance.  
If any of these three directories are missing, the pipeline will fail to run.

Raw image source citation (BibTeX):

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

Image download and original paper project link:  
`https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel`

Default reproducibility data selection:  
Use `train_l` and `val_l` under `dd_dp_dataset_png` after downloading [All images used for training/testing](https://ln2.sync.com/dl/c45358c50/view/default/10770664840008?sync_id=0#r7kpybwk-xw8hhszh-qkj249ap-y8k2344d).  
Other data sources can also be used; this is only for easier reproduction.

### 4.1 Detailed `data/` Directory Description

```text
data/
├─ raw/
│  ├─ train_img/                           # input training raw images (provided by user)
│  └─ valid_img/                           # input validation raw images (provided by user)
├─ normalized/
│  ├─ sample1.png ...                      # normalized training images (grayscale + size alignment)
│  └─ samples_info.csv                     # training normalization mapping info
├─ valid_normalized/
│  ├─ valid_sample1.png ...                # normalized validation images
│  └─ valid_samples_info.csv               # validation normalization mapping info
├─ samples/
│  ├─ sample1/
│  │  ├─ sample1_0_0.png                   # patch file with row/col naming
│  │  └─ ...
│  └─ sampleN/
├─ valid_samples/
│  ├─ valid_sample1/
│  │  ├─ valid_sample1_0_0.png
│  │  └─ ...
│  └─ valid_sampleN/
├─ samples_labels/
│  ├─ sample1_labels/
│  │  ├─ sample1.csv                       # patch labels and scores
│  │  ├─ sample1_label_overlay.png         # label overlay
│  │  ├─ sample1_score_overlay.png         # score heatmap overlay
│  │  ├─ sample1_pca_2d_distribution.png
│  │  └─ sample1_pca_3d_distribution.png
│  ├─ sample2_labels/
│  └─ merged_samples_labels.csv            # merged training table (auto-generated before training)
└─ valid_samples_labels/
   ├─ valid_sample1_labels/
   │  ├─ valid_sample1.csv
   │  ├─ valid_sample1_label_overlay.png
   │  ├─ valid_sample1_score_overlay.png
   │  ├─ valid_sample1_pca_2d_distribution.png
   │  └─ valid_sample1_pca_3d_distribution.png
   ├─ valid_sample2_labels/
   └─ merged_valid_samples_labels.csv      # merged validation table (auto-generated before evaluation)
```

### 4.2 Key CSV Fields

`normalized/samples_info.csv` and `valid_normalized/valid_samples_info.csv`:

- `filename`: normalized filename (e.g., `sample1.png`)
- `original_size`: original image size (`WxH`)
- `current_size`: normalized size (divisible by `patch_size`)
- `original_filename`: original filename (from `data/raw/...`)

`samples_labels/*/*.csv` and `valid_samples_labels/*/*.csv`:

- `filename`: patch filename (contains grid coordinates)
- `lap_score`: Laplacian score
- `sobel_score`: Sobel score
- `fft_score`: FFT high-frequency ratio score
- `total_score`: fused focus score
- `label`: label (`-1/0/1/2`, see 6.4)

### 4.3 Data Flow (From Input to Training)

1. Raw images in `data/raw/*` -> `normalize_raw.py` -> `normalized/valid_normalized`
2. Normalized images -> `segment_nor_img.py` -> `samples/valid_samples`
3. Patches -> `label_patches.py` -> `samples_labels/valid_samples_labels`
4. Merge label CSVs -> `merged_samples_labels.csv` / `merged_valid_samples_labels.csv`
5. Training reads `merged_samples_labels.csv`, evaluation reads `merged_valid_samples_labels.csv`

## 5. How to Run

Run from project root:

```bash
python src/Main.py
```

Default `Main.py` flow:

1. Preprocess (normalize -> patch segmentation -> auto-labeling -> label visualization)
2. Train (models enabled by switches)
3. Evaluate (metrics + prediction visualizations on validation set)
4. Package (logs to CSV, auto plotting, consolidated outputs)

## 6. Configuration (`src/Main.py`)

### 6.1 Experiment Parameters `get_experiment_config()`

- `training.patch_size`: patch edge length, common values `16/32/64/128`.
- `training.sharp_threshold` / `training.blur_threshold`: **label thresholds** (0~1).
- `training.PCA_components`: PCA parameter, `-1` means PCA disabled.
- `training.sample_percentage`: training sample ratio (0~100].
- `models.*`: per-model hyperparameters (DT/RF/SVM/CNN).

### 6.1.1 Detailed Model Parameters (`models`)

#### A. Decision Tree (`models.decision_tree`)

- `max_depth`: maximum tree depth. Larger values fit training data more but may overfit.
- `min_samples_split`: minimum samples required to split a node. Larger values reduce overfitting.
- `min_samples_leaf`: minimum samples in leaf nodes. Larger values make boundaries smoother.
- `class_weight`: class weighting (e.g., `balanced`). Recommended for imbalanced classes.
- `random_state`: random seed for reproducibility.

#### B. Random Forest (`models.random_forest`)

- `n_estimators`: number of trees. More trees are usually more stable but slower.
- `max_depth`: max depth per tree, controls model complexity.
- `random_state`: random seed.
- `class_weight`: class weighting (commonly `balanced_subsample`).
- `n_jobs`: parallel worker count, `-1` means use as many CPU cores as possible.

#### C. SVM (`models.svm`, implemented as `StandardScaler + Nystroem + LinearSVC`)

- `nystroem_components`: kernel approximation dimension.
- `nystroem_kernel`: kernel type (`rbf/cosine/poly/sigmoid`).
- `nystroem_gamma`: kernel parameter (mainly affects `rbf/poly/sigmoid`).
- `random_state`: random seed (for Nystroem sampling).
- `svc_c`: penalty factor C.
- `class_weight`: class weighting (recommended `balanced` for imbalance).
- `max_iter`: maximum iterations.

#### D. CNN (`models.cnn`)

- `epochs`: number of epochs.
- `batch_base`: base batch size; actual batch size adjusts by device.
- `seed`: random seed.
- `learning_rate`: Adam learning rate.
- `use_weighted_sampler`: whether to enable weighted sampler (`auto/true/false`).
- `sampler_weight_power`: weighted-sampler exponent; larger values emphasize minority class more.
- `loss_weight_power`: class-loss weight exponent; larger values emphasize minority class more.
- `build_lmdb_if_missing`: whether to build **LMDB cache** (currently rebuilt by implementation).
- `assume_fixed_size`: whether patch size must strictly equal `patch_size`.
- `noise_std` (optional): Gaussian noise strength for augmentation.

### 6.1.2 CNN Training Implementation Notes

- Network depth adapts with `patch_size`:
  - `<=16`: 1 conv block
  - `<=64`: 2 conv blocks
  - `>64`: 3 conv blocks
- Loss uses `CrossEntropyLoss` with class weights built from class frequencies.
- Device priority is `CUDA > MPS > CPU`; AMP, `batch_size`, and `num_workers` are auto-set.
- Specifically:
  - `CUDA` maps to NVIDIA GPU acceleration (recommended).
  - `MPS` maps to Apple Silicon Metal acceleration.
- If LMDB is enabled, patch cache is stored at `data/samples_labels/patches_ps<patch_size>.lmdb`.

### 6.2 Control Switches `get_control_config()`

- `pipeline.preprocess`: enable/disable preprocessing.
- `pipeline.train_evaluate`: enable/disable train + evaluate.
- `models.decision_tree/random_forest/svm/cnn`: per-model enable switches.

### 6.3 Sweep (Parameter Scan)

Setting a parameter to a list enables **Sweep** (e.g., `patch_size: [16, 32, 64]`).  
Only **one** list parameter is allowed; multiple list parameters will raise an error.

Important: each training run clears `src/training/model_save/` before saving models.  
So in sweep mode, **only the model weights from the final run are kept**; earlier ones are overwritten.

### 6.4 Label Mode

- Tri-class mode: `sharp_threshold > blur_threshold`
  - `1`: sharp
  - `2`: mid-band
  - `0`: blur
  - `-1`: textureless (filtered in training)
- Binary mode: `sharp_threshold == blur_threshold`
  - `1`: sharp
  - `0`: blur
  - `-1`: textureless (filtered)

## 7. Output Description

### 7.1 Model Weights

Saved after training in:

```text
src/training/model_save/
```

Typical files:

- `decision_tree_model.joblib` / `decision_tree_pca.joblib`
- `random_forest_model.joblib` / `random_forest_pca.joblib`
- `svm_model.joblib` / `svm_pca.joblib`
- `cnn_model.pth`

Note: since the training entrypoint recreates `model_save` each run, in sweep mode these files correspond to the **final sweep value** only.

### 7.2 Logs and Packaged Output Directory

Each run generates a packaged directory under `logs/`, common naming patterns:

- `NM_<timestamp>`: normal single run
- `TR_<param>_<timestamp>` / `DT_...` / `RF_...` / `SVM_...` / `CNN_...`: sweep run

Typical structure:

```text
logs/
└─ NM_2026_0423_222346/                      # full packaged output of one run (example)
   ├─ 20260423_2222.log                      # structured log (print_and_save / save)
   ├─ 20260423_2222_complete.log             # complete stdout/stderr log
   ├─ Decision_Tree_<control>.csv            # DT metrics summary
   ├─ Random_Forest_<control>.csv            # RF metrics summary
   ├─ SVM_<control>.csv                      # SVM metrics summary
   ├─ CNN_<control>.csv                      # CNN metrics summary
   ├─ plots/                                 # auto-generated plots from CSV
   │  ├─ <Model>_<control>_Time.png
   │  ├─ <Model>_<control>_Evaluate.png
   │  └─ CNN_<control>_Loss_*.png
   └─ predict_images/                        # validation prediction overlays
      └─ valid_sample*_predict_images[_tag]/
         └─ valid_sample*_pred_overlay_<DT|RF|SVM|CNN>.png
```

Where:

- `<control>` is the **Sweep control variable** (usually `normal` if no sweep).
- `[_tag]` appears only in **Sweep** mode to distinguish values.
- `logs/_evaluate_cache` is a temporary evaluation folder and is removed after packaging.

Directory typically contains:

- per-model summary CSV files (parsed from logs)
- `plots/` charts (time metrics, evaluation metrics, CNN loss curves)
- `predict_images/` prediction visualizations (grouped by sample)
- run log and complete log

### 7.3 Visualization Outputs

Preprocessing stage generates:

- label overlays (`*_label_overlay.png`)
- score heatmap overlays (`*_score_overlay.png`)
- PCA 2D/3D distribution plots (per sample)

Evaluation stage generates:

- validation prediction overlays (per model, e.g., `*_pred_overlay_DT.png`)

## 8. FAQ

### 8.1 Why are some folders deleted every preprocess run?

`Main.py` removes and rebuilds the following intermediate folders to ensure clean and reproducible runs:

- `data/samples`
- `data/samples_labels`
- `data/valid_samples`
- `data/valid_samples_labels`

### 8.2 How can I speed up execution?

- Enable only one model in `get_control_config()`.
- Reduce `sample_percentage` (e.g., 30~50).
- Reduce `patch_size` or the number of sweep values.
- Reduce CNN `epochs`.

### 8.3 Can I only plot or only parse logs?

Yes, use:

- `src/tools/log_tools.py`
- `src/tools/plot_csv.py`

### 8.4 Why does `label=-1` appear?

`-1` indicates a **textureless** patch (low variance and low gradient), considered unreliable for labeling.  
Rows with `label=-1` are automatically filtered during train/eval CSV merging.

### 8.5 How to switch between binary and tri-class labeling?

- Binary: `sharp_threshold == blur_threshold`
- Tri-class: `sharp_threshold > blur_threshold`

If `blur_threshold > sharp_threshold`, the program raises an error and stops.

### 8.6 Why is only one model set left after sweep?

Training clears `src/training/model_save/` before each run.  
So after sweep, only the last run’s model files remain.

### 8.7 Why is a model skipped in evaluation (missing model files)?

Possible causes:

- model disabled in `get_control_config().models`
- latest training did not finish successfully
- model files overwritten by later sweep runs

### 8.8 Why are `predict_images` grouped by sample instead of by model?

Current implementation groups by validation sample; each sample folder contains overlays from all models.  
This makes side-by-side comparison easier on the same image.

### 8.9 What is `logs/_evaluate_cache`?

It is a temporary cache for writing prediction overlays during evaluation.  
After packaging to `logs/<run_dir>/predict_images`, it is automatically cleaned.

### 8.10 Why is there no test-set pipeline?

Current main flow is fixed as `train + valid evaluate`.  
If needed, add a `test_img` pipeline similar to `data/raw/valid_img`.

### 8.11 How should I set `PCA_components`?

- `-1`: disable PCA (use original pixel features).
- `0~1` (e.g., `0.95`): keep components by **cumulative explained variance**.
- If set greater than or equal to original dimension, code skips PCA automatically.

### 8.12 SVM is slow or not converging. What should I do?

- reduce `nystroem_components`
- reduce `svc_c` moderately
- increase `max_iter`
- or reduce `sample_percentage` for faster iteration

### 8.13 CNN runs out of memory. What should I do?

- reduce `models.cnn.batch_base`
- reduce `patch_size`
- reduce `epochs`
- optionally run only CNN to avoid resource contention

### 8.14 Why do results vary slightly between repeated runs?

Random seeds are set, but backend/device differences (CPU/CUDA/MPS) and low-level parallel behavior can still introduce minor differences.  
For strict comparison, fix device, data, and all parameters.

## 9. Dependencies

See `requirements.txt`:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `opencv-python`
- `torch`
- `lmdb`
- `matplotlib`

## 10. Development Statement

Parts of this project were built with CodeX assistance, including but not limited to:

- large-scale regex parsing and structured log extraction (e.g., in `src/tools/log_tools.py`)
- unified log collection and packaging pipeline (e.g., dual-log stream in `src/tools/log.py` and archive flow in `src/Main.py`)
- strict data typing and intermediate-structure constraints, with a style close to Java / C / C++ explicit typing semantics

## 11. Copyright and Third-Party Notes (Consolidated)

- **CodeX**: CodeX is an OpenAI code-generation and coding-assistance tool.
- **Third-party open-source dependencies**: libraries such as `numpy`, `pandas`, `scikit-learn`, `joblib`, `opencv-python`, `torch`, `lmdb`, and `matplotlib` are owned by their respective communities/maintainers; please follow their licenses.
- **Datasets and image assets**: raw images under `data/raw/*` are owned by the original authors/providers; ensure legal authorization for use/distribution/publication.
- **Image source used in this project (BibTeX)**:
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
- **Method and terminology references**: algorithm/model names in this document and code (e.g., PCA, SVM, Random Forest, CNN, Nystroem) are standard academic/engineering terms; related papers and materials are copyrighted by original publishers/authors.
