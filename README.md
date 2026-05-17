# EarthScan AI: Building Damage Assessment from Pre/Post Satellite Images

EarthScan AI is an Artificial Intelligence Lab final project for building damage assessment using paired pre-disaster and post-disaster satellite images. The system predicts one of four damage classes and provides a browser-based interface for uploading image pairs, viewing CNN and CNN+KNN predictions, confidence values, evaluation charts, and a supporting visual-change heatmap.

## Project Summary

The project uses the xBD / xView2 disaster dataset to classify satellite image pairs into:

- No damage
- Minor damage
- Major damage
- Destroyed

The trained backend uses two approaches:

- **CNN**: a Siamese MobileNetV2-based convolutional neural network that processes the pre-disaster and post-disaster images together.
- **CNN+KNN**: CNN feature extraction followed by a K-Nearest Neighbors classifier.

The final web application is built with Flask, HTML, CSS, and JavaScript. Users upload one pre-disaster image and its matching post-disaster image, then receive model predictions and confidence scores.

## Main Features

- Functional desktop web interface.
- Python Flask backend.
- Real trained CNN and CNN+KNN models.
- Pre/post image upload workflow.
- Correct preprocessing pipeline: resize to `256 x 256`, normalize to `[0, 1]`, combine as `256 x 256 x 6`.
- Prediction confidence bars and comparison chart.
- Visual-change heatmap for supporting interpretation.
- Matplotlib evaluation visualizations:
  - 5-run mean and standard deviation chart.
  - Training accuracy/loss curve.
  - CNN confusion matrix.
  - CNN+KNN confusion matrix.
- Final 5-run evaluation with Accuracy, Precision, Recall, and F1-score.

## Dataset

Dataset source: **xBD / xView2 Challenge Dataset**.

Dataset links:

- Official xView2 dataset page: https://xview2.org/dataset
- Kaggle mirror used for convenient download: https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test/data

This project uses the labeled training split from xBD and creates paired pre/post samples from matching image IDs.

Final processed dataset used for training:

- Total labeled pre/post pairs: `2,799`
- Train split: `1,959`
- Validation split: `420`
- Test split: `420`
- Input shape: `256 x 256 x 6`
- Channels `0-2`: pre-disaster RGB image
- Channels `3-5`: post-disaster RGB image

Labels are taken from post-disaster JSON annotations. For image-level classification, the label is selected using the most severe building damage class found in the post-disaster tile.

### Runtime vs. Retraining Data Requirement

The full raw dataset is **not required** to run the web application if the trained model files are already present. The web app only needs:

- `frontend-AI-LAB/`
- `models_final_aug_256_15ep_b16/run_02/`
- final JSON/CSV metric files
- final visualization PNG files

The full dataset is required only when someone wants to reproduce preprocessing or train the models again. Because the dataset is very large, it should be downloaded separately from the dataset link and placed in the project root as:

```text
Full dataset/
```

The processed NumPy arrays in `processed_balanced_pairs_256/` are also not required for normal app prediction. They are useful for retraining/evaluation, but they are too large for normal GitHub upload.

### Large Dataset and GitHub Upload Solution

GitHub regular upload has a 100 MB limit per file, and this project has large files:

- Raw xBD/xView2 dataset: many GB.
- `processed_balanced_pairs_256/`: around several GB because it contains `.npy` arrays.
- `cnn_model.keras`: larger than 100 MB.

Because of this, the correct GitHub solution is:

1. Keep the GitHub repository for source code, README, report, scripts, frontend, backend, and small visualization files.
2. Do **not** upload `Full dataset/` or `processed_balanced_pairs_256/` directly to GitHub.
3. Provide dataset download links in the README and report.
4. Provide preprocessing scripts so anyone can regenerate `processed_balanced_pairs_256/` from the downloaded raw dataset.
5. For faster retraining without preprocessing, optionally upload `processed_balanced_pairs_256/` as a zipped artifact to Google Drive, OneDrive, Kaggle Dataset, or Hugging Face Datasets, then place that download link here.

Recommended external artifact links to add before final submission:

```text
Raw dataset link:
https://xview2.org/dataset

Kaggle mirror:
https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test/data

Processed balanced dataset folder and Trained Model link:
[https://drive.google.com/drive/folders/1jO6sM5ITwhpdJQgm1GqR2TbwZUswZLKB?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto]


```

If someone downloads the project from GitHub and only wants to run the web demo, they do **not** need `processed_balanced_pairs_256/`. They only need the final trained model artifacts. If they want to retrain/evaluate exactly from saved arrays, then they should download `processed_balanced_pairs_256/` from the external artifact link and place it in the project root.

## Final Evaluation Results

Final training configuration:

- Runs: `5`
- Epochs per run: `10`
- Batch size: `16`
- Augmentation: enabled for training only
- Validation augmentation: disabled
- Test augmentation: disabled

Final 5-run mean +/- standard deviation:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN | 67.10% +/- 1.73% | 67.81% +/- 3.10% | 67.10% +/- 1.73% | 66.51% +/- 2.48% |
| CNN+KNN | 69.14% +/- 2.84% | 66.57% +/- 3.43% | 69.14% +/- 2.84% | 67.10% +/- 3.07% |

Best backend model selected for the web app:

```text
models_final_aug_256_15ep_b16/run_02
```

Run 02 was selected because it achieved the strongest CNN+KNN accuracy among the final runs.

## Folder Structure

Important project folders and files:

```text
Dataset/
├── frontend-AI-LAB/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       ├── js/
│       │   └── app.js
│       ├── visualizations/
│       │   ├── five_run_mean_std.png
│       │   ├── run02_training_accuracy_loss.png
│       │   ├── run02_cnn_confusion_matrix.png
│       │   └── run02_cnn_knn_confusion_matrix.png
│       └── bg.jpg
├── models_final_aug_256_15ep_b16/
│   ├── five_run_evaluation_summary.json
│   ├── five_run_metrics.csv
│   └── run_02/
│       ├── cnn_model.keras
│       ├── cnn_feature_extractor.keras
│       ├── cnn_knn.joblib
│       ├── history.csv
│       └── metrics.json
├── processed_balanced_pairs_256/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   ├── y_test.npy
│   └── split/preprocessing metadata files
├── viz_final_aug_256_15ep_b16/
│   ├── five_run_mean_std.png
│   └── run_01 ... run_05 evaluation charts
├── train_evaluate_models.py
├── preprocess_balanced_pairs.py
├── paired_augmentation.py
├── verify_data_split.py
├── requirements.txt
└── README.md
```

Large local-only folders:

```text
venv/
Full dataset/
processed_balanced_pairs_256/
__pycache__/
old experiment folders
```

These should usually not be committed to GitHub.

## Setup Instructions

### 1. Clone or open the project folder

```powershell
cd D:\Dataset\Dataset
```

If you are getting the project from GitHub, first download the model artifacts from the external model link and place them in:

```text
models_final_aug_256_15ep_b16/run_02/
```

Required model files:

```text
cnn_model.keras
cnn_feature_extractor.keras
cnn_knn.joblib
```

To retrain from scratch, also download the xBD/xView2 dataset and place it as `Full dataset/` in the project root. For prediction-only demo, the full dataset is not required.

For retraining/evaluation without repeating raw preprocessing, download the processed balanced dataset artifact and place it as:

```text
processed_balanced_pairs_256/
```

This processed folder is the main training/evaluation input, but it should be stored externally because it is too large for regular GitHub upload.

### 2. Create and activate a virtual environment

If a virtual environment already exists:

```powershell
..\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
..\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the application

```powershell
python frontend-AI-LAB\app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

## How to Use the Web App

1. Open the Upload tab.
2. Upload a pre-disaster satellite image.
3. Upload the matching post-disaster satellite image.
4. Select the model option:
   - Both models
   - CNN only
   - CNN+KNN only
5. Click **Analyze Damage**.
6. View:
   - Pre/post previews
   - Visual-change heatmap
   - CNN prediction
   - CNN+KNN prediction
   - Confidence chart
   - Evaluation visualizations in the About tab

Use matching image IDs. Example:

```text
hurricane-michael_00000004_pre_disaster.png
hurricane-michael_00000004_post_disaster.png
```

Matching IDs are important because the model compares the same location before and after the disaster.

## Heatmap Explanation

The heatmap is a visual-change map, not pixel-level damage segmentation.

Color meaning:

- Red / yellow: high visual change, possible damaged area.
- Green / cyan: moderate visual change.
- Blue / purple: low visual change.

The final damage class is predicted by the trained CNN and CNN+KNN models. The heatmap is included only as a supporting visualization.

## Training and Evaluation Commands

Final 5-run command used for the project:

```powershell
python train_evaluate_models.py --processed-dir processed_balanced_pairs_256 --models-dir models_final_aug_256_15ep_b16 --visualizations-dir viz_final_aug_256_15ep_b16 --runs 5 --epochs 10 --batch-size 16
```

Verify split:

```powershell
python verify_data_split.py --processed-dir processed_balanced_pairs_256
```

Verify paired augmentation:

```powershell
python paired_augmentation.py --processed-dir processed_balanced_pairs_256 --output-dir augmentation_checks_256
```

## GitHub Notes

The trained `cnn_model.keras` file and the processed dataset arrays are too large for regular GitHub upload. GitHub blocks regular files above 100 MB, so use one of these options:

1. Use Git LFS for model files only if Git LFS storage is available.
2. Upload model files and `processed_balanced_pairs_256/` to Google Drive / OneDrive / Kaggle / Hugging Face and provide links in the README.
3. Keep the GitHub repository code-focused and include setup instructions for downloading external artifacts.

Recommended final approach for this project:

- GitHub: source code, frontend/backend, scripts, README, report, requirements, small images, and final Matplotlib visualizations.
- External dataset link: raw xBD/xView2 dataset.
- External artifact link: optional `processed_balanced_pairs_256/` zipped folder.
- External model link: final `models_final_aug_256_15ep_b16/run_02/` model files.

Recommended files for GitHub:

- Source code scripts.
- `frontend-AI-LAB/`
- Final evaluation JSON/CSV files.
- Final visualization PNG files.
- `requirements.txt`
- `README.md`
- Report PDF.
- Optional small sample pre/post images for demo.

Do not upload:

- `venv/`
- Raw full dataset folder.
- `processed_balanced_pairs_256/`
- `__pycache__/`
- Temporary experiment folders unless needed for evidence.

## GitHub and Google Drive Upload Guide

Use this structure for final submission.

### 1. Upload These Files/Folders to GitHub

Upload the code and small project files:

```text
frontend-AI-LAB/
augmentation_checks/
dataset_work/
viz_final_aug_256_15ep_b16/
create_balanced_pairs.py
preprocess_balanced_pairs.py
paired_augmentation.py
train_evaluate_models.py
verify_balanced_dataset.py
verify_data_split.py
requirements.txt
README.md
AI_Lab_Final_Report_IEEE.tex
.gitignore
```

The `viz_final_aug_256_15ep_b16/` folder is safe to upload because it contains small final result graphs. These graphs are useful for the report and viva.

### 2. Do Not Upload These Folders to GitHub

Do not upload these directly to GitHub because they are too large:

```text
Full dataset/
processed_balanced_pairs_256/
models_final_aug_256_15ep_b16/
venv/
__pycache__/
```

### 3. Upload These to Google Drive / OneDrive / Hugging Face

Create one folder on Google Drive named:

```text
EarthScan-AI-Artifacts
```

Inside it, upload:

```text
models_final_aug_256_15ep_b16/run_02/
processed_balanced_pairs_256/
```

Optional but recommended:

```text
models_final_aug_256_15ep_b16/five_run_evaluation_summary.json
models_final_aug_256_15ep_b16/five_run_metrics.csv
```

The `run_02/` folder is required for running the web app because it contains:

```text
cnn_model.keras
cnn_feature_extractor.keras
cnn_knn.joblib
metrics.json
history.csv
```

The `processed_balanced_pairs_256/` folder is required only for retraining or re-evaluation from already processed arrays. It is not required for normal prediction/demo.

### 4. Add These Links Before Final Submission

Replace these placeholders with your real links:

```text
Processed balanced dataset folder:
[ADD GOOGLE DRIVE / HUGGING FACE LINK HERE]

Final trained model artifacts:
[ADD GOOGLE DRIVE / HUGGING FACE LINK HERE]
```

Raw dataset links:

```text
Official xView2 dataset:
https://xview2.org/dataset

Kaggle mirror:
https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test/data
```

### 5. How Someone Should Run the Project After Downloading from GitHub

For web app demo only:

1. Clone the GitHub repository.
2. Download the final model artifact folder from Google Drive.
3. Place it exactly as:

```text
models_final_aug_256_15ep_b16/run_02/
```

4. Install requirements:

```powershell
pip install -r requirements.txt
```

5. Run:

```powershell
python frontend-AI-LAB\app.py
```

6. Open:

```text
http://127.0.0.1:5000
```

For retraining/evaluation:

1. Clone the GitHub repository.
2. Download `processed_balanced_pairs_256/` from the artifact link.
3. Place it in the project root:

```text
processed_balanced_pairs_256/
```

4. Run the final training command:

```powershell
python train_evaluate_models.py --processed-dir processed_balanced_pairs_256 --models-dir models_final_aug_256_15ep_b16 --visualizations-dir viz_final_aug_256_15ep_b16 --runs 5 --epochs 10 --batch-size 16
```

For full reproduction from raw dataset:

1. Download the full xBD/xView2 dataset from the official or Kaggle link.
2. Place it in the project root as:

```text
Full dataset/
```

3. Run preprocessing scripts to regenerate `processed_balanced_pairs_256/`.
4. Run training/evaluation.

## Project Status

Completed:

- Dataset preprocessing
- Training and augmentation pipeline
- CNN and CNN+KNN models
- 5-run evaluation
- Flask backend
- Web frontend
- Matplotlib visualizations
- Visual-change heatmap

Remaining submission tasks:

- Add final report PDF.
- Add annotated screenshots.
- Upload repository to GitHub.
- Add final GitHub link to the report.
