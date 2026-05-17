# EarthScan AI: Building Damage Assessment from Pre/Post Satellite Images

EarthScan AI is a desktop web-based artificial intelligence application for classifying building damage from paired pre-disaster and post-disaster satellite images. The project was developed for **AIL 201 - Artificial Intelligence Lab** and uses the xBD/xView2 disaster dataset.

The user uploads one pre-disaster image and its matching post-disaster image. The backend preprocesses both images, runs trained models, and returns the predicted damage class with confidence values, charts, and a visual-change heatmap.

## Damage Classes

The system predicts one of four image-level classes:

- No damage
- Minor damage
- Major damage
- Destroyed

## Main Features

- Functional desktop web interface.
- Python Flask backend.
- Trained CNN model.
- Hybrid CNN+KNN model.
- Pre/post satellite image upload.
- Image preprocessing: resize to `256 x 256`, normalize to `[0, 1]`, combine into `256 x 256 x 6`.
- Confidence bars and model comparison chart.
- Visual-change heatmap for supporting interpretation.
- Matplotlib evaluation visualizations.
- Final 5-run evaluation with mean and standard deviation.

## Models Used

### CNN

The CNN model receives a combined six-channel input:

```text
Channels 0-2 = pre-disaster RGB image
Channels 3-5 = post-disaster RGB image
```

It learns visual differences between the pre-disaster and post-disaster images and predicts the damage class.

### CNN+KNN

The CNN+KNN model is a hybrid approach:

1. A CNN feature extractor converts the image pair into feature vectors.
2. A K-Nearest Neighbors classifier predicts the class using those extracted features.

This approach compares deep visual features using distance-based classification.

## Dataset

Dataset source: **xBD / xView2 Challenge Dataset**.

Dataset links:

- Official xView2 dataset page: https://xview2.org/dataset
- Kaggle mirror: https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test/data

This project uses labeled pre-disaster and post-disaster image pairs from the xBD/xView2 dataset. Labels are taken from post-disaster JSON annotations. For image-level classification, the selected label is the most severe building damage class found in the post-disaster tile.

### Dataset Counts

Complete labeled paired dataset:

```text
Pre-disaster images:  2,799
Post-disaster images: 2,799
Complete pairs:       2,799
Total paired images:  5,598
```

Class distribution in the full paired dataset:

| Class | Pairs |
|---|---:|
| No damage | 1,709 |
| Minor damage | 418 |
| Major damage | 429 |
| Destroyed | 243 |
| **Total** | **2,799** |

The dataset is imbalanced, so the final training uses class weights and training-time augmentation.

### Balanced Subset

A balanced subset was also created for class-balance analysis:

```text
dataset_work/balanced_pairs_index.csv
```

Balanced subset distribution:

| Class | Pairs |
|---|---:|
| No damage | 243 |
| Minor damage | 243 |
| Major damage | 243 |
| Destroyed | 243 |
| **Total** | **972** |

The balanced subset is kept as analysis evidence, but the final trained models use all 2,799 complete labeled pairs through the processed dataset.

### Final Processed Dataset

Final training uses:

```text
processed_balanced_pairs_256/
```

Although the folder name contains `balanced`, the final processed dataset was generated from:

```text
dataset_work/all_pairs_index.csv
```

So it contains all 2,799 complete labeled pre/post pairs.

Final split:

| Split | Pairs |
|---|---:|
| Train | 1,959 |
| Validation | 420 |
| Test | 420 |
| **Total** | **2,799** |

Each sample has shape:

```text
256 x 256 x 6
```

The processed folder contains ready-to-train NumPy arrays:

```text
X_train.npy
y_train.npy
X_val.npy
y_val.npy
X_test.npy
y_test.npy
```

It also contains metadata files such as:

```text
train_pairs.csv
validation_pairs.csv
test_pairs.csv
class_distribution.csv
preprocessing_summary.json
split_verification_summary.json
sample_pair_preview.png
split_class_distribution.png
```

## External Artifacts

Large files are not stored directly in this GitHub repository because GitHub regular upload has a 100 MB per-file limit.

Download the required model and processed dataset artifacts from:

```text
Google Drive artifacts:
https://drive.google.com/drive/folders/1jO6sM5ITwhpdJQgm1GqR2TbwZUswZLKB?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto
```

The artifact folder contains or should contain:

```text
models_final_aug_256_15ep_b16/run_02/
processed_balanced_pairs_256/
```

For only running the web app, download:

```text
models_final_aug_256_15ep_b16/run_02/
```

For retraining or re-evaluation, also download:

```text
processed_balanced_pairs_256/
```

For full preprocessing from raw images, download the original xBD/xView2 dataset from the dataset links above and place it as:

```text
Full dataset/
```

## Repository Folder Structure

Expected project structure after downloading the repository and external artifacts:

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
├── dataset_work/
│   ├── all_pairs_index.csv
│   ├── balanced_pairs_index.csv
│   └── dataset summary/verification files
├── viz_final_aug_256_15ep_b16/
│   ├── five_run_mean_std.png
│   └── run_01 ... run_05 evaluation charts
├── models_final_aug_256_15ep_b16/
│   └── run_02/
│       ├── cnn_model.keras
│       ├── cnn_feature_extractor.keras
│       ├── cnn_knn.joblib
│       ├── metrics.json
│       └── history.csv
├── processed_balanced_pairs_256/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   └── y_test.npy
├── create_balanced_pairs.py
├── preprocess_balanced_pairs.py
├── paired_augmentation.py
├── train_evaluate_models.py
├── verify_balanced_dataset.py
├── verify_data_split.py
├── requirements.txt
└── README.md
```

Folders such as `Full dataset/`, `processed_balanced_pairs_256/`, `models_final_aug_256_15ep_b16/`, and `venv/` are intentionally excluded from GitHub when they are too large.

## Installation

Clone the repository:

```powershell
git clone <repository-url>
cd <repository-folder>
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Web App

Before running, download the final model artifacts from Google Drive and place them at:

```text
models_final_aug_256_15ep_b16/run_02/
```

Required files inside `run_02/`:

```text
cnn_model.keras
cnn_feature_extractor.keras
cnn_knn.joblib
metrics.json
history.csv
```

Run the Flask app:

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
6. View predictions, confidence scores, comparison chart, and heatmap on the Results tab.

Use matching image IDs. Example:

```text
hurricane-michael_00000004_pre_disaster.png
hurricane-michael_00000004_post_disaster.png
```

Matching IDs are important because the model compares the same location before and after a disaster.

## Heatmap Explanation

The heatmap is a visual-change map, not pixel-level damage segmentation.

Color meaning:

- Red / yellow: high visual change, possible damaged area.
- Green / cyan: moderate visual change.
- Blue / purple: low visual change.

The final damage class is predicted by the trained CNN and CNN+KNN models. The heatmap is included as a supporting visualization only.

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

Best backend model folder:

```text
models_final_aug_256_15ep_b16/run_02/
```

Run 02 was selected for backend deployment because it achieved the strongest CNN+KNN result among the final runs.

## Training and Evaluation

To retrain using the processed NumPy arrays, first download `processed_balanced_pairs_256/` from the Google Drive artifact link and place it in the project root.

Then run:

```powershell
python train_evaluate_models.py --processed-dir processed_balanced_pairs_256 --models-dir models_final_aug_256_15ep_b16 --visualizations-dir viz_final_aug_256_15ep_b16 --runs 5 --epochs 10 --batch-size 16
```

Verify the processed split:

```powershell
python verify_data_split.py --processed-dir processed_balanced_pairs_256
```

Check paired augmentation:

```powershell
python paired_augmentation.py --processed-dir processed_balanced_pairs_256 --output-dir augmentation_checks_256
```

To reproduce preprocessing from the raw dataset, download the xBD/xView2 dataset and place it as:

```text
Full dataset/
```

Then run the preprocessing scripts before training.

## Large File Policy

This repository keeps source code and small result files on GitHub. Large files are provided separately through the artifact link.

Not included directly in GitHub:

```text
Full dataset/
processed_balanced_pairs_256/
models_final_aug_256_15ep_b16/
venv/
__pycache__/
```

Reason:

- The raw dataset is several GB.
- The processed NumPy arrays are several GB.
- `cnn_model.keras` is larger than GitHub's normal 100 MB per-file limit.

## Technologies Used

- Python
- TensorFlow/Keras
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Flask
- HTML, CSS, JavaScript
- Chart.js

## Project Status

EarthScan AI includes the complete project workflow:

- Dataset indexing and preprocessing.
- Class imbalance analysis.
- Training-time augmentation.
- CNN model training.
- CNN feature extraction with KNN classification.
- Five-run evaluation.
- Flask backend integration.
- Web frontend for user input and prediction output.
- Matplotlib evaluation charts.
- Visual-change heatmap.

