# Script-by-Script Guide

This file explains what each script in the repository does.

## Root

### `main.py`
Orchestrates the full pipeline:
- load data
- preprocess
- feature selection
- train ML models
- train DL model
- evaluate all models
- save metrics/plots

### `requirements.txt`
Dependency list for Python packages needed by the project.

---

## `src/` package

### `src/data_loader.py`
- Loads CSV files from disk
- Detects label column (e.g., `label`)
- Drops ID-like columns if present
- Splits features (`X`) and labels (`y`)
- Aligns test columns to train columns

### `src/preprocessing.py`
- Builds reusable sklearn preprocessing pipeline
- Missing value handling
- Categorical encoding via `OneHotEncoder`
- Numerical scaling via `StandardScaler`
- Saves/loads fitted preprocessor object

### `src/feature_engineering.py`
- Provides optional feature selection utilities
- Includes variance-threshold-based selector
- Includes helper to reduce sparse dimensionality for DL when needed

### `src/train_ml.py`
Trains and saves traditional ML models:
- Logistic Regression
- Random Forest
- SVM
- KNN

### `src/train_dl.py`
Builds and trains deep learning model using TensorFlow/Keras:
- Dense neural network with ReLU hidden layers
- Sigmoid output for binary classification
- Uses `EarlyStopping`
- Uses `ModelCheckpoint`
- Saves DL model artifacts

### `src/evaluate.py`
Computes and stores model performance:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Saves:
- `outputs/metrics.csv`
- `outputs/metrics.json`

### `src/visualize.py`
Generates visual outputs:
- Confusion matrix image for each model
- Comparison bar chart across models

Outputs are saved under `outputs/`.

### `src/utils.py`
Shared helper utilities:
- path constants
- directory creation helpers
- logging setup
- JSON helpers

---

## `app/`

### `app/dashboard.py`
Streamlit UI for inspecting model results:
- Sidebar model selector
- Metrics display
- Confusion matrix rendering
- Model comparison display
- Optional uploaded CSV preview
