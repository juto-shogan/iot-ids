# Script-by-Script Guide

## Root

### `config.yaml`
Central configuration for paths, seed, preprocessing, ML, and DL runtime parameters.

### `main.py`
End-to-end orchestrator:
1. Load config and set seeds
2. Load train/test data
3. Preprocess and select features
4. Train ML models
5. Reduce features for DL + train DL model
6. Evaluate all models
7. Save metrics, plots, and run metadata

### `requirements.txt`
Python dependencies.

## `src/`

### `src/data_loader.py`
Loads CSVs, detects label column, drops irrelevant ID columns, and aligns train/test feature schema.

### `src/preprocessing.py`
Reusable sklearn preprocessing pipeline (imputation + scaling + one-hot encoding) with save/load.

### `src/feature_engineering.py`
Feature selection (`VarianceThreshold`) and optional DL dimensionality reduction (`TruncatedSVD`) with persistence.

### `src/train_ml.py`
Trains/saves Logistic Regression, Random Forest, SVM, and KNN.

### `src/train_dl.py`
Builds/trains Keras DNN, applies callbacks (`EarlyStopping`, `ModelCheckpoint`), and tunes decision threshold on validation set.

### `src/evaluate.py`
Computes core + extended metrics (Accuracy, Precision, Recall, F1, Balanced Accuracy, MCC, ROC-AUC, PR-AUC), and writes CSV/JSON outputs.

### `src/visualize.py`
Saves confusion matrix images for each model and a grouped model-comparison bar chart.

### `src/utils.py`
Shared helpers for logging, config loading, directory setup, JSON I/O, and run metadata assembly.

## `app/`

### `app/dashboard.py`
Streamlit UI with tabs:
- **Overview** (table + ranking chart + metadata)
- **Model Detail** (metrics cards + confusion matrix)
- **Artifacts** (saved plot preview + CSV download + optional uploaded CSV preview)
