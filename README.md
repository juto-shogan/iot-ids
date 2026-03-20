# IoT Intrusion Detection System (UNSW-NB15)

A production-style Python project for simulating an IoT Intrusion Detection System (IDS) with:

- **Traditional machine learning** classifiers (Logistic Regression, Random Forest, SVM, KNN)
- **Deep learning** classifier (TensorFlow/Keras feed-forward neural network)
- **Streamlit dashboard** for interactive model evaluation and visualization

## Dataset

This project uses the **UNSW-NB15** dataset with local files only:

- `data/UNSW_NB15_training-set.csv`
- `data/UNSW_NB15_testing-set.csv`

> The pipeline is configured to load from the local `data/` directory and does not download data from the internet.

## Project Structure

```text
iot_ids_project/
│
├── data/                  # local dataset files (provided externally)
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
│
├── models/                # trained model artifacts
├── outputs/               # metrics and plots
├── app/
│   └── dashboard.py       # Streamlit dashboard
│
├── main.py                # full pipeline entrypoint
├── requirements.txt
└── README.md
```

## Key Features

1. **Data loading and validation**
   - Train/test CSV loading
   - Automatic label column detection (`label`, `class`, etc.)
   - Irrelevant ID-like columns dropped when present

2. **Reusable preprocessing**
   - Missing value imputation
   - Categorical encoding with `OneHotEncoder`
   - Numerical scaling with `StandardScaler`
   - Consistent train/test transformation via saved preprocessor artifact

3. **Model training**
   - ML models trained and saved to `models/`
   - DNN model trained with `EarlyStopping` and `ModelCheckpoint`

4. **Evaluation and outputs**
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix
   - Saved as `outputs/metrics.csv` and `outputs/metrics.json`
   - Confusion matrix plots + model comparison chart saved to `outputs/`

5. **Interactive dashboard**
   - Model selection from sidebar
   - Metrics display cards
   - Confusion matrix display
   - Cross-model comparison table/chart
   - Optional CSV upload preview


## Beginner Guides

- Happy path quickstart: `docs/HAPPY_PATH_README.md`
- Script-by-script walkthrough: `docs/SCRIPTS_README.md`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Training Pipeline

```bash
python main.py
```

Pipeline steps:

1. Load training/testing data from `data/`
2. Preprocess and feature-select
3. Train ML and DL models
4. Evaluate all models on test data
5. Save models, metrics, and plots

## Run Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Then open the local Streamlit URL in your browser.

## Notes on CPU Optimization

- Pipeline is CPU-friendly and does not assume GPU.
- SVM training can be computationally heavy on very large data; controlled sampling is used by default for practical runtime.
- Deep learning training uses early stopping to avoid unnecessary epochs.

## Academic / Defensive Use

This repository is structured for reproducibility and defensible reporting in cybersecurity and ML coursework.
