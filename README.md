# IoT Intrusion Detection System (UNSW-NB15)

A modular, reproducible IoT IDS project that trains and compares:

- Traditional ML models (Logistic Regression, Random Forest, SVM, KNN)
- A TensorFlow/Keras deep neural network
- A Streamlit dashboard for model analysis and artifacts

## Dataset

Place these local files in `data/`:

- `data/UNSW_NB15_training-set.csv`
- `data/UNSW_NB15_testing-set.csv`

> The project reads local files only and does not download data.

## Project Structure

```text
├── app/
│   └── dashboard.py
├── data/
├── docs/
│   ├── HAPPY_PATH_README.md
│   └── SCRIPTS_README.md
├── models/
├── outputs/
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_dl.py
│   ├── train_ml.py
│   ├── utils.py
│   └── visualize.py
├── config.yaml
├── main.py
└── requirements.txt
```

## Beginner Guides

- Happy path quickstart: `docs/HAPPY_PATH_README.md`
- Script-by-script walkthrough: `docs/SCRIPTS_README.md`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Pipeline

```bash
python main.py
```

Or with a custom config file:

```bash
python main.py --config config.yaml
```

Pipeline outputs:

- **models/**: trained model artifacts + preprocessors/selectors
- **outputs/metrics.csv** and **outputs/metrics.json**
- **outputs/run_metadata.json** (config + runtime info)
- confusion matrix images and model comparison chart

## Run Dashboard

```bash
streamlit run app/dashboard.py
```

Dashboard features:

- comparison overview tab with sortable chart
- per-model detail tab with extended metrics
- artifact tab (saved chart preview + CSV download)
- refresh control to reload latest outputs

## Notes

- CPU-focused defaults; GPU is not required.
- SVM and DL sampling controls are configurable in `config.yaml`.
- DL uses threshold tuning on validation data for better F1 behavior.
