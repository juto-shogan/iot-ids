# Happy Path Quickstart (Beginner-Friendly)

This is the **fastest path** to run the IoT IDS project end-to-end.

## 1) Put the dataset files in `data/`
Make sure these files exist locally:

- `data/UNSW_NB15_training-set.csv`
- `data/UNSW_NB15_testing-set.csv`

The project reads from local `data/` only (no dataset downloads).

---

## 2) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Run the full training + evaluation pipeline

```bash
python main.py
```

What this does:
1. Loads training/testing data
2. Preprocesses features
3. Trains ML models (Logistic Regression, Random Forest, SVM, KNN)
4. Trains deep learning model (TensorFlow/Keras)
5. Evaluates all models (Accuracy, Precision, Recall, F1, Confusion Matrix)
6. Saves artifacts to:
   - `models/` (trained models)
   - `outputs/` (metrics + plots)

---

## 4) Open the Streamlit dashboard

```bash
streamlit run app/dashboard.py
```

In the dashboard you can:
- Select a model from the sidebar
- View Accuracy / Precision / Recall / F1
- View confusion matrix
- Compare all models
- Optionally upload a CSV for preview

---

## 5) If something fails

- Confirm your dataset filenames match exactly.
- Confirm your virtual environment is active.
- Re-run dependency installation with:

```bash
pip install -r requirements.txt
```

- Then re-run:

```bash
python main.py
```
