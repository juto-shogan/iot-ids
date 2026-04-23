"""Streamlit dashboard for IoT IDS model inspection and evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "balanced_accuracy",
    "specificity",
    "mcc",
    "roc_auc",
    "pr_auc",
    "npv",
    "fpr",
    "fnr",
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "KNN": "knn.joblib",
}


@st.cache_data(show_spinner=False)
def load_metrics() -> tuple[pd.DataFrame, dict, dict]:
    """Load metrics table, detailed JSON results, and run metadata."""
    csv_path = OUTPUTS_DIR / "metrics.csv"
    json_path = OUTPUTS_DIR / "metrics.json"
    metadata_path = OUTPUTS_DIR / "run_metadata.json"

    if not csv_path.exists() or not json_path.exists():
        return pd.DataFrame(), {}, {}

    metrics_df = pd.read_csv(csv_path)
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    return metrics_df, payload, metadata


@st.cache_resource(show_spinner=False)
def load_prediction_artifacts():
    """Load preprocessing and model artifacts for single-row predictions."""
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    selector_path = MODELS_DIR / "feature_selector.joblib"

    if not preprocessor_path.exists() or not selector_path.exists():
        return None

    preprocessor = joblib.load(preprocessor_path)
    selector = joblib.load(selector_path)

    ml_models = {}
    for model_name, file_name in MODEL_FILES.items():
        model_path = MODELS_DIR / file_name
        if model_path.exists():
            ml_models[model_name] = joblib.load(model_path)

    dl_model = None
    try:
        import tensorflow as tf

        dl_path = MODELS_DIR / "dl_model.keras"
        if dl_path.exists():
            dl_model = tf.keras.models.load_model(dl_path)
    except Exception:
        dl_model = None

    reducer_path = MODELS_DIR / "dl_reducer.joblib"
    reducer = joblib.load(reducer_path) if reducer_path.exists() else None

    return {
        "preprocessor": preprocessor,
        "selector": selector,
        "ml_models": ml_models,
        "dl_model": dl_model,
        "reducer": reducer,
    }


def draw_confusion_matrix(matrix: list[list[int]], model_name: str):
    """Render confusion matrix as matplotlib figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Normal", "Pred Attack"],
        yticklabels=["True Normal", "True Attack"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    return fig


def _extract_raw_schema(preprocessor) -> tuple[list[str], list[str], dict]:
    """Extract raw numeric/categorical column schema from fitted ColumnTransformer."""
    numeric_cols, categorical_cols = [], []
    category_map: dict[str, list] = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            numeric_cols.extend(list(columns))
        elif name == "cat":
            categorical_cols.extend(list(columns))
            encoder = transformer.named_steps.get("encoder")
            if encoder is not None and hasattr(encoder, "categories_"):
                for col, cats in zip(columns, encoder.categories_):
                    category_map[col] = list(cats)

    return numeric_cols, categorical_cols, category_map


def _predict_single_row(model_name: str, input_df: pd.DataFrame, artifacts: dict, payload: dict) -> tuple[int, float | None]:
    """
    Run single-row inference for selected model.

    Uses exactly the same preprocessing + feature selection pipeline as training,
    which is essential for consistent predictions.
    """
    preprocessor = artifacts["preprocessor"]
    selector = artifacts["selector"]

    x_processed = preprocessor.transform(input_df)
    x_selected = selector.transform(x_processed)

    if model_name == "Deep Learning":
        dl_model = artifacts["dl_model"]
        if dl_model is None:
            raise ValueError("Deep Learning model artifact not found.")

        reducer = artifacts["reducer"]
        x_dl = reducer.transform(x_selected) if reducer is not None else x_selected
        if hasattr(x_dl, "toarray"):
            x_dl = x_dl.toarray()

        prob = float(dl_model.predict(x_dl, verbose=0).flatten()[0])
        
        # Deep learning threshold is tuned during training (not always 0.5).
        threshold = payload.get("Deep Learning", {}).get("threshold", 0.5)
        pred = int(prob >= threshold)
        return pred, prob

    model = artifacts["ml_models"].get(model_name)
    if model is None:
        raise ValueError(f"Model artifact for '{model_name}' not found.")

    pred = int(model.predict(x_selected)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_selected)
        if proba.ndim == 2 and proba.shape[1] > 1:
            prob = float(proba[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(x_selected)[0])
        prob = score

    return pred, prob


def render_prediction_panel(selected_model: str, payload: dict) -> None:
    """Render interactive manual input panel for single prediction."""
    st.subheader("Single Prediction Input")
    artifacts = load_prediction_artifacts()

    if artifacts is None:
        st.info("Prediction artifacts not found. Run `python main.py` first.")
        return

    numeric_cols, categorical_cols, category_map = _extract_raw_schema(artifacts["preprocessor"])

    with st.form("single_prediction_form"):
        st.caption("Enter feature values and click Predict. Numeric/categorical fields are inferred from the fitted training preprocessor.")
        values = {}

        with st.expander("Numeric features", expanded=True):
            for col in numeric_cols:
                values[col] = st.number_input(col, value=0.0, format="%.6f")

        with st.expander("Categorical features", expanded=True):
            for col in categorical_cols:
                options = category_map.get(col, [])
                if options:
                    values[col] = st.selectbox(col, options=[str(o) for o in options])
                else:
                    values[col] = st.text_input(col, value="")

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([values])
        try:
            pred, score = _predict_single_row(selected_model, input_df, artifacts, payload)
            label = "Attack" if pred == 1 else "Normal"
            st.success(f"Prediction ({selected_model}): {label}")
            if score is not None:
                st.write(f"Model score/probability: `{score:.6f}`")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_overview(metrics_df: pd.DataFrame, metadata: dict) -> None:
    """Render a global view of model performance."""
    st.subheader("Model Comparison Overview")
    st.dataframe(metrics_df, use_container_width=True)

    available_metrics = [m for m in DEFAULT_METRICS if m in metrics_df.columns]
    selected_metric = st.selectbox("Metric for ranking", available_metrics, index=3)

    chart_df = metrics_df[["model", selected_metric]].sort_values(by=selected_metric, ascending=False)
    st.bar_chart(chart_df.set_index("model"))

    if metadata:
        with st.expander("Run metadata"):
            st.json(metadata)


def render_model_detail(selected_model: str, metrics_df: pd.DataFrame, payload: dict) -> None:
    """Render per-model detail view."""
    selected_row = metrics_df.loc[metrics_df["model"] == selected_model].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{selected_row['accuracy']:.4f}")
    col2.metric("Precision", f"{selected_row['precision']:.4f}")
    col3.metric("Recall", f"{selected_row['recall']:.4f}")
    col4.metric("F1-score", f"{selected_row['f1_score']:.4f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Balanced Accuracy", f"{selected_row.get('balanced_accuracy', 0):.4f}")
    col6.metric("MCC", f"{selected_row.get('mcc', 0):.4f}")
    roc_val = selected_row.get("roc_auc")
    pr_val = selected_row.get("pr_auc")
    col7.metric("ROC-AUC", "N/A" if pd.isna(roc_val) else f"{roc_val:.4f}")
    col8.metric("PR-AUC", "N/A" if pd.isna(pr_val) else f"{pr_val:.4f}")

    st.subheader(f"Confusion Matrix: {selected_model}")
    if selected_model in payload:
        fig = draw_confusion_matrix(payload[selected_model]["confusion_matrix"], selected_model)
        st.pyplot(fig)


def render_artifacts() -> None:
    """Render saved plot previews and downloads."""
    st.subheader("Saved Artifacts")
    artifact_images = [
        ("model_comparison.png", "Model comparison chart"),
        ("metric_heatmap.png", "Metric heatmap"),
        ("radar_comparison.png", "Radar comparison"),
        ("roc_curves.png", "ROC curves"),
        ("pr_curves.png", "Precision-Recall curves"),
    ]
    for file_name, caption in artifact_images:
        plot_path = OUTPUTS_DIR / file_name
        if plot_path.exists():
            st.image(str(plot_path), caption=caption, use_container_width=True)

    st.download_button(
        label="Download metrics.csv",
        data=(OUTPUTS_DIR / "metrics.csv").read_bytes() if (OUTPUTS_DIR / "metrics.csv").exists() else b"",
        file_name="metrics.csv",
        mime="text/csv",
    )


def render_model_notes() -> None:
    """Explain how each model works in beginner-friendly terms."""
    st.subheader("How each model works")
    st.markdown(
        """
- **Logistic Regression**: Learns a linear decision boundary and outputs attack probability from weighted features.
- **Random Forest**: Combines many decision trees; each tree votes and the majority decision is used.
- **SVM (RBF)**: Finds a maximum-margin boundary in transformed feature space; good for non-linear separation.
- **KNN**: Looks at the closest training examples and predicts based on neighbor majority label.
- **Deep Learning (DNN)**: Multi-layer neural network learns hierarchical patterns; final sigmoid output is thresholded into normal/attack.

All models are trained on the same preprocessed UNSW-NB15 feature set; DL additionally uses dimensionality reduction before training for CPU-friendly runtime.
        """
    )


def build_dashboard() -> None:
    """Build and launch Streamlit IDS dashboard."""
    st.set_page_config(page_title="IoT IDS Dashboard", layout="wide")
    st.title("🛡️ IoT Intrusion Detection System Dashboard")
    st.caption("UNSW-NB15 model evaluation (ML + Deep Learning)")

    st.sidebar.header("Controls")
    if st.sidebar.button("Refresh outputs"):
        load_metrics.clear()
        load_prediction_artifacts.clear()

    metrics_df, payload, metadata = load_metrics()

    if metrics_df.empty:
        st.sidebar.warning("No outputs found. Run `python main.py` first.")
        st.stop()

    model_options = metrics_df["model"].tolist()
    selected_model = st.sidebar.selectbox("Select model", model_options)

    tab_overview, tab_model, tab_predict, tab_artifacts = st.tabs([
        "Overview",
        "Model Detail",
        "Predict",
        "Artifacts",
    ])

    with tab_overview:
        render_overview(metrics_df, metadata)
    with tab_model:
        render_model_detail(selected_model, metrics_df, payload)
    with tab_predict:
        render_prediction_panel(selected_model, payload)
    with tab_artifacts:
        render_artifacts()

    st.divider()
    render_model_notes()


if __name__ == "__main__":
    build_dashboard()
