"""Streamlit dashboard for IoT IDS model inspection and evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy", "mcc"]


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
    """Render saved plot previews and optional uploads."""
    st.subheader("Saved Artifacts")
    plot_path = OUTPUTS_DIR / "model_comparison.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Model comparison chart", use_container_width=True)

    st.download_button(
        label="Download metrics.csv",
        data=(OUTPUTS_DIR / "metrics.csv").read_bytes() if (OUTPUTS_DIR / "metrics.csv").exists() else b"",
        file_name="metrics.csv",
        mime="text/csv",
    )

    st.subheader("Optional: Upload new dataset for quick preview")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("Preview", uploaded_df.head(10))



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

    metrics_df, payload, metadata = load_metrics()

    if metrics_df.empty:
        st.sidebar.warning("No outputs found. Run `python main.py` first.")
        st.stop()

    model_options = metrics_df["model"].tolist()
    selected_model = st.sidebar.selectbox("Select model", model_options)

    tab_overview, tab_model, tab_artifacts = st.tabs(["Overview", "Model Detail", "Artifacts"])

    with tab_overview:
        render_overview(metrics_df, metadata)
    with tab_model:
        render_model_detail(selected_model, metrics_df, payload)
    with tab_artifacts:
        render_artifacts()

    st.divider()
    render_model_notes()


if __name__ == "__main__":
    build_dashboard()
