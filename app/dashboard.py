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


def load_metrics() -> tuple[pd.DataFrame, dict]:
    """Load metrics table and detailed JSON results from outputs directory."""
    csv_path = OUTPUTS_DIR / "metrics.csv"
    json_path = OUTPUTS_DIR / "metrics.json"

    if not csv_path.exists() or not json_path.exists():
        return pd.DataFrame(), {}

    metrics_df = pd.read_csv(csv_path)
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return metrics_df, payload


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


def build_dashboard() -> None:
    """Build and launch Streamlit IDS dashboard."""
    st.set_page_config(page_title="IoT IDS Dashboard", layout="wide")
    st.title("🛡️ IoT Intrusion Detection System Dashboard")
    st.caption("UNSW-NB15 model evaluation (ML + Deep Learning)")

    metrics_df, payload = load_metrics()

    st.sidebar.header("Controls")
    if metrics_df.empty:
        st.sidebar.warning("No outputs found. Run `python main.py` first.")
        st.stop()

    model_options = metrics_df["model"].tolist()
    selected_model = st.sidebar.selectbox("Select model", model_options)
    run_eval = st.sidebar.button("Run evaluation")

    if run_eval:
        st.sidebar.success("Evaluation results loaded.")

    selected_row = metrics_df.loc[metrics_df["model"] == selected_model].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{selected_row['accuracy']:.4f}")
    col2.metric("Precision", f"{selected_row['precision']:.4f}")
    col3.metric("Recall", f"{selected_row['recall']:.4f}")
    col4.metric("F1-score", f"{selected_row['f1_score']:.4f}")

    st.subheader(f"Confusion Matrix: {selected_model}")
    if selected_model in payload:
        fig = draw_confusion_matrix(payload[selected_model]["confusion_matrix"], selected_model)
        st.pyplot(fig)

    st.subheader("Model Comparison")
    st.dataframe(metrics_df, use_container_width=True)

    plot_path = OUTPUTS_DIR / "model_comparison.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Comparison chart", use_container_width=True)

    st.subheader("Optional: Upload new dataset for quick scoring")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("Preview", uploaded_df.head())
        st.info(
            "Advanced scoring pipeline is optional; this preview allows manual inspection "
            "before integrating real-time prediction endpoints."
        )


if __name__ == "__main__":
    build_dashboard()
