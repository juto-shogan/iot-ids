"""Deep-learning focused presentation dashboard for final project walkthrough."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[dict, dict, dict]:
    """Load deep learning metrics, run metadata, and config payloads."""
    metrics_payload = _load_json(OUTPUTS_DIR / "metrics.json")
    metadata_payload = _load_json(OUTPUTS_DIR / "run_metadata.json")

    config_payload = {}
    if CONFIG_PATH.exists():
        import yaml

        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config_payload = yaml.safe_load(file)

    return metrics_payload, metadata_payload, config_payload


@st.cache_resource(show_spinner=False)
def load_dl_model_summary() -> str:
    """Load Keras model and return printable summary text."""
    dl_path = MODELS_DIR / "dl_model.keras"
    if not dl_path.exists():
        return "DL model artifact not found. Run `python main.py` first."

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(dl_path)
        stream = StringIO()
        model.summary(print_fn=lambda line: stream.write(line + "\n"))
        return stream.getvalue()
    except Exception as exc:
        return f"Could not load DL model summary: {exc}"


def draw_confusion_matrix(matrix: list[list[int]]):
    """Render confusion matrix figure."""
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
    ax.set_title("Deep Learning Confusion Matrix")
    fig.tight_layout()
    return fig


def render_metrics(dl_metrics: dict) -> None:
    """Show deep learning metrics in cards."""
    st.subheader("Deep Learning Final Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{dl_metrics.get('accuracy', 0):.4f}")
    c2.metric("Precision", f"{dl_metrics.get('precision', 0):.4f}")
    c3.metric("Recall", f"{dl_metrics.get('recall', 0):.4f}")
    c4.metric("F1-score", f"{dl_metrics.get('f1_score', 0):.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Balanced Accuracy", f"{dl_metrics.get('balanced_accuracy', 0):.4f}")
    c6.metric("MCC", f"{dl_metrics.get('mcc', 0):.4f}")
    c7.metric("ROC-AUC", f"{dl_metrics.get('roc_auc', 0):.4f}")
    c8.metric("PR-AUC", f"{dl_metrics.get('pr_auc', 0):.4f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Specificity", f"{dl_metrics.get('specificity', 0):.4f}")
    c10.metric("NPV", f"{dl_metrics.get('npv', 0):.4f}")
    c11.metric("FPR", f"{dl_metrics.get('fpr', 0):.4f}")
    c12.metric("FNR", f"{dl_metrics.get('fnr', 0):.4f}")

    threshold = dl_metrics.get("threshold")
    if threshold is not None:
        st.info(f"Final DL decision threshold: **{threshold:.2f}**")


def render_technique_summary(config_payload: dict) -> None:
    """Explain how the deep learning model was built."""
    dl_cfg = config_payload.get("dl", {}) if config_payload else {}

    st.subheader("How the Deep Learning Model Was Built")
    st.markdown(
        """
1. **Preprocessing**: Missing value imputation, categorical encoding (OneHot), numerical scaling (StandardScaler).  
2. **Feature Selection**: Variance threshold removes near-constant features.  
3. **DL Reduction**: TruncatedSVD optionally reduces sparse high-dimensional feature space.  
4. **Model Architecture**: Dense network with ReLU hidden layers and sigmoid output for binary attack detection.  
5. **Training Controls**: EarlyStopping and ModelCheckpoint.  
6. **Threshold Tuning**: Validation-based threshold sweep to maximize F1.  
7. **Evaluation**: Metrics + confusion matrix + ROC/PR artifacts written to `outputs/`.
        """
    )

    if dl_cfg:
        st.caption("Runtime DL config used")
        st.json(dl_cfg)


def build_presentation() -> None:
    """Render final deep-learning presentation dashboard."""
    st.set_page_config(page_title="DL Presentation - IoT IDS", layout="wide")
    st.title("🎓 IoT IDS Final Presentation (Deep Learning Focus)")
    st.caption("Project summary page for the deep learning model, techniques, and final results")

    metrics_payload, metadata_payload, config_payload = load_artifacts()
    dl_metrics = metrics_payload.get("Deep Learning", {})

    if not dl_metrics:
        st.warning("Deep Learning metrics not found. Run `python main.py` first.")
        st.stop()

    render_metrics(dl_metrics)

    st.subheader("Deep Learning Confusion Matrix")
    cm = dl_metrics.get("confusion_matrix")
    if cm:
        st.pyplot(draw_confusion_matrix(cm))

    st.subheader("Model Architecture Summary")
    st.code(load_dl_model_summary(), language="text")

    render_technique_summary(config_payload)

    st.subheader("Run Metadata")
    if metadata_payload:
        st.json(metadata_payload)
    else:
        st.info("run_metadata.json not found.")


if __name__ == "__main__":
    build_presentation()
