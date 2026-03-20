"""Visualization utilities for model diagnostics and comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve


sns.set_theme(style="whitegrid")


CORE_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "balanced_accuracy",
    "specificity",
    "mcc",
    "roc_auc",
    "pr_auc",
]


def plot_confusion_matrices(results: dict, outputs_dir: Path) -> list[Path]:
    """Create confusion matrix plot for each model."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for model_name, metrics in results.items():
        matrix = metrics["confusion_matrix"]

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred Normal", "Pred Attack"],
            yticklabels=["True Normal", "True Attack"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        file_name = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        output_path = outputs_dir / file_name
        plt.savefig(output_path, dpi=140)
        plt.close()
        saved_paths.append(output_path)

    return saved_paths


def plot_model_comparison(results: dict, outputs_dir: Path) -> Path:
    """Create grouped bar chart comparing model performance metrics."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update({metric: metrics.get(metric) for metric in CORE_METRICS})
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    melted = metrics_df.melt(id_vars="model", var_name="metric", value_name="score").dropna()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison on UNSW-NB15")
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path = outputs_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=140)
    plt.close()

    return output_path


def plot_metric_heatmap(results: dict, outputs_dir: Path) -> Path:
    """Create heatmap across models and selected metrics."""
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update({metric: metrics.get(metric) for metric in CORE_METRICS})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    plt.figure(figsize=(10, 5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.title("Metric Heatmap Across Models")
    plt.tight_layout()

    output_path = outputs_dir / "metric_heatmap.png"
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def plot_radar_chart(results: dict, outputs_dir: Path) -> Path:
    """Create radar chart for core metrics per model."""
    radar_metrics = ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy", "specificity"]
    labels = np.array(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for model_name, metrics in results.items():
        values = [float(metrics.get(metric, 0) or 0) for metric in radar_metrics]
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Radar Comparison of Core Metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
    plt.tight_layout()

    output_path = outputs_dir / "radar_comparison.png"
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def plot_roc_curves(curve_data: dict, outputs_dir: Path) -> Path:
    """Plot ROC curves for all models with score traces."""
    plt.figure(figsize=(8, 6))
    for model_name, payload in curve_data.items():
        y_true = np.array(payload["y_true"]) 
        y_score = np.array(payload["y_score"]) 
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=model_name)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()

    output_path = outputs_dir / "roc_curves.png"
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def plot_pr_curves(curve_data: dict, outputs_dir: Path) -> Path:
    """Plot precision-recall curves for all models with score traces."""
    plt.figure(figsize=(8, 6))
    for model_name, payload in curve_data.items():
        y_true = np.array(payload["y_true"]) 
        y_score = np.array(payload["y_score"]) 
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=model_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()

    output_path = outputs_dir / "pr_curves.png"
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path
