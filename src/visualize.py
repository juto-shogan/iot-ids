"""Visualization utilities for model diagnostics and comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


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
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    melted = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison on UNSW-NB15")
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path = outputs_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=140)
    plt.close()

    return output_path
