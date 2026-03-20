"""Evaluation routines for IDS classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.train_dl import predict_dl


def evaluate_predictions(y_true, y_pred) -> dict:
    """Compute core binary classification metrics and confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_all_models(
    trained_ml_models: Dict[str, object],
    dl_model,
    x_test,
    y_test,
) -> dict:
    """Evaluate all trained models and return structured metrics."""
    results: dict[str, dict] = {}

    for model_name, model in trained_ml_models.items():
        predictions = model.predict(x_test)
        results[model_name] = evaluate_predictions(y_test, predictions)

    dl_predictions = predict_dl(dl_model, x_test)
    min_len = min(len(y_test), len(dl_predictions))
    results["Deep Learning"] = evaluate_predictions(
        y_test.iloc[:min_len],
        dl_predictions[:min_len],
    )

    return results


def save_metrics(results: dict, outputs_dir: Path) -> Path:
    """Save flattened metric table as CSV and full payload as JSON."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for model_name, metrics in results.items():
        row = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "tn": metrics["confusion_matrix"][0][0],
            "fp": metrics["confusion_matrix"][0][1],
            "fn": metrics["confusion_matrix"][1][0],
            "tp": metrics["confusion_matrix"][1][1],
        }
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1_score", ascending=False)
    csv_path = outputs_dir / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    json_path = outputs_dir / "metrics.json"
    import json

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return csv_path
