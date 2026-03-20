"""Evaluation routines for IDS classifiers."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.train_dl import predict_dl
from src.utils import save_json


def _compute_score_metrics(y_true, y_score) -> dict:
    """Compute score-based metrics safely."""
    metrics = {"roc_auc": None, "pr_auc": None}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        pass

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        pass

    return metrics


def evaluate_predictions(y_true, y_pred, y_score=None) -> dict:
    """Compute core binary classification metrics and confusion matrix."""
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_score is not None:
        payload.update(_compute_score_metrics(y_true, y_score))
    else:
        payload.update({"roc_auc": None, "pr_auc": None})

    return payload


def _get_model_scores(model, x_test):
    """Get continuous model scores for AUC metrics when available."""
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs

    return None


def evaluate_all_models(
    trained_ml_models: Dict[str, object],
    dl_model,
    dl_threshold: float,
    x_test_ml,
    x_test_dl,
    y_test,
) -> dict:
    """Evaluate all trained models and return structured metrics."""
    results: dict[str, dict] = {}

    for model_name, model in trained_ml_models.items():
        logging.info("Evaluating %s...", model_name)
        predictions = model.predict(x_test_ml)
        scores = _get_model_scores(model, x_test_ml)
        results[model_name] = evaluate_predictions(y_test, predictions, y_score=scores)

    logging.info("Evaluating Deep Learning model...")
    dl_probs, dl_predictions = predict_dl(dl_model, x_test_dl, threshold=dl_threshold)
    min_len = min(len(y_test), len(dl_predictions))
    results["Deep Learning"] = evaluate_predictions(
        y_test.iloc[:min_len],
        dl_predictions[:min_len],
        y_score=dl_probs[:min_len],
    )
    results["Deep Learning"]["threshold"] = float(dl_threshold)

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
            "balanced_accuracy": metrics["balanced_accuracy"],
            "mcc": metrics["mcc"],
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
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
    save_json(results, json_path)

    return csv_path
