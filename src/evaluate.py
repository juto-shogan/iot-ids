"""Evaluation routines for IDS classifiers.

This module is intentionally verbose because it is often used in reports and
presentations. The goal is to make *why* a metric exists as clear as *how* it
is computed.
"""

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
    """Compute score-based metrics safely.

    Why these metrics matter:
    - ROC-AUC: measures ranking quality across all thresholds.
    - PR-AUC: focuses on positive class quality and is useful when class
      imbalance exists (common in intrusion detection).
    """
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
    """Compute core binary classification metrics and confusion matrix.

    Metric notes for IDS context:
    - Accuracy: global correctness, but can hide class imbalance effects.
    - Precision: among alerts, how many are true attacks (alert quality).
    - Recall: among real attacks, how many were caught (detection coverage).
    - F1-score: balance of precision/recall when both matter.
    - Balanced accuracy: average recall across classes; robust to imbalance.
    - MCC: strong single-score summary that includes all confusion terms.
    - Specificity / FPR: normal-traffic protection vs false alarm rate.
    - NPV / FNR: confidence in "normal" predictions and missed-attack rate.
    """
    # Confusion terms are foundational for nearly every security metric.
    # tn: benign correctly predicted benign
    # fp: benign incorrectly predicted attack (false alarm)
    # fn: attack incorrectly predicted benign (missed attack)
    # tp: attack correctly predicted attack
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0

    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": specificity,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    if y_score is not None:
        payload.update(_compute_score_metrics(y_true, y_score))
    else:
        payload.update({"roc_auc": None, "pr_auc": None})

    return payload


def _get_model_scores(model, x_test):
    """Get continuous model scores for AUC metrics when available."""
    # decision_function often exists for margin-based models (e.g., SVM) and is
    # usually faster than calibrated probabilities while still valid for AUC.
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
) -> tuple[dict, dict]:
    """Evaluate all trained models and return structured metrics and curve data.

    `curve_data` stores raw score traces so ROC/PR curves can be recreated later
    without retraining models.
    """
    results: dict[str, dict] = {}
    curve_data: dict[str, dict] = {}

    for model_name, model in trained_ml_models.items():
        logging.info("Evaluating %s...", model_name)
        predictions = model.predict(x_test_ml)
        scores = _get_model_scores(model, x_test_ml)
        results[model_name] = evaluate_predictions(y_test, predictions, y_score=scores)
        if scores is not None:
            curve_data[model_name] = {
                "y_true": y_test.astype(int).tolist(),
                "y_score": pd.Series(scores).astype(float).tolist(),
            }

    # Deep learning path uses its tuned decision threshold from validation.
    logging.info("Evaluating Deep Learning model...")
    dl_probs, dl_predictions = predict_dl(dl_model, x_test_dl, threshold=dl_threshold)
    min_len = min(len(y_test), len(dl_predictions))
    dl_y_true = y_test.iloc[:min_len].astype(int)
    dl_scores = dl_probs[:min_len]

    results["Deep Learning"] = evaluate_predictions(
        dl_y_true,
        dl_predictions[:min_len],
        y_score=dl_scores,
    )
    results["Deep Learning"]["threshold"] = float(dl_threshold)
    curve_data["Deep Learning"] = {
        "y_true": dl_y_true.tolist(),
        "y_score": pd.Series(dl_scores).astype(float).tolist(),
    }

    return results, curve_data


def save_metrics(results: dict, outputs_dir: Path) -> Path:
    """Save flattened metric table as CSV and full payload as JSON.

    CSV is convenient for spreadsheet/report workflows; JSON preserves
    full structure for downstream tooling and dashboards.
    """
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
            "specificity": metrics["specificity"],
            "npv": metrics["npv"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
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


def save_curve_data(curve_data: dict, outputs_dir: Path) -> Path:
    """Save model score traces for ROC/PR plotting."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    curve_path = outputs_dir / "curve_data.json"
    save_json(curve_data, curve_path)
    return curve_path
