"""Training functions for traditional ML IDS classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def build_ml_models(random_seed: int = 42) -> Dict[str, object]:
    """Instantiate baseline ML models for binary intrusion detection.

    Why these models:
    - Logistic Regression: strong linear baseline and interpretable coefficients.
    - Random Forest: robust non-linear ensemble with good tabular performance.
    - SVM (RBF): margin-based non-linear classifier for complex boundaries.
    - KNN: distance-based local baseline for comparative analysis.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=600, random_state=random_seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=random_seed),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def _sample_if_needed(x_data, y_data, max_samples: int | None):
    """
    Subsample data for expensive models to keep runtime CPU-friendly.
    """
    if max_samples is None or len(y_data) <= max_samples:
        return x_data, y_data

    indices = np.random.default_rng(42).choice(len(y_data), size=max_samples, replace=False)
    indices = np.sort(indices)

    if issparse(x_data):
        x_sample = x_data[indices]
    else:
        x_sample = x_data[indices, :]
    return x_sample, y_data.iloc[indices]


def train_ml_models(
    x_train,
    y_train,
    models_dir: Path,
    svm_max_samples: int | None = 30000,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Train and persist all traditional ML models.

    Each trained estimator is saved to `models/` so dashboards and prediction
    flows can reuse them without retraining.
    """
    models = build_ml_models(random_seed=random_seed)
    trained_models: Dict[str, object] = {}

    for model_name, model in models.items():
        if model_name == "SVM":
            x_fit, y_fit = _sample_if_needed(x_train, y_train, svm_max_samples)
        else:
            x_fit, y_fit = x_train, y_train

        model.fit(x_fit, y_fit)
        trained_models[model_name] = model

        model_path = models_dir / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)

    return trained_models
