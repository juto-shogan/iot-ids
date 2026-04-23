"""Reusable preprocessing pipelines for mixed tabular data."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    """
    Wrapper around sklearn preprocessing objects for consistent train/test transforms.

    Consistency is critical: fitting on train and reusing identical transforms on
    test/inference prevents leakage and schema drift.
    """

    def __init__(self) -> None:
        self.transformer: ColumnTransformer | None = None

    @staticmethod
    def _build_transformer(x_data: pd.DataFrame) -> ColumnTransformer:
        """
        Build mixed-type preprocessing graph for tabular IDS features."""
        numeric_columns = x_data.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = x_data.select_dtypes(exclude=["number"]).columns.tolist()

        # Numeric features: median imputation + z-score scaling.
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical features: most-frequent imputation + one-hot encoding.
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

    def fit_transform(self, x_train: pd.DataFrame):
        """Fit preprocessor on training data then transform it."""
        self.transformer = self._build_transformer(x_train)
        return self.transformer.fit_transform(x_train)

    def transform(self, x_data: pd.DataFrame):
        """Transform data with previously fitted preprocessor."""
        if self.transformer is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.transformer.transform(x_data)

    def save(self, path: Path) -> None:
        """Persist fitted transformer using joblib."""
        if self.transformer is None:
            raise ValueError("Cannot save an unfitted preprocessor.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformer, path)

    def load(self, path: Path) -> None:
        """Load transformer from disk."""
        self.transformer = joblib.load(path)


def preprocess_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    preprocessor_path: Path,
) -> Tuple[object, object, Preprocessor]:
    """Fit preprocessing on train split and apply to test split."""
    preprocessor = Preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    preprocessor.save(preprocessor_path)
    return x_train_processed, x_test_processed, preprocessor
