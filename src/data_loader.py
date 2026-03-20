"""Data loading logic for UNSW-NB15 train/test CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

LABEL_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target"]
DROP_COLUMNS = ["id", "ID", "record_id", "Record_ID"]


def _detect_label_column(frame: pd.DataFrame) -> str:
    for candidate in LABEL_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not find a label column. Tried: {LABEL_CANDIDATES}. "
        f"Available columns: {list(frame.columns)}"
    )


def _drop_irrelevant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    removable = [column for column in DROP_COLUMNS if column in frame.columns]
    return frame.drop(columns=removable, errors="ignore")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a single CSV dataset file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def split_features_labels(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X and y for binary IDS classification."""
    clean_frame = _drop_irrelevant_columns(frame)
    label_column = _detect_label_column(clean_frame)
    x_data = clean_frame.drop(columns=[label_column])
    y_data = clean_frame[label_column].astype(int)
    return x_data, y_data


def load_train_test(
    train_path: Path,
    test_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and test splits from local CSV files."""
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    x_train, y_train = split_features_labels(train_df)
    x_test, y_test = split_features_labels(test_df)

    missing_test_cols = sorted(set(x_train.columns) - set(x_test.columns))
    if missing_test_cols:
        raise ValueError(
            "Test set is missing feature columns found in training set: "
            f"{missing_test_cols}"
        )

    x_test = x_test.reindex(columns=x_train.columns)
    return x_train, y_train, x_test, y_test
