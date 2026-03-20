"""Data loading logic for local IDS CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target"]
DEFAULT_DROP_COLUMNS = ["id", "ID", "record_id", "Record_ID"]


def _detect_label_column(frame: pd.DataFrame, explicit_label: str | None = None) -> str:
    if explicit_label:
        if explicit_label not in frame.columns:
            raise ValueError(
                f"Configured label column '{explicit_label}' not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return explicit_label

    for candidate in LABEL_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not find a label column. Tried: {LABEL_CANDIDATES}. "
        f"Available columns: {list(frame.columns)}"
    )


def _drop_irrelevant_columns(frame: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    drop_columns = drop_columns or DEFAULT_DROP_COLUMNS
    removable = [column for column in drop_columns if column in frame.columns]
    return frame.drop(columns=removable, errors="ignore")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a single CSV dataset file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def split_features_labels(
    frame: pd.DataFrame,
    label_column: str | None = None,
    drop_columns: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X and y for binary IDS classification."""
    clean_frame = _drop_irrelevant_columns(frame, drop_columns=drop_columns)
    label_column = _detect_label_column(clean_frame, explicit_label=label_column)
    x_data = clean_frame.drop(columns=[label_column])
    y_data = clean_frame[label_column].astype(int)
    return x_data, y_data


def load_train_test(
    train_path: Path,
    test_path: Path,
    label_column: str | None = None,
    drop_columns: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and test splits from local CSV files."""
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    x_train, y_train = split_features_labels(train_df, label_column=label_column, drop_columns=drop_columns)
    x_test, y_test = split_features_labels(test_df, label_column=label_column, drop_columns=drop_columns)

    missing_test_cols = sorted(set(x_train.columns) - set(x_test.columns))
    if missing_test_cols:
        raise ValueError(
            "Test set is missing feature columns found in training set: "
            f"{missing_test_cols}"
        )

    x_test = x_test.reindex(columns=x_train.columns)
    return x_train, y_train, x_test, y_test


def load_single_file_with_split(
    data_path: Path,
    label_column: str | None = None,
    drop_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load a single CSV then create train/test split."""
    data_df = load_dataset(data_path)
    x_all, y_all = split_features_labels(data_df, label_column=label_column, drop_columns=drop_columns)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_all,
    )
    x_test = x_test.reindex(columns=x_train.columns)
    return x_train, y_train, x_test, y_test


def load_data_by_config(config: dict, project_root: Path):
    """Load dataset according to config mode: separate or single-file split."""
    data_cfg = config.get("data", {})
    mode = data_cfg.get("mode", "separate_files")
    label_column = data_cfg.get("label_column")
    drop_columns = data_cfg.get("drop_columns", DEFAULT_DROP_COLUMNS)

    if mode == "single_file_split":
        data_path = project_root / data_cfg["single_file"]
        return load_single_file_with_split(
            data_path=data_path,
            label_column=label_column,
            drop_columns=drop_columns,
            test_size=float(data_cfg.get("test_size", 0.2)),
            random_seed=int(config.get("random_seed", 42)),
        )

    train_path = project_root / data_cfg.get("train_file", "data/UNSW_NB15_training-set.csv")
    test_path = project_root / data_cfg.get("test_file", "data/UNSW_NB15_testing-set.csv")
    return load_train_test(
        train_path=train_path,
        test_path=test_path,
        label_column=label_column,
        drop_columns=drop_columns,
    )
