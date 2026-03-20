"""Data loading logic for local IDS CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_CANDIDATES = [
    "label", "Label", "class", "Class", "target", "Target",
    "attack", "Attack", "is_attack", "IsAttack", "anomaly", "Anomaly",
    "malicious", "Malicious", "outcome", "Outcome",
]
DEFAULT_DROP_COLUMNS = ["id", "ID", "record_id", "Record_ID"]


def _normalize_label_column(label_column) -> str | None:
    """Normalize label_column config input to a single string or None."""
    if label_column is None:
        return None
    if isinstance(label_column, str):
        cleaned = label_column.strip()
        if not cleaned or cleaned in {"<your_label_column_name>", "null", "None"}:
            return None
        return cleaned
    if isinstance(label_column, list):
        if len(label_column) == 0:
            return None
        if len(label_column) == 1:
            return _normalize_label_column(label_column[0])
        raise ValueError(
            f"data.label_column must be a single column name, got list with {len(label_column)} entries: {label_column}"
        )
    raise ValueError(f"data.label_column must be string or null, got type: {type(label_column).__name__}")


def _detect_label_column(frame: pd.DataFrame, explicit_label: str | None = None) -> str:
    explicit_label = _normalize_label_column(explicit_label)
    if explicit_label:
        if explicit_label not in frame.columns:
            raise ValueError(
                f"Configured label column '{explicit_label}' not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return explicit_label

    # 1) direct candidate match
    for candidate in LABEL_CANDIDATES:
        if candidate in frame.columns:
            return candidate

    # 2) fuzzy keyword match in column names
    keyword_hits = [
        column
        for column in frame.columns
        if any(keyword in column.lower() for keyword in ["label", "class", "target", "attack", "anomaly"])
    ]
    if len(keyword_hits) == 1:
        return keyword_hits[0]

    # 3) fallback: pick binary-like non-feature column if unique values are small
    for column in frame.columns:
        unique_count = frame[column].nunique(dropna=True)
        if unique_count == 2:
            return column

    raise ValueError(
        "Could not infer a label column automatically. "
        "Please set data.label_column in config.yaml. "
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
    y_series = clean_frame[label_column]
    if pd.api.types.is_numeric_dtype(y_series):
        y_data = y_series.astype(int)
    else:
        y_data, _ = pd.factorize(y_series.astype(str))
        y_data = pd.Series(y_data, index=clean_frame.index, name=label_column).astype(int)

    if y_data.nunique() > 2:
        raise ValueError(
            f"Label column '{label_column}' has {y_data.nunique()} classes; expected binary classification."
        )
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
    label_column = _normalize_label_column(data_cfg.get("label_column"))
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
