"""Feature engineering helpers for IDS model training."""

from __future__ import annotations

from typing import Tuple

from scipy.sparse import issparse
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    """Remove near-constant features using a configurable variance threshold."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.selector = VarianceThreshold(threshold=threshold)

    def fit_transform(self, x_data):
        """Fit selector on train features and transform."""
        return self.selector.fit_transform(x_data)

    def transform(self, x_data):
        """Transform features using an already-fitted selector."""
        return self.selector.transform(x_data)


def prepare_features_for_dl(x_train, x_test, max_features: int | None = 256) -> Tuple[object, object]:
    """Optionally reduce high-dimensional sparse matrices for DL compatibility."""
    if not issparse(x_train) or max_features is None:
        return x_train, x_test

    from sklearn.decomposition import TruncatedSVD

    n_components = min(max_features, x_train.shape[1] - 1)
    if n_components < 2:
        return x_train, x_test

    reducer = TruncatedSVD(n_components=n_components, random_state=42)
    return reducer.fit_transform(x_train), reducer.transform(x_test)
