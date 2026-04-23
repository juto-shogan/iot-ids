"""Feature engineering helpers for IDS model training."""

from __future__ import annotations

from pathlib import Path

import joblib
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    """Remove near-constant features using a configurable variance threshold.

    Near-constant columns add noise and computation while contributing little to
    decision boundaries.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.selector = VarianceThreshold(threshold=threshold)

    def fit_transform(self, x_data):
        """Fit selector on train features and transform."""
        return self.selector.fit_transform(x_data)

    def transform(self, x_data):
        """Transform features using an already-fitted selector."""
        return self.selector.transform(x_data)

    def save(self, path: Path) -> None:
        """Persist fitted selector."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.selector, path)


class DLFeatureReducer:
    """Optional dimensionality reducer for DL inputs.

    One-hot encoded IDS data can be high-dimensional and sparse; TruncatedSVD
    helps keep DL training memory and latency manageable on CPU.
    """

    def __init__(self, max_features: int | None = 256) -> None:
        self.max_features = max_features
        self.reducer: TruncatedSVD | None = None

    def fit_transform(self, x_data):
        """Fit dimensionality reducer and transform training data."""
        if not issparse(x_data) or self.max_features is None:
            return x_data

        n_components = min(self.max_features, x_data.shape[1] - 1)
        if n_components < 2:
            return x_data

        self.reducer = TruncatedSVD(n_components=n_components, random_state=42)
        return self.reducer.fit_transform(x_data)

    def transform(self, x_data):
        """Transform test data with fitted reducer if available."""
        if self.reducer is None:
            return x_data
        return self.reducer.transform(x_data)

    def save(self, path: Path) -> None:
        """Persist reducer only when fitted."""
        if self.reducer is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.reducer, path)
