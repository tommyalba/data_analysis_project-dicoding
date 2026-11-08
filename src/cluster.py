"""
Unsupervised customer segmentation via k-means — reusable module.

Implements the standard RFM-clustering workflow (course reference:
"Data Preprocessing for Clustering" + "Customer Segmentation with
K-means"):

    1. ``preprocess()`` — log-transform skewed features, then centre and
       scale with ``StandardScaler`` so k-means distances are meaningful
    2. ``elbow()``     — sum-of-squared-errors (inertia) and silhouette
       scores across a range of *k* to choose the cluster count
    3. ``fit()``       — run ``KMeans`` on the normalised features
    4. ``profile()``   — per-cluster means of the *raw* features plus sizes
    5. ``relative_importance()`` — cluster mean / population mean − 1,
       i.e. how much each attribute deviates per segment
    6. ``snake_data()``— melted normalised frame ready for a snake plot

Design notes
------------

* **Log-then-scale order matters** — ``np.log1p`` first (handles zeros,
  unlike ``np.log``), then ``StandardScaler``. Reversing the order would
  standardise the skew, not remove it.
* **``log1p`` over ``log``** — ``log1p(x) = log(1 + x)`` is monotone,
  defined at zero, and ≈ ``log`` for large x, so it is a safe drop-in
  for count-like features that can be 0.
* **Clusters ≠ segments** — k-means partitions on geometry, not business
  rules. Always inspect ``profile()`` on the *raw* scale before naming
  personas; the normalised space is only for fitting.
* **Degenerate features** — if a feature is (nearly) constant, its
  z-score variance explodes relative to its information content. Check
  ``raw_.describe()`` first; drop or winsorise such columns upstream.

Example
-------
>>> from src.cluster import ClusterAnalyzer
>>> ca = ClusterAnalyzer(rfm_df, feature_cols=["recency", "frequency", "monetary"])
>>> ca.preprocess()
>>> sse, sil = ca.elbow(k_range=range(2, 11))
>>> ca.fit(k=4)
>>> ca.profile()
>>> ca.relative_importance()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class ClusterAnalyzer:
    """K-means clustering pipeline for entity-level feature tables.

    Parameters
    ----------
    df : pd.DataFrame
        One row per entity (e.g. the output of ``RFMAnalyzer.scores``).
    feature_cols : list[str]
        Numeric columns used as clustering features.
    entity_col : str, optional
        Identifier column carried through for joining labels back.
    random_state : int
        Seed for reproducible k-means initialisation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        entity_col: str | None = None,
        random_state: int = 42,
    ) -> None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols not in df: {missing}")
        if entity_col is not None and entity_col not in df.columns:
            raise ValueError(f"entity_col '{entity_col}' not in df")
        nonnum = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if nonnum:
            raise ValueError(f"feature_cols must be numeric: {nonnum}")
        if df[feature_cols].isna().any().any():
            raise ValueError("feature_cols contain missing values — clean upstream")
        if len(df) < 2:
            raise ValueError("need at least 2 rows to cluster")

        self.feature_cols = list(feature_cols)
        self.entity_col = entity_col
        self.random_state = random_state
        keep = ([entity_col] if entity_col else []) + self.feature_cols
        self.raw_ = df[keep].copy()
        self.X_: pd.DataFrame | None = None      # normalised features
        self.model_: KMeans | None = None
        self.labels_: np.ndarray | None = None

    # ── Step 1: preprocess ────────────────────────────────────────────────
    def preprocess(self, transform: str = "log1p") -> "ClusterAnalyzer":
        """Log-transform then standardise features → ``self.X_``.

        ``transform='log1p'`` applies ``log1p`` (skew removal for
        non-negative count/value features); ``'none'`` skips it.
        """
        feats = self.raw_[self.feature_cols].astype(float)
        if transform == "log1p":
            feats = np.log1p(feats.clip(lower=0))
        elif transform != "none":
            raise ValueError(f"unknown transform '{transform}'")
        scaled = StandardScaler().fit_transform(feats)
        self.X_ = pd.DataFrame(
            scaled, index=self.raw_.index, columns=self.feature_cols
        )
        return self

    # ── Step 2: choose k ──────────────────────────────────────────────────
    def elbow(
        self,
        k_range: range | list[int] = range(2, 11),
        sil_sample: int = 10000,
    ) -> tuple[pd.Series, pd.Series]:
        """Return (SSE, silhouette) per k. Requires ``preprocess()``.

        SSE (inertia) is computed on all rows; silhouette is O(n²), so it
        is estimated on a random ``sil_sample`` subset when n is large.
        """
        self._require_X()
        k_range = list(k_range)
        if min(k_range) < 2 or max(k_range) >= len(self.X_):
            raise ValueError("k_range must satisfy 2 <= k < n_rows")
        sse, sil = {}, {}
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(self.X_)
            sse[k] = km.inertia_
            sil[k] = silhouette_score(
                self.X_, labels,
                sample_size=min(sil_sample, len(self.X_)),
                random_state=self.random_state,
            )
        return pd.Series(sse, name="sse"), pd.Series(sil, name="silhouette")

    # ── Step 3: fit ───────────────────────────────────────────────────────
    def fit(self, k: int) -> "ClusterAnalyzer":
        """Fit k-means with ``k`` clusters → ``self.labels_``."""
        self._require_X()
        if not 2 <= k < len(self.X_):
            raise ValueError(f"k must satisfy 2 <= k < n_rows, got {k}")
        self.model_ = KMeans(
            n_clusters=k, n_init=10, random_state=self.random_state
        ).fit(self.X_)
        self.labels_ = self.model_.labels_
        return self

    # ── Step 4-6: interpret ───────────────────────────────────────────────
    def labels_frame(self) -> pd.DataFrame:
        """Raw features + cluster label, one row per entity."""
        self._require_labels()
        out = self.raw_.copy()
        out["cluster"] = self.labels_
        return out

    def profile(self) -> pd.DataFrame:
        """Mean of *raw* features and size per cluster."""
        lf = self.labels_frame()
        grouped = lf.groupby("cluster")[self.feature_cols].mean()
        grouped["n_customers"] = lf.groupby("cluster").size()
        return grouped

    def relative_importance(self) -> pd.DataFrame:
        """cluster mean / population mean − 1 per feature."""
        prof = self.profile()[self.feature_cols]
        pop = self.raw_[self.feature_cols].mean()
        return (prof.div(pop) - 1).round(3)

    def snake_data(self) -> pd.DataFrame:
        """Melted normalised features for a snake plot."""
        self._require_labels()
        df = self.X_.copy()
        df["cluster"] = self.labels_
        return df.melt(
            id_vars="cluster",
            var_name="metric",
            value_name="value",
        )

    # ── internals ─────────────────────────────────────────────────────────
    def _require_X(self) -> None:
        if self.X_ is None:
            raise RuntimeError("call preprocess() first")

    def _require_labels(self) -> None:
        if self.labels_ is None:
            raise RuntimeError("call fit() first")
