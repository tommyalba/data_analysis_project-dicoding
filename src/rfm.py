"""
RFM (Recency-Frequency-Monetary) customer segmentation — reusable module.

Turns an event-level DataFrame into entity-level behavioural metrics and
ordinal segment scores, following the standard quantile-binning approach:

    * ``datamart`` — one row per entity with ``recency``, ``frequency``,
      ``monetary`` computed over a configurable observation window
    * ``scores``   — datamart plus ordinal ``R``/``F``/``M`` scores
      (``qcut`` quantile bins), an ``RFM_profile`` string (e.g. ``'511'``)
      and a numeric ``RFM_score`` (``R + F + M``)
    * ``segment()``— map profiles/scores to business-defined named segments
      (Champions, At Risk, Lost, ...) via ordered rules

Design notes (see course reference in README / study notes):

* **Recency is directionally reversed** — score ``q`` means *most recent*,
  while Frequency and Monetary score ``q`` means *highest*. This matches
  the convention that a ``'555'`` profile is the best customer.
* **Quantile binning, not raw thresholds** — transactional features are
  typically skewed; ``qcut`` produces balanced ordinal groups. Because
  heavy ties can make quantile edges degenerate (e.g. ~97% of customers
  sharing ``frequency == 1``), each feature is scored independently and
  falls back to rank-based binning when ``qcut`` fails — see
  ``tie_policy``.
* **Composite scores are heuristic** — ``RFM_score`` sums ordinal ranks,
  it is *not* a calibrated CLV estimate, and equal scores can hide
  different profiles. The raw ``R``/``F``/``M`` columns are always kept.
* **Segments are rules, not clusters** — named segments come from
  explicit business logic, not unsupervised discovery.

Example
-------
>>> from src.rfm import RFMAnalyzer
>>> rfm = RFMAnalyzer(
...     df,
...     entity_col="customer_unique_id",
...     time_col="order_purchase_timestamp",
...     value_col="payment_total_value",
...     order_col="order_id",        # dedup to one row per order first
...     snapshot_date=None,          # default: max(time) + 1 day
...     window="365D",               # optional trailing window
...     quantiles=5,
... )
>>> rfm.datamart.head()
>>> rfm.scores.head()
>>> rfm.segment().groupby("segment").size()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Segment rules ─────────────────────────────────────────────────────────────
# Ordered (name, predicate) rules evaluated on the scores frame.
# Predicates receive a row with integer R/F/M columns.
# Mirrors the segment logic used in notebook.ipynb Step 17.

DEFAULT_RULES: list[tuple[str, callable]] = [
    ("Champions",          lambda r, f, m: r >= 4 and f >= 4 and m >= 4),
    ("Loyal Customers",    lambda r, f, m: r >= 3 and f >= 3),
    ("Recent Customers",   lambda r, f, m: r >= 4 and f <= 2),
    ("At Risk",            lambda r, f, m: r <= 2 and f >= 3),
    ("Cant Lose Them",     lambda r, f, m: r <= 2 and f <= 2 and m >= 3),
    ("Lost",               lambda r, f, m: r <= 1),
    ("Potential Loyalists", lambda r, f, m: True),
]


# ── Core class ────────────────────────────────────────────────────────────────

class RFMAnalyzer:
    """Build RFM metrics and ordinal segment scores from event data.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level data (e.g. orders, transactions, sessions).
    entity_col : str
        Column identifying the entity being scored (customer id, user id,
        subscriber id, ...).
    time_col : str
        Timestamp column of each event. Parsed with ``errors='coerce'``;
        rows with missing entity/timestamp are dropped.
    value_col : str, optional
        Column summed into ``monetary`` per entity (e.g. order value).
        If ``None``, ``monetary`` equals the event count.
    order_col : str, optional
        Column identifying the event grain (e.g. ``order_id``). When given,
        the frame is validated so each event maps to one entity/timestamp
        and deduplicated to one row per event before aggregation —
        preventing double counting when ``df`` is at a finer grain
        (e.g. order-line items).
    snapshot_date : Timestamp-like, optional
        Evaluation epoch for recency. Defaults to ``max(time_col) + 1 day``
        (the course convention: a customer who purchased "today" has
        recency 0).
    window : Timedelta/DateOffset/str, optional
        Trailing observation window (e.g. ``pd.Timedelta(days=365)``,
        ``'365D'``, ``pd.DateOffset(months=12)``). Only events at or after
        ``snapshot_date - window`` contribute to the metrics.
    quantiles : int, default 5
        Number of ordinal bins per feature (scores run 1..quantiles).
    tie_policy : {'auto', 'quantile', 'rank'}, default 'auto'
        How to handle degenerate quantile edges caused by heavy ties
        (e.g. most customers sharing ``frequency == 1``):

        * ``'quantile'`` — plain ``qcut``; may produce NaN scores or raise.
        * ``'rank'``    — ``qcut`` over ``rank(method='first')``; always
          produces balanced bins but ordering within ties is arbitrary.
          The frame is sorted by ``entity_col`` first, so results are
          deterministic for a given input.
        * ``'auto'``    — try ``quantile`` per feature; on failure fall
          back to ``rank`` for that feature with a warning. Recommended.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        value_col: str | None = None,
        order_col: str | None = None,
        snapshot_date=None,
        window=None,
        quantiles: int = 5,
        tie_policy: str = "auto",
    ) -> None:
        required = [entity_col, time_col] + ([value_col] if value_col else []) + \
                   ([order_col] if order_col else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if tie_policy not in {"auto", "quantile", "rank"}:
            raise ValueError(f"tie_policy must be 'auto'|'quantile'|'rank', got {tie_policy!r}")
        if quantiles < 2:
            raise ValueError("quantiles must be >= 2")

        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col
        self.order_col = order_col
        self.quantiles = quantiles
        self.tie_policy = tie_policy

        self._events = self._prepare_events(df)
        self.snapshot_date = self._resolve_snapshot(snapshot_date)
        self.window = window
        if window is not None:
            self._events = self._events[
                self._events[self.time_col] >= self.snapshot_date - window
            ].copy()

        self._datamart: pd.DataFrame | None = None
        self._scores: pd.DataFrame | None = None

    # ── Preparation ───────────────────────────────────────────────────────

    def _prepare_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, clean, and deduplicate events to the order grain."""
        cols = [self.entity_col, self.time_col] + \
               ([self.value_col] if self.value_col else []) + \
               ([self.order_col] if self.order_col else [])
        events = df[cols].copy()
        events[self.time_col] = pd.to_datetime(events[self.time_col], errors="coerce")
        events = events.dropna(subset=[self.entity_col, self.time_col])

        if self.value_col is not None:
            events[self.value_col] = pd.to_numeric(
                events[self.value_col], errors="coerce"
            ).fillna(0)

        if self.order_col is not None:
            consistency = (
                events.groupby(self.order_col)[[self.entity_col, self.time_col]]
                .nunique()
            )
            if (consistency > 1).any().any():
                raise ValueError(
                    f"Some {self.order_col} values map to multiple "
                    f"{self.entity_col} or {self.time_col} values."
                )
            events = events.drop_duplicates(subset=self.order_col)

        return events.sort_values(self.time_col).reset_index(drop=True)

    def _resolve_snapshot(self, snapshot_date) -> pd.Timestamp:
        if snapshot_date is None:
            return self._events[self.time_col].max() + pd.Timedelta(days=1)
        snap = pd.Timestamp(snapshot_date)
        if snap <= self._events[self.time_col].min():
            raise ValueError(
                f"snapshot_date {snap} is not after the earliest event "
                f"({self._events[self.time_col].min()})"
            )
        return snap

    # ── Metrics ───────────────────────────────────────────────────────────

    def _build_datamart(self) -> pd.DataFrame:
        ev = self._events
        named = {
            "recency": (self.time_col,
                        lambda s: (self.snapshot_date - s.max()).days),
            "frequency": (self.order_col, "nunique") if self.order_col
                          else (self.time_col, "count"),
            "monetary": (self.value_col, "sum") if self.value_col
                        else (self.time_col, "count"),
        }
        dm = ev.groupby(self.entity_col).agg(**named)
        dm.index.name = self.entity_col
        return dm.sort_index()

    @property
    def datamart(self) -> pd.DataFrame:
        """Entity-level R/F/M metrics (index = ``entity_col``)."""
        if self._datamart is None:
            self._datamart = self._build_datamart()
        return self._datamart

    # ── Ordinal scoring ───────────────────────────────────────────────────

    def _score(self, series: pd.Series, ascending: bool) -> pd.Series:
        """Quantile-bin ``series`` into 1..quantiles with tie handling."""
        q = self.quantiles
        labels = list(range(1, q + 1)) if ascending else list(range(q, 0, -1))

        def _qcut(s):
            return pd.qcut(s, q, labels=labels)

        def _rank_qcut(s):
            return pd.qcut(s.rank(method="first"), q, labels=labels)

        policy = self.tie_policy
        if policy == "quantile":
            return _qcut(series)
        if policy == "rank":
            return _rank_qcut(series)

        # auto: try quantile, fall back to rank on degenerate edges
        try:
            scored = _qcut(series)
            if scored.isna().any():
                raise ValueError("qcut produced NaN bins")
            return scored
        except (ValueError, TypeError) as exc:
            warnings.warn(
                f"qcut failed for a feature ({exc}); falling back to "
                f"rank-based binning. Scores within tied values are "
                f"arbitrary — interpret with care.",
                UserWarning,
                stacklevel=2,
            )
            return _rank_qcut(series)

    @property
    def scores(self) -> pd.DataFrame:
        """``datamart`` + ordinal ``R``/``F``/``M``, ``RFM_profile`` and
        numeric ``RFM_score`` (R + F + M)."""
        if self._scores is None:
            dm = self.datamart
            out = dm.copy()
            out["R"] = self._score(dm["recency"], ascending=False)   # 5 = most recent
            out["F"] = self._score(dm["frequency"], ascending=True) # 5 = most frequent
            out["M"] = self._score(dm["monetary"], ascending=True)  # 5 = highest spend
            for c in ("R", "F", "M"):
                out[c] = out[c].astype("Int64")
            out["RFM_profile"] = (
                out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
            )
            out["RFM_score"] = out[["R", "F", "M"]].sum(axis=1, skipna=False)
            self._scores = out
        return self._scores

    # ── Segmentation ──────────────────────────────────────────────────────

    def segment(
        self,
        rules: list[tuple[str, callable]] | None = None,
        name: str = "segment",
    ) -> pd.DataFrame:
        """Return ``scores`` with a named-segment column appended.

        ``rules`` is an ordered list of ``(segment_name, predicate)`` where
        ``predicate(r, f, m)`` receives the integer R/F/M scores and returns
        bool. The first matching rule wins; unmatched rows become
        ``'Unassigned'`` unless the last rule is a catch-all (as in
        ``DEFAULT_RULES``). Defaults to :data:`DEFAULT_RULES`.
        """
        rules = DEFAULT_RULES if rules is None else rules
        df = self.scores.copy()

        r, f, m = df["R"], df["F"], df["M"]
        labels = np.full(len(df), "Unassigned", dtype=object)
        for seg_name, pred in rules:
            mask = np.array([pred(ri, fi, mi) for ri, fi, mi in zip(r, f, m)])
            labels[mask & (labels == "Unassigned")] = seg_name
        df[name] = labels
        return df

    def summary_by_segment(self, name: str = "segment") -> pd.DataFrame:
        """Mean R/F/M + size per named segment (sorted by monetary desc)."""
        seg = self.segment(name=name)
        agg = seg.groupby(name).agg(
            customers=("recency", "size"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
        ).round(1).sort_values("avg_monetary", ascending=False)
        return agg

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Headline diagnostics."""
        dm = self.datamart
        return {
            "snapshot_date": self.snapshot_date,
            "window": str(self.window) if self.window is not None else None,
            "n_entities": int(len(dm)),
            "n_events": int(len(self._events)),
            "recency_min": int(dm["recency"].min()),
            "recency_max": int(dm["recency"].max()),
            "frequency_mean": float(dm["frequency"].mean()),
            "monetary_total": float(dm["monetary"].sum()),
        }
