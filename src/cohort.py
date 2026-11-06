"""
Cohort retention analysis — reusable module.

Provides a single entry point (:class:`CohortAnalyzer`) that turns an
event-level DataFrame into the standard cohort artefacts:

    * ``counts``     — unique entities per (cohort, period) cell
    * ``retention``  — counts normalised by cohort acquisition size;
                       NaN marks periods that are not yet observable
    * ``sizes``      — period-0 (acquisition) counts per cohort
    * ``eligible_*`` — matrices restricted to cohorts with at least
                       ``min_cohort_size`` entities at acquisition
    * ``diagnostics``— macro / weighted / best retention per period
    * ``balanced_curve`` — macro & size-weighted average retention over
                       cohorts with enough follow-up history
    * ``period_retention(n)`` — retention at a single follow-up period

The module is deliberately dataset-agnostic: any DataFrame with an entity
identifier column and a timestamp column can be analysed, at any pandas
period frequency (``'D'``, ``'W'``, ``'M'``, ``'Q'``, ``'Y'``, ...).
Typical use cases include customer repeat-purchase cohorts, user
return/retention cohorts, session-based engagement cohorts, and
subscription renewal cohorts.

Example
-------
>>> from src.cohort import CohortAnalyzer
>>> ca = CohortAnalyzer(
...     df,
...     customer_col="customer_unique_id",
...     time_col="order_purchase_timestamp",
...     order_col="order_id",      # optional: enforces 1 event per order
...     freq="M",
...     min_cohort_size=50,
... )
>>> ca.counts.shape, ca.retention.shape
>>> ca.diagnostics(periods=[1, 3, 6])
>>> ca.balanced_curve(max_period=6)["macro"]
>>> ca.period_retention(1)          # Month-1 retention per cohort
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class BalancedCurve:
    """Retention curve over cohorts with sufficient observation history."""
    cohorts: pd.Index                 # cohort periods used
    retention: pd.DataFrame           # eligible_retention subset
    counts: pd.DataFrame              # eligible_counts subset
    sizes: pd.Series                  # period-0 counts for those cohorts
    macro: pd.Series                  # unweighted mean retention per period
    weighted: pd.Series               # size-weighted retention per period


@dataclass
class CohortResult:
    """Bundle returned by :meth:`CohortAnalyzer.summary`."""
    n_events: int
    n_entities: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    n_cohorts: int
    n_eligible: int


# ── Core class ────────────────────────────────────────────────────────────────

class CohortAnalyzer:
    """Build and analyse entity-level cohort retention tables.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level data. May contain multiple rows per entity per period
        (e.g. several orders in one month); entities are counted with
        ``nunique`` so duplicates collapse naturally.
    customer_col : str
        Column identifying the entity being cohorted (customer id, user id,
        subscriber id, ...). Must be unique per real-world entity — e.g. a
        ``customer_unique_id`` rather than an order-level id.
    time_col : str
        Timestamp column of the event. Parsed with ``errors='coerce'``;
        rows with missing entity/timestamp values are dropped.
    freq : str, default 'M'
        Period frequency for cohorting and period indexing. Any freq
        accepted by :meth:`pandas.Series.dt.to_period` (``'D'``, ``'W'``,
        ``'M'``, ``'Q'``, ``'Y'``, ...).
    order_col : str, optional
        Column identifying the event grain (e.g. ``order_id``). When given,
        the frame is validated so that each order maps to exactly one
        entity and one timestamp, then deduplicated to one row per order.
        Pass ``None`` when events are already unique per row.
    min_cohort_size : int, default 50
        Cohorts whose period-0 entity count is below this threshold are
        excluded from ``eligible_*`` artefacts and downstream statistics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        customer_col: str,
        time_col: str,
        freq: str = "M",
        order_col: str | None = None,
        min_cohort_size: int = 50,
    ) -> None:
        # Validate frequency early — to_period raises otherwise
        pd.tseries.frequencies.to_offset(freq)

        required = [customer_col, time_col] + ([order_col] if order_col else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        self.customer_col = customer_col
        self.time_col = time_col
        self.order_col = order_col
        self.freq = freq
        self.min_cohort_size = min_cohort_size

        self._base = self._build_base(df)
        self._counts: pd.DataFrame | None = None
        self._retention: pd.DataFrame | None = None
        self._eligible_counts: pd.DataFrame | None = None
        self._eligible_retention: pd.DataFrame | None = None

    # ── Construction ──────────────────────────────────────────────────────

    def _build_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean events and attach cohort/period columns."""
        cols = [self.customer_col, self.time_col] + ([self.order_col] if self.order_col else [])
        base = df[cols].copy()

        base[self.time_col] = pd.to_datetime(base[self.time_col], errors="coerce")
        base = base.dropna(subset=[self.customer_col, self.time_col])

        if self.order_col is not None:
            # Enforce one entity + one timestamp per event id
            consistency = (
                base.groupby(self.order_col)[[self.customer_col, self.time_col]]
                .nunique()
            )
            if (consistency > 1).any().any():
                raise ValueError(
                    f"Some {self.order_col} values map to multiple "
                    f"{self.customer_col} or {self.time_col} values."
                )
            base = base.drop_duplicates(subset=self.order_col)

        base = base.sort_values(self.time_col).reset_index(drop=True)

        order_month = base[self.time_col].dt.to_period(self.freq)
        base["order_period"] = order_month
        base["cohort_period"] = (
            base.groupby(self.customer_col)["order_period"].transform("min")
        )
        # Period ordinals make the period index frequency-agnostic
        base["period_index"] = (
            base["order_period"].astype("int64")
            - base["cohort_period"].astype("int64")
        )
        if (base["period_index"] < 0).any():
            raise AssertionError("Negative period_index detected")
        return base

    # ── Matrices ──────────────────────────────────────────────────────────

    @property
    def base(self) -> pd.DataFrame:
        """Event-level frame with ``order_period``, ``cohort_period``,
        ``period_index`` columns."""
        return self._base

    @property
    def counts(self) -> pd.DataFrame:
        """Unique entities per (cohort, period); Int64, NaN = unobserved."""
        if self._counts is None:
            counts = (
                self._base.groupby(["cohort_period", "period_index"])
                [self.customer_col]
                .nunique()
                .unstack()
            )
            max_period = int(self._base["period_index"].max())
            counts = counts.reindex(columns=range(max_period + 1))

            # A cell is "observed" iff its period has fully elapsed by the
            # latest observed event period; unobserved cells stay NaN.
            max_ord = int(self._base["order_period"].astype("int64").max())
            last_observed = max_ord - counts.index.astype("int64")
            observed = (
                counts.columns.to_numpy()[None, :]
                <= last_observed.to_numpy()[:, None]
            )
            counts = counts.mask(observed & counts.isna(), 0).astype("Int64")
            self._counts = counts
        return self._counts

    @property
    def retention(self) -> pd.DataFrame:
        """Retention rate per cell (0–1); NaN where unobserved or size 0."""
        if self._retention is None:
            self._retention = self.counts.divide(self.sizes, axis=0)
        return self._retention

    @property
    def sizes(self) -> pd.Series:
        """Period-0 entity count per cohort (acquisition size)."""
        return self.counts[0]

    @property
    def max_observed_period(self) -> pd.Period:
        return self._base["order_period"].max()

    @property
    def eligible_counts(self) -> pd.DataFrame:
        """``counts`` restricted to cohorts with >= min_cohort_size."""
        if self._eligible_counts is None:
            keep = self.sizes[self.sizes >= self.min_cohort_size].index
            self._eligible_counts = self.counts.loc[keep]
        return self._eligible_counts

    @property
    def eligible_retention(self) -> pd.DataFrame:
        """``retention`` restricted to eligible cohorts."""
        if self._eligible_retention is None:
            self._eligible_retention = self.retention.loc[self.eligible_counts.index]
        return self._eligible_retention

    # ── Diagnostics & curves ──────────────────────────────────────────────

    def diagnostics(
        self, periods: list[int] | tuple[int, ...] = (1, 3, 6)
    ) -> pd.DataFrame:
        """Per-period retention summary over eligible cohorts.

        For each requested period reports the number of cohorts with an
        observed cell, the macro-average (unweighted mean) retention, the
        size-weighted retention, and the best single-cohort retention —
        all in percent. Periods beyond the observed range are skipped.
        """
        ec, er = self.eligible_counts, self.eligible_retention
        rows = []
        for period in periods:
            if period not in ec.columns:
                continue
            observed = ec[period].dropna()
            if observed.empty:
                continue
            period_ret = er.loc[observed.index, period]
            rows.append({
                "Period": f"Period-{period}",
                "Observed Cohorts": len(observed),
                "Macro Retention (%)": period_ret.mean() * 100,
                "Weighted Retention (%)": (
                    observed.sum()
                    / ec.loc[observed.index, 0].sum()
                    * 100
                ),
                "Best Cohort (%)": period_ret.max() * 100,
            })
        return pd.DataFrame(rows).set_index("Period") if rows else pd.DataFrame()

    def balanced_curve(self, max_period: int = 6) -> BalancedCurve:
        """Average retention curve over cohorts with >= ``max_period``
        follow-up months of observation history (and eligible size).

        Returns macro (unweighted) and size-weighted averages so small and
        large cohorts can be compared; the weighted series is the one to
        quote when cohort sizes differ materially.
        """
        cutoff = self.max_observed_period - max_period
        cohorts = self.eligible_counts.index[self.eligible_counts.index <= cutoff]
        if len(cohorts) == 0:
            raise ValueError(
                f"No eligible cohort has {max_period} periods of history; "
                f"reduce max_period."
            )
        periods = [c for c in range(max_period + 1) if c in self.counts.columns]
        ret = self.eligible_retention.loc[cohorts, periods]
        cnt = self.eligible_counts.loc[cohorts, periods]
        sizes = self.eligible_counts.loc[cohorts, 0]
        return BalancedCurve(
            cohorts=cohorts,
            retention=ret,
            counts=cnt,
            sizes=sizes,
            macro=ret.mean(axis=0) * 100,
            weighted=(cnt.sum(axis=0) / sizes.sum()) * 100,
        )

    def period_retention(
        self, period: int, min_cohort_size: int | None = None
    ) -> pd.Series:
        """Retention at a single follow-up ``period`` per eligible cohort
        (percent). Pass ``min_cohort_size`` to override the instance
        threshold for this call only."""
        if min_cohort_size is None or min_cohort_size == self.min_cohort_size:
            ret = self.eligible_retention
        else:
            keep = self.sizes[self.sizes >= min_cohort_size].index
            ret = self.retention.loc[keep]
        if period not in ret.columns:
            raise KeyError(f"Period {period} not in retention table")
        return ret[period].dropna() * 100

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> CohortResult:
        """Headline diagnostics of the built cohort base."""
        return CohortResult(
            n_events=len(self._base),
            n_entities=int(self._base[self.customer_col].nunique()),
            date_min=self._base[self.time_col].min(),
            date_max=self._base[self.time_col].max(),
            n_cohorts=int(self.counts.shape[0]),
            n_eligible=int(self.eligible_counts.shape[0]),
        )
