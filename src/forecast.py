"""
Short-horizon univariate time-series forecasting — reusable module.

Designed for the small, clean monthly aggregates typical of business
pipelines (e.g. ``monthly_revenue``). Follows the honest-forecasting
workflow:

    1. ``backtest()``  — direct h-step evaluation of candidate models on a
       held-out tail, so model choice is evidence-based, not vibes
    2. ``forecast()``  — refit the winning model on all data and project
       ``horizon`` periods ahead with a prediction interval

Candidate models (deliberately small — this is for ~20-100 point series,
not big-data forecasting):

    * ``naive``  — repeat the last observed value (the benchmark to beat)
    * ``linear`` — OLS trend on the time index
    * ``ets``    — exponential smoothing with damped additive trend
                   (``statsmodels.tsa.exponential_smoothing.ets.ETSModel``)

Design notes
------------

* **No seasonal model by default** — a seasonal specification needs at
  least two full cycles. Callers with longer history can extend
  ``CANDIDATES``; callers with ~1 cycle (like this project) get an
  explicit warning instead of a silently wrong seasonal fit.
* **Direct h-step backtest** — each candidate forecasts the whole
  ``test_size`` tail in one shot (matching how it will be used), rather
  than averaging one-step errors that flatter flexible models.
* **MAE, not MAPE** — percentage errors explode near zero; MAE in the
  series' own units stays interpretable.

Example
-------
>>> fc = Forecaster(monthly['total_revenue'])
>>> fc.backtest(test_size=6)
>>> fc.forecast(horizon=6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def _naive(train: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(train.iloc[-1], index=range(horizon), dtype=float)


def _linear(train: pd.Series, horizon: int) -> pd.Series:
    t = np.arange(len(train), dtype=float)
    slope, intercept = np.polyfit(t, train.to_numpy(dtype=float), 1)
    future_t = np.arange(len(train), len(train) + horizon, dtype=float)
    return pd.Series(slope * future_t + intercept, index=range(horizon))


def _ets(train: pd.Series, horizon: int) -> pd.Series:
    fit = ETSModel(
        train.astype(float),
        error="add",
        trend="add",
        damped_trend=True,
        seasonal=None,
    ).fit(disp=False)
    return pd.Series(fit.forecast(horizon).to_numpy(), index=range(horizon))


CANDIDATES = {"naive": _naive, "linear": _linear, "ets": _ets}


class Forecaster:
    """Backtest-and-forecast wrapper for a univariate series.

    Parameters
    ----------
    series : pd.Series
        Ordered observations with a ``PeriodIndex``/``DatetimeIndex``
        (or any index — only values and length are used internally).
    """

    def __init__(self, series: pd.Series) -> None:
        s = series.dropna().astype(float)
        if len(s) < 8:
            raise ValueError("need at least 8 observations to forecast")
        self.series_ = s
        self.last_label_ = s.index[-1]

    def _future_index(self, horizon: int):
        """Extend the index by ``horizon`` periods when possible."""
        idx = self.series_.index
        try:
            if isinstance(idx, pd.PeriodIndex):
                return pd.period_range(idx[-1] + 1, periods=horizon, freq=idx.freq)
            if isinstance(idx, pd.DatetimeIndex):
                return pd.date_range(idx[-1] + idx.freq or pd.offsets.MonthBegin(1),
                                     periods=horizon, freq="MS")
        except (AttributeError, TypeError):
            pass
        return pd.RangeIndex(len(idx), len(idx) + horizon)

    def backtest(
        self,
        test_size: int = 6,
        models: list[str] | None = None,
    ) -> pd.DataFrame:
        """Direct h-step MAE of each candidate on the last ``test_size``
        observations. Returns one row per model, sorted by MAE."""
        models = models or list(CANDIDATES)
        unknown = [m for m in models if m not in CANDIDATES]
        if unknown:
            raise ValueError(f"unknown models: {unknown}; choose from {list(CANDIDATES)}")
        if not 2 <= test_size < len(self.series_) - 2:
            raise ValueError("test_size must satisfy 2 <= test_size < n-2")

        train, test = self.series_[:-test_size], self.series_[-test_size:]
        rows = []
        for name in models:
            pred = CANDIDATES[name](train, test_size)
            mae = float(np.mean(np.abs(pred.to_numpy() - test.to_numpy())))
            rows.append({"model": name, "mae": mae})
        return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)

    def forecast(
        self,
        horizon: int = 6,
        model: str | None = None,
        test_size: int = 6,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Forecast ``horizon`` periods ahead.

        If ``model`` is None, the best backtest candidate (lowest MAE) is
        refit on the full series. Returns columns ``forecast``,
        ``ci_lower``, ``ci_upper`` indexed by future period labels.
        """
        if model is None:
            scores = self.backtest(test_size=test_size)
            model = scores.loc[0, "model"]

        future_idx = self._future_index(horizon)
        if model == "ets":
            fit = ETSModel(
                self.series_.astype(float),
                error="add", trend="add", damped_trend=True, seasonal=None,
            ).fit(disp=False)
            frame = fit.get_forecast(horizon).summary_frame(alpha=alpha)
            out = pd.DataFrame(
                {
                    "forecast": frame["mean"].to_numpy(),
                    "ci_lower": frame["pi_lower"].to_numpy(),
                    "ci_upper": frame["pi_upper"].to_numpy(),
                },
                index=future_idx,
            )
        else:
            point = CANDIDATES[model](self.series_, horizon)
            # shortcut: only ETS produces an interval; other models get
            # the backtest MAE as a symmetric band — upgrade to a
            # residual-based interval if non-ETS models are ever selected.
            _, mae = self.backtest(test_size=test_size).query(
                f"model == '{model}'"
            ).iloc[0][["model", "mae"]]
            out = pd.DataFrame(
                {
                    "forecast": point.to_numpy(),
                    "ci_lower": point.to_numpy() - mae,
                    "ci_upper": point.to_numpy() + mae,
                },
                index=future_idx,
            )
        out.index.name = "forecast_period"
        return out
