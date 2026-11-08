"""
Next-purchase / churn prediction — reusable module.

Binary classifier answering: *as of a fixed landmark date, which active
customers will place another order within the next ``window_days``?*
This converts the descriptive "96.9% buy once" finding into a ranked
outreach queue with honest temporal separation:

    features  = everything observable BEFORE the landmark
    label     = any purchase in (landmark, landmark + window_days]

A landmark design (rather than "predict from first purchase") is what
makes the task learnable: recency/frequency/spend *to date* are the
genuine predictors of near-term repurchase, and using them does NOT
leak the label because the label is strictly in the future window.

Design notes
------------

* **Temporal integrity** — features are computed on orders ≤ landmark;
  labels only on orders inside the future window. No row can see its
  own outcome.
* **Eligibility** — a customer needs ≥ ``min_history_days`` of history
  before the landmark to enter train/eval (brand-new buyers have no
  behavioural signal yet). ``predict_proba`` scores everyone with any
  pre-landmark purchase.
* **Stratified split** — customers are split randomly (stratified on
  the label); since features and labels occupy disjoint time windows,
  random per-customer splits do not leak across time.
* **Ranking metrics** — ROC-AUC / PR-AUC / Brier vs. the prevalence
  baseline, because outreach prioritisation lives or dies on ranking.

Example
-------
>>> from src.churn import ChurnAnalyzer
>>> ca = ChurnAnalyzer(delivered_df).fit()
>>> scores = ca.predict_proba()          # p(buy again within 90d)
>>> ca.metrics_                         # test-set metrics + baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PAY_TYPES = ("credit_card", "boleto", "voucher", "debit_card")
TOP_STATES = 10  # one-hot only the largest states; rest -> "OTHER"


class ChurnAnalyzer:
    """Landmark-based next-purchase classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Order-line/item-level table (e.g. ``delivered_df``).
    customer_col, time_col, order_col : str
        Entity, timestamp, and order identifiers.
    value_col : str
        Value summed per order (e.g. ``payment_total_value``).
    window_days : int
        Label window: purchase within this many days AFTER the landmark
        counts as a positive. The landmark sits ``window_days`` before
        the data end, so every label window is fully observed.
    min_history_days : int
        Minimum customer tenure before the landmark for train/eval.
    test_frac : float
        Fraction of eligible customers held out (stratified).
    random_state : int
        Seed for split and classifier.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        customer_col: str = "customer_unique_id",
        time_col: str = "order_purchase_timestamp",
        order_col: str = "order_id",
        value_col: str = "payment_total_value",
        window_days: int = 90,
        min_history_days: int = 30,
        test_frac: float = 0.25,
        random_state: int = 42,
    ) -> None:
        for col in (customer_col, time_col, order_col, value_col):
            if col not in df.columns:
                raise ValueError(f"column '{col}' not in df")
        self.df = df
        self.customer_col = customer_col
        self.time_col = time_col
        self.order_col = order_col
        self.value_col = value_col
        self.window_days = window_days
        self.min_history_days = min_history_days
        self.test_frac = test_frac
        self.random_state = random_state
        self.model_ = None
        self.metrics_ = None
        self.feature_names_ = None
        self._category_stats = None
        self._state_levels = None
        self.landmark_ = None

    # ── internals ─────────────────────────────────────────────────────────
    def _order_level(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Collapse item rows to one row per order."""
        src = self.df if df is None else df
        out = src.groupby(self.order_col, sort=False).agg(
            customer=(self.customer_col, "first"),
            ts=(self.time_col, "min"),
            value=(self.value_col, "sum"),
            n_items=(self.order_col, "size"),
            category=("product_category_name", "first"),
            state=("customer_state", "first"),
            dow=("purchase_dow", "first"),
            hour=("purchase_hour", "first"),
            pay_types=("payment_types", "first"),
            installments=("payment_installments", "max"),
            delivery_days=("delivery_days", "mean"),
            is_late=("is_late", "mean"),
            review=("review_score", "mean"),
        ).rename(columns={"customer": self.customer_col})
        return out.reset_index()

    def _landmark(self, orders: pd.DataFrame) -> pd.Timestamp:
        return orders["ts"].max() - pd.Timedelta(days=self.window_days)

    def _encode(
        self,
        cust: pd.DataFrame,
        orders: pd.DataFrame,
        cat_stats: pd.DataFrame,
        fit_states: bool = False,
    ) -> pd.DataFrame:
        """Customer-history frame -> model feature matrix.

        ``cust`` holds per-customer aggregates computed on pre-landmark
        orders only. ``orders`` is the pre-landmark order table (used for
        top-state discovery on first fit).
        """
        X = cust.copy()
        X["cat_review"] = X["top_category"].map(cat_stats["cat_review"])
        X["cat_price"] = X["top_category"].map(cat_stats["cat_price"])
        for pt in PAY_TYPES:
            X[f"pay_{pt}"] = X[f"ever_{pt}"].astype(int)
        X["n_pay_types"] = X["n_pay_types"]
        X = X.drop(columns=["top_category", "state", "first_ts", "last_ts",
                            "ever_credit_card", "ever_boleto", "ever_voucher",
                            "ever_debit_card"])

        if fit_states or self._state_levels is None:
            self._state_levels = (
                orders.groupby("state")[self.order_col].nunique()
                .nlargest(TOP_STATES).index.tolist()
            )
        state_d = pd.get_dummies(
            cust["state"].where(
                cust["state"].isin(self._state_levels), "OTHER"),
            prefix="st",
        ).astype(float)
        # align columns to the fitted state set (missing -> 0, extras drop)
        for lvl in self._state_levels + ["OTHER"]:
            col = f"st_{lvl}"
            if col not in state_d.columns:
                state_d[col] = 0.0
        state_d = state_d[[f"st_{l}" for l in self._state_levels + ["OTHER"]]]

        X = pd.concat([X.reset_index(drop=True), state_d.reset_index(drop=True)],
                      axis=1)
        X.index = cust.index
        return X.rename(columns={
            "n_orders": "orders_so_far",
            "total_value": "total_spend",
            "avg_value": "avg_order_value",
            "recency_days": "days_since_last_purchase",
            "tenure_days": "customer_tenure_days",
            "n_categories": "category_diversity",
            "avg_review": "avg_review_score",
            "late_rate": "late_order_rate",
            "avg_delivery_days": "avg_delivery_days",
            "avg_installments": "avg_installments",
        })

    def _customer_features(
        self, orders: pd.DataFrame, landmark: pd.Timestamp
    ) -> pd.DataFrame:
        """Per-customer aggregates using ONLY orders ≤ landmark."""
        before = orders[orders["ts"] <= landmark]
        g = before.groupby(self.customer_col, sort=False)
        cust = g.agg(
            first_ts=("ts", "min"),
            last_ts=("ts", "max"),
            n_orders=(self.order_col, "nunique"),
            total_value=("value", "sum"),
            avg_value=("value", "mean"),
            n_categories=("category", "nunique"),
            state=("state", "first"),
            avg_review=("review", "mean"),
            late_rate=("is_late", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_installments=("installments", "mean"),
        )
        # top_category by spend: category of the customer's highest-value order
        top_order = before.loc[before.groupby(self.customer_col)["value"].idxmax()]
        cust["top_category"] = top_order.set_index(self.customer_col)["category"]
        cust["recency_days"] = (landmark - cust["last_ts"]).dt.days
        cust["tenure_days"] = (landmark - cust["first_ts"]).dt.days

        for pt in PAY_TYPES:
            cust[f"ever_{pt}"] = (
                before.assign(hit=before["pay_types"].fillna("").str.contains(pt))
                .groupby(self.customer_col)["hit"].max().astype(int)
            )
        cust["n_pay_types"] = (
            before["pay_types"].fillna("").apply(
                lambda s: len([t for t in str(s).split(",") if t.strip()]))
            .groupby(before[self.customer_col]).max()
        )
        return cust

    # ── public API ────────────────────────────────────────────────────────
    def features(self) -> tuple[pd.DataFrame, pd.Series]:
        """(X, y) for landmark-eligible customers.

        ``y`` = 1 if the customer bought again within ``window_days``
        after the landmark. Customers need ≥ ``min_history_days`` of
        pre-landmark tenure.
        """
        orders = self._order_level()
        self.landmark_ = self._landmark(orders)
        landmark = self.landmark_
        window_end = landmark + pd.Timedelta(days=self.window_days)

        cust = self._customer_features(orders, landmark)
        after = orders[(orders["ts"] > landmark) & (orders["ts"] <= window_end)]
        buyers = set(after[self.customer_col].unique())
        y = pd.Series(
            [1 if c in buyers else 0 for c in cust.index],
            index=cust.index, name="y",
        )

        cat_stats = (
            self.df.groupby("product_category_name")
            .agg(cat_review=("review_score", "mean"),
                 cat_price=("price", "mean"))
        )
        self._category_stats = cat_stats
        X = self._encode(cust, orders[orders["ts"] <= landmark], cat_stats,
                         fit_states=True)

        eligible = cust["tenure_days"] >= self.min_history_days
        return X[eligible], y[eligible]

    def fit(self) -> "ChurnAnalyzer":
        """Stratified split → train HGBT → store test metrics."""
        X, y = self.features()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=self.test_frac,
            stratify=y, random_state=self.random_state,
        )
        self.feature_names_ = list(X.columns)
        self.model_ = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06,
            early_stopping=True, validation_fraction=0.15,
            random_state=self.random_state,
        )
        self.model_.fit(Xtr, ytr)

        p = self.model_.predict_proba(Xte)[:, 1]
        self.metrics_ = {
            "roc_auc": float(roc_auc_score(yte, p)),
            "pr_auc": float(average_precision_score(yte, p)),
            "brier": float(brier_score_loss(yte, p)),
            "baseline_pr_auc": float(yte.mean()),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "positive_rate": float(yte.mean()),
            "landmark": str(self.landmark_.date()),
        }
        return self

    def predict_proba(self) -> pd.DataFrame:
        """Score every customer with pre-landmark history.

        Returns DataFrame(customer_unique_id, p_repeat, ...) — the
        probability of purchasing within ``window_days`` of the
        landmark. Scores are as-of the landmark date, not "now".
        """
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        orders = self._order_level()
        landmark = self.landmark_ or self._landmark(orders)
        before = orders[orders["ts"] <= landmark]
        cust = self._customer_features(before, landmark)
        X = self._encode(cust, before, self._category_stats)
        p = self.model_.predict_proba(X[self.feature_names_])[:, 1]
        return pd.DataFrame(
            {
                "customer_unique_id": cust.index.to_numpy(),
                "p_repeat_90d": p,
                "orders_so_far": cust["n_orders"].to_numpy(),
                "recency_days": cust["recency_days"].to_numpy(),
                "total_spend": cust["total_value"].to_numpy(),
            }
        )
