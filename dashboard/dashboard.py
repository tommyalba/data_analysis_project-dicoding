"""
Olist E-Commerce Analytics Dashboard
=====================================
Run from the project root:
    streamlit run dashboard/dashboard.py

Requires:  streamlit, pandas, plotly, pyarrow  (all in venv)
Data:      dashboard/data/*.parquet  (exported from notebook.ipynb)
"""

import os
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve data dir relative to this file so the app works from any cwd
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    monthly  = pd.read_parquet(f"{DATA_DIR}/monthly_revenue.parquet")
    category = pd.read_parquet(f"{DATA_DIR}/category_stats.parquet")
    delivery = pd.read_parquet(f"{DATA_DIR}/delivery_stats.parquet")
    state    = pd.read_parquet(f"{DATA_DIR}/state_revenue.parquet")
    review   = pd.read_parquet(f"{DATA_DIR}/review_delivery.parquet")
    rfm      = pd.read_parquet(f"{DATA_DIR}/rfm.parquet")
    cohort   = pd.read_parquet(f"{DATA_DIR}/cohort_retention.parquet")
    cohort_sizes = pd.read_parquet(f"{DATA_DIR}/cohort_sizes.parquet")
    delivered = pd.read_parquet(
        f"{DATA_DIR}/delivered_df.parquet",
        columns=[
            "order_id", "customer_unique_id", "order_purchase_timestamp",
            "purchase_year_month", "purchase_year", "purchase_month_num",
            "purchase_dow", "purchase_hour",
            "product_category_name", "customer_state", "seller_state",
            "payment_total_value", "payment_types", "payment_installments",
            "delivery_days", "delivery_delay_days", "is_late",
            "review_score", "revenue", "price", "freight_value",
        ],
    )
    return monthly, category, delivery, state, review, rfm, cohort, cohort_sizes, delivered


monthly, category, delivery, state, review, rfm, cohort, cohort_sizes, delivered = load_data()

# ── Colour palette ────────────────────────────────────────────────────────────
PRIMARY   = "#4C72B0"
SECONDARY = "#DD8452"
TEMPLATE  = "plotly_white"

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Olist_logo.png/640px-Olist_logo.png",
        width=160,
    )
    st.markdown("## Olist E-Commerce")
    st.markdown("**Brazilian marketplace · Oct 2016 – Aug 2018**")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Revenue & Trends",
            "Product Categories",
            "Delivery Performance",
            "Geographic Analysis",
            "Payment Behaviour",
            "RFM Segmentation",
            "Cohort Retention",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: Olist Public Dataset via Kaggle")

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL KPI STRIP  (shown on every page)
# ═════════════════════════════════════════════════════════════════════════════
total_rev = (
    delivered.drop_duplicates(subset=["order_id"])["payment_total_value"].sum()
)
total_orders = delivered["order_id"].nunique()
total_cust   = delivered["customer_unique_id"].nunique()
avg_review   = delivered.drop_duplicates("order_id")["review_score"].mean()
late_rate    = delivered.drop_duplicates("order_id")["is_late"].mean() * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue",   f"BRL {total_rev:,.0f}")
k2.metric("Total Orders",    f"{total_orders:,}")
k3.metric("Unique Customers", f"{total_cust:,}")
k4.metric("Avg Review Score", f"{avg_review:.2f} / 5")
k5.metric("Late Delivery Rate", f"{late_rate:.1f}%")
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — REVENUE & TRENDS
# ═════════════════════════════════════════════════════════════════════════════
if page == "Revenue & Trends":
    st.header("Revenue & Order Volume Trends")
    st.markdown(
        "_How has monthly revenue and order volume evolved, "
        "and is there a seasonal pattern?_"
    )

    # ── Dual-axis monthly chart ───────────────────────────────────────────
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["purchase_year_month"],
            y=monthly["total_revenue"],
            name="Total Revenue (BRL)",
            marker_color=PRIMARY,
            opacity=0.82,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["purchase_year_month"],
            y=monthly["orders"],
            name="Order Count",
            mode="lines+markers",
            line=dict(color=SECONDARY, width=3),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )
    # Annotate Black Friday
    nov17 = monthly[monthly["purchase_year_month"] == "2017-11"]
    if not nov17.empty:
        fig.add_annotation(
            x="2017-11",
            y=float(nov17["total_revenue"].values[0]),
            text="Black Friday<br>Nov 2017",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=-50,
            font=dict(size=11, color="gray"),
        )
    fig.update_layout(
        title="Monthly Revenue & Order Volume",
        template=TEMPLATE,
        legend=dict(orientation="h", y=1.12),
        height=420,
        xaxis_tickangle=-45,
    )
    fig.update_yaxes(title_text="Total Revenue (BRL)", secondary_y=False)
    fig.update_yaxes(title_text="Order Count",          secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── Avg order value trend ────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.line(
            monthly, x="purchase_year_month", y="avg_order_value",
            title="Average Order Value per Month",
            markers=True,
            color_discrete_sequence=[PRIMARY],
            template=TEMPLATE,
            labels={"purchase_year_month": "Month",
                    "avg_order_value": "Avg Order Value (BRL)"},
        )
        fig2.update_layout(height=320, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        # Purchase hour heatmap
        hour_dow = (
            delivered
            .drop_duplicates("order_id")
            .groupby(["purchase_dow", "purchase_hour"])["order_id"]
            .nunique()
            .reset_index(name="orders")
        )
        dow_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        fig3 = px.density_heatmap(
            hour_dow,
            x="purchase_hour", y="purchase_dow",
            z="orders",
            color_continuous_scale="Blues",
            category_orders={"purchase_dow": dow_order},
            title="Order Count by Hour × Day of Week",
            labels={"purchase_hour": "Hour", "purchase_dow": "Day", "orders": "Orders"},
            template=TEMPLATE,
        )
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Insight callout ───────────────────────────────────────────────────
    peak = monthly.loc[monthly["total_revenue"].idxmax()]
    peak_hour = int(hour_dow.groupby("purchase_hour")["orders"].sum().idxmax())
    st.info(
        f"**Peak month:** {peak['purchase_year_month']} — "
        f"BRL {peak['total_revenue']:,.0f} with {peak['orders']:,.0f} orders.  "
        f"Peak purchase hour: **{peak_hour:02d}:00** (weekday afternoons dominate)."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRODUCT CATEGORIES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Product Categories":
    st.header("Product Categories — Revenue vs Satisfaction")
    st.markdown(
        "_Which categories generate the most revenue, "
        "and do the top-revenue categories also achieve a mean review "
        "score of at least 4.0 out of 5?_"
    )

    n_cats = st.slider("Number of categories to show", 5, 30, 15)
    top = category.head(n_cats).copy()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            top.sort_values("total_revenue"),
            x="total_revenue", y="product_category_name",
            orientation="h",
            color="avg_review",
            color_continuous_scale="RdYlGn",
            range_color=[3.5, 4.8],
            labels={
                "total_revenue": "Total Revenue (BRL)",
                "product_category_name": "Category",
                "avg_review": "Avg Review",
            },
            title=f"Top {n_cats} Categories by Revenue",
            template=TEMPLATE,
        )
        fig.update_layout(height=500, coloraxis_colorbar_title="Review")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            top,
            x="avg_review", y="total_revenue",
            size="total_orders",
            color="avg_review",
            text="product_category_name",
            color_continuous_scale="RdYlGn",
            range_color=[3.5, 4.8],
            labels={
                "avg_review": "Avg Review Score",
                "total_revenue": "Total Revenue (BRL)",
                "total_orders": "Order Count",
            },
            title="Revenue vs Satisfaction (bubble = order volume)",
            template=TEMPLATE,
        )
        fig2.update_traces(textposition="top center", textfont_size=8)
        fig2.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Full category table ───────────────────────────────────────────────
    with st.expander("Full Category Table"):
        st.dataframe(
            category.rename(columns={
                "product_category_name": "Category",
                "total_revenue": "Revenue (BRL)",
                "total_orders":  "Orders",
                "avg_price":     "Avg Price (BRL)",
                "avg_review":    "Avg Review",
            }).style.format({
                "Revenue (BRL)": "{:,.0f}",
                "Avg Price (BRL)": "{:.2f}",
                "Avg Review": "{:.2f}",
            }),
            use_container_width=True,
            height=400,
        )

    top1 = category.iloc[0]
    best_review = category[category["total_orders"] >= 500].nlargest(1, "avg_review").iloc[0]
    st.info(
        f"**Highest revenue:** `{top1['product_category_name']}` — "
        f"BRL {top1['total_revenue']:,.0f} (review: {top1['avg_review']:.2f}).  "
        f"**Highest satisfaction (≥500 orders):** `{best_review['product_category_name']}` — "
        f"avg score {best_review['avg_review']:.2f}."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DELIVERY PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Delivery Performance":
    st.header("Delivery Performance vs Customer Satisfaction")
    st.markdown(
        "_How does on-time delivery vary by state, "
        "and does being late directly hurt review scores?_"
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            delivery.sort_values("late_rate", ascending=False),
            x="customer_state", y="late_rate",
            color="avg_delivery_days",
            color_continuous_scale="RdYlGn_r",
            labels={
                "customer_state": "State",
                "late_rate": "Late Delivery Rate",
                "avg_delivery_days": "Avg Days",
            },
            title="Late Delivery Rate by State",
            template=TEMPLATE,
        )
        fig.update_layout(
            height=380,
            yaxis_tickformat=".0%",
            coloraxis_colorbar_title="Avg<br>Days",
        )
        fig.add_hline(
            y=delivery["late_rate"].mean(),
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Avg {delivery['late_rate'].mean()*100:.1f}%",
            annotation_position="bottom right",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            delivery,
            x="avg_delivery_days", y="avg_review_score",
            size="orders",
            color="late_rate",
            text="customer_state",
            color_continuous_scale="RdYlGn_r",
            labels={
                "avg_delivery_days": "Avg Delivery Days",
                "avg_review_score":  "Avg Review Score",
                "late_rate":         "Late Rate",
            },
            title="Review Score vs Avg Delivery Days per State",
            template=TEMPLATE,
        )
        fig2.update_traces(textposition="top center", textfont_size=8)
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Review score by delivery delay bucket ────────────────────────────
    st.subheader("Review Score Distribution by Delivery Timing")

    delay_df = (
        delivered
        .drop_duplicates("order_id")
        .dropna(subset=["review_score", "delivery_delay_days"])
        .copy()
    )
    bins   = [-999, -10, -5, 0, 5, 10, 999]
    labels = ["Early >10d", "Early 5-10d", "Early 0-5d",
              "Late 0-5d",  "Late 5-10d",  "Late >10d"]
    delay_df["delay_bucket"] = pd.cut(
        delay_df["delivery_delay_days"], bins=bins, labels=labels
    )
    bucket_dist = (
        delay_df
        .groupby("delay_bucket", observed=True)["review_score"]
        .value_counts(normalize=True)
        .rename("pct")
        .reset_index()
    )
    fig3 = px.bar(
        bucket_dist,
        x="delay_bucket", y="pct", color="review_score",
        barmode="stack",
        color_discrete_sequence=px.colors.diverging.RdBu,
        labels={
            "delay_bucket":  "Delivery Timing",
            "pct":           "Share of Reviews",
            "review_score":  "Score",
        },
        title="Review Score Share by Delivery Delay Bucket",
        template=TEMPLATE,
    )
    fig3.update_layout(height=340, yaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    worst  = delivery.iloc[0]
    fastest = delivery.sort_values("avg_delivery_days").iloc[0]
    late_orders = delay_df[delay_df["delivery_delay_days"] > 0]["review_score"].mean()
    early_orders = delay_df[delay_df["delivery_delay_days"] <= 0]["review_score"].mean()
    st.info(
        f"**Worst state:** {worst['customer_state']} — "
        f"{worst['late_rate']*100:.1f}% late, avg {worst['avg_delivery_days']:.1f} days.  "
        f"**Fastest:** {fastest['customer_state']} — {fastest['avg_delivery_days']:.1f} days.  "
        f"Late delivery drops review score by **{early_orders - late_orders:.2f} pts** "
        f"({early_orders:.2f} → {late_orders:.2f})."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GEOGRAPHIC ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Geographic Analysis":
    st.header("Geographic Distribution of Revenue")
    st.markdown(
        "_Where are customers and revenue concentrated, "
        "and which states are under-served?_"
    )

    c1, c2 = st.columns([2, 1])

    with c1:
        fig = px.scatter_geo(
            state,
            lat="avg_lat", lon="avg_lng",
            size="total_revenue",
            color="total_revenue",
            color_continuous_scale="Blues",
            hover_name="customer_state",
            hover_data={
                "total_orders":     True,
                "total_customers":  True,
                "total_revenue":    ":,.0f",
                "avg_lat":  False,
                "avg_lng":  False,
            },
            scope="south america",
            title="Revenue Bubble Map — Brazilian States",
            template=TEMPLATE,
        )
        fig.update_geos(
            fitbounds="locations",
            showland=True, landcolor="#f0f0f0",
            showcoastlines=True, coastlinecolor="gray",
        )
        fig.update_layout(height=520, coloraxis_colorbar_title="Revenue<br>(BRL)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        total = state["total_revenue"].sum()
        state_pct = state.copy()
        state_pct["share_pct"] = (state_pct["total_revenue"] / total * 100).round(2)
        fig2 = px.bar(
            state_pct.sort_values("total_revenue", ascending=True).tail(10),
            x="total_revenue", y="customer_state",
            orientation="h",
            color="share_pct",
            color_continuous_scale="Blues",
            labels={
                "total_revenue": "Revenue (BRL)",
                "customer_state": "State",
                "share_pct": "Share %",
            },
            title="Top 10 States by Revenue",
            template=TEMPLATE,
        )
        fig2.update_layout(height=520, coloraxis_colorbar_title="Share %")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Treemap ───────────────────────────────────────────────────────────
    fig3 = px.treemap(
        state,
        path=["customer_state"],
        values="total_revenue",
        color="total_revenue",
        color_continuous_scale="Blues",
        title="Revenue Share Treemap by State",
        template=TEMPLATE,
    )
    fig3.update_traces(textinfo="label+percent root")
    fig3.update_layout(height=360)
    st.plotly_chart(fig3, use_container_width=True)

    sp_share = (
        state[state["customer_state"] == "SP"]["total_revenue"].values[0] / total * 100
    )
    sp_cust_share = (
        state[state["customer_state"] == "SP"]["total_customers"].values[0]
        / state["total_customers"].sum() * 100
    )
    top3 = state.head(3)["total_revenue"].sum() / total * 100
    small = (state["total_revenue"] / total < 0.01).sum()
    st.info(
        f"**SP** accounts for **{sp_share:.1f}%** of revenue and "
        f"**{sp_cust_share:.1f}%** of customers.  "
        f"Top 3 states (SP, RJ, MG) = **{top3:.1f}%** combined revenue.  "
        f"**{small} states** each contribute less than 1%."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PAYMENT BEHAVIOUR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Payment Behaviour":
    st.header("Payment Behaviour")
    st.markdown(
        "_What payment types dominate, and do customers paying in 12 "
        "installments spend more than single-payment buyers?_"
    )

    orders_pay = delivered.drop_duplicates("order_id").copy()

    # ── Payment type donut ────────────────────────────────────────────────
    pay_expanded = (
        orders_pay["payment_types"]
        .str.split(", ").explode()
        .value_counts()
        .reset_index()
    )
    pay_expanded.columns = ["payment_type", "count"]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            pay_expanded,
            names="payment_type", values="count",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Payment Type Distribution",
            template=TEMPLATE,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Installment distribution for credit card orders
        cc_df = orders_pay[
            (orders_pay["payment_types"] == "credit_card") &
            (orders_pay["payment_installments"] <= 12)
        ].copy()
        cc_df["payment_installments"] = cc_df["payment_installments"].astype(int)
        inst_sum = (
            cc_df
            .groupby("payment_installments", as_index=False)
            .agg(
                order_count=("order_id",           "nunique"),
                avg_value  =("payment_total_value", "mean"),
            )
        )
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Orders per Installment", "Avg Value per Installment"),
        )
        fig2.add_trace(
            go.Bar(x=inst_sum["payment_installments"], y=inst_sum["order_count"],
                   marker_color=PRIMARY, name="Orders"),
            row=1, col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=inst_sum["payment_installments"], y=inst_sum["avg_value"],
                mode="lines+markers",
                line=dict(color=SECONDARY, width=3), marker=dict(size=8),
                name="Avg Value (BRL)",
            ),
            row=1, col=2,
        )
        fig2.update_layout(
            title_text="Credit Card: Installment Breakdown",
            template=TEMPLATE,
            height=380, showlegend=False,
        )
        fig2.update_xaxes(title_text="Installments", dtick=1)
        fig2.update_yaxes(title_text="Orders", row=1, col=1)
        fig2.update_yaxes(title_text="BRL",    row=1, col=2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Revenue by payment type ───────────────────────────────────────────
    pay_revenue = (
        orders_pay
        .groupby("payment_types", as_index=False)
        .agg(
            orders=("order_id",           "nunique"),
            total =("payment_total_value", "sum"),
            avg   =("payment_total_value", "mean"),
        )
        .sort_values("total", ascending=False)
    )
    fig3 = px.bar(
        pay_revenue,
        x="payment_types", y="total",
        color="avg",
        color_continuous_scale="Blues",
        text="orders",
        labels={
            "payment_types": "Payment Type",
            "total": "Total Revenue (BRL)",
            "avg":   "Avg Order Value",
            "orders": "# Orders",
        },
        title="Total Revenue & Avg Order Value by Payment Type",
        template=TEMPLATE,
    )
    fig3.update_traces(texttemplate="%{text:,} orders", textposition="outside")
    fig3.update_layout(height=340)
    st.plotly_chart(fig3, use_container_width=True)

    cc_share = (
        pay_expanded[pay_expanded["payment_type"] == "credit_card"]["count"].values[0]
        / pay_expanded["count"].sum() * 100
    )
    v1  = cc_df[cc_df["payment_installments"] == 1]["payment_total_value"].mean()
    v12 = cc_df[cc_df["payment_installments"] == 12]["payment_total_value"].mean()
    st.info(
        f"**Credit card** dominates at **{cc_share:.1f}%** of transactions.  "
        f"12-installment buyers spend **{v12/v1:.1f}×** more than single-payment buyers "
        f"(BRL {v12:.0f} vs BRL {v1:.0f})."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RFM SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "RFM Segmentation":
    st.header("RFM Customer Segmentation")
    st.markdown(
        f"_Recency · Frequency · Monetary segmentation "
        f"of {len(rfm):,} unique customers._"
    )

    seg_summary = (
        rfm.groupby("segment", as_index=False)
        .agg(
            customers   =("customer_unique_id", "count"),
            avg_recency =("recency",   "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary =("monetary",  "mean"),
            total_revenue=("monetary",  "sum"),
        )
        .sort_values("avg_monetary", ascending=False)
    )
    total_rfm_rev = seg_summary["total_revenue"].sum()
    seg_summary["rev_share_pct"] = (
        seg_summary["total_revenue"] / total_rfm_rev * 100
    ).round(1)

    # ── Segment overview ─────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            seg_summary.sort_values("customers"),
            x="customers", y="segment",
            orientation="h",
            color="avg_monetary",
            color_continuous_scale="RdYlGn",
            labels={
                "customers":    "Number of Customers",
                "segment":      "Segment",
                "avg_monetary": "Avg Spend (BRL)",
            },
            title="Customer Count per Segment",
            template=TEMPLATE,
            text="customers",
        )
        fig.update_traces(texttemplate="%{x:,}", textposition="outside")
        fig.update_layout(height=400, coloraxis_colorbar_title="Avg<br>Spend")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            seg_summary.sort_values("rev_share_pct"),
            x="rev_share_pct", y="segment",
            orientation="h",
            color="avg_recency",
            color_continuous_scale="RdYlGn_r",
            labels={
                "rev_share_pct": "Revenue Share (%)",
                "segment":       "Segment",
                "avg_recency":   "Avg Recency (days)",
            },
            title="Revenue Share & Avg Recency per Segment",
            template=TEMPLATE,
            text="rev_share_pct",
        )
        fig2.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig2.update_layout(height=400, coloraxis_colorbar_title="Avg<br>Recency")
        st.plotly_chart(fig2, use_container_width=True)

    # ── RFM scatter: recency vs monetary ─────────────────────────────────
    st.subheader("Recency vs Monetary — Segment Scatter")
    fig3 = px.scatter(
        rfm.sample(min(5000, len(rfm)), random_state=42),
        x="recency", y="monetary",
        color="segment",
        opacity=0.55,
        labels={
            "recency":  "Recency (days since last purchase)",
            "monetary": "Monetary (total spend BRL)",
            "segment":  "Segment",
        },
        title="RFM Scatter — 5,000 random customers",
        template=TEMPLATE,
        height=420,
    )
    fig3.update_traces(marker=dict(size=5))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Segment table ─────────────────────────────────────────────────────
    with st.expander("Segment Summary Table"):
        st.dataframe(
            seg_summary.rename(columns={
                "segment":       "Segment",
                "customers":     "Customers",
                "avg_recency":   "Avg Recency (days)",
                "avg_frequency": "Avg Frequency",
                "avg_monetary":  "Avg Spend (BRL)",
                "total_revenue": "Total Revenue (BRL)",
                "rev_share_pct": "Revenue Share (%)",
            }).style.format({
                "Avg Recency (days)": "{:.0f}",
                "Avg Frequency":      "{:.2f}",
                "Avg Spend (BRL)":    "{:.2f}",
                "Total Revenue (BRL)": "{:,.0f}",
                "Revenue Share (%)":  "{:.1f}",
            }),
            use_container_width=True,
        )

    champs    = seg_summary[seg_summary["segment"] == "Champions"].iloc[0]
    at_risk   = seg_summary[seg_summary["segment"] == "At Risk"].iloc[0]
    st.info(
        f"**Champions** ({champs['customers']:,} customers) avg BRL {champs['avg_monetary']:.0f} "
        f"— purchased within {champs['avg_recency']:.0f} days.  "
        f"**At Risk** ({at_risk['customers']:,} customers) avg BRL {at_risk['avg_monetary']:.0f} "
        f"— silent for {at_risk['avg_recency']:.0f} days on average."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — COHORT RETENTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Cohort Retention":
    st.header("Cohort Retention Analysis")
    st.markdown(
        "_When customers buy again after their first purchase — "
        "and how quickly retention decays._"
    )

    # cohort_retention.parquet contains eligible cohorts only (>= 50 customers
    # at acquisition), matching the notebook's conclusion heatmap.
    # cohort_sizes.parquet holds each cohort's acquisition count.
    cohort_work = cohort.copy()
    cohort_work.columns = cohort_work.columns.astype(int)  # "0" → 0

    max_period = st.slider("Max period to display", 3, min(20, cohort_work.columns.max()), 12)
    max_size   = int(cohort_sizes["cohort_size"].max())
    min_size   = st.slider("Min cohort size at acquisition (period 0)", 50, max_size, 50)

    keep = cohort_sizes.index[cohort_sizes["cohort_size"] >= min_size]
    cohort_work = cohort_work.loc[cohort_work.index.intersection(keep)]

    periods = [c for c in range(max_period + 1) if c in cohort_work.columns]
    ret_pct = (cohort_work[periods] * 100).round(2)

    # Average retention across the FULL period range (not sliced by the slider)
    # so the Month-1 / Month-6 metrics below never go out of bounds.
    avg_full = (cohort_work * 100).mean()

    # ── Plotly heatmap ────────────────────────────────────────────────────
    z_vals   = ret_pct.values.tolist()
    x_labels = [str(c) for c in ret_pct.columns.tolist()]
    y_labels = ret_pct.index.tolist()
    # NaN = period not yet observable → render as blank, not "nan%"
    text_vals = [
        ["" if pd.isna(v) else f"{v:.1f}%" for v in row]
        for row in z_vals
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            colorscale="YlOrRd_r",
            zmin=0, zmax=5,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="Retention %"),
        )
    )
    fig.update_layout(
        title="Cohort Retention Rate (%) — Monthly Cohorts",
        xaxis_title="Months since First Purchase",
        yaxis_title="Cohort (First Purchase Month)",
        template=TEMPLATE,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Retention decay line chart ────────────────────────────────────────
    st.subheader("Average Retention Curve (filtered cohorts)")
    avg_retention = avg_full.loc[:max_period].reset_index()
    avg_retention.columns = ["period", "avg_retention_pct"]
    avg_retention["period"] = avg_retention["period"].astype(int)

    fig2 = px.line(
        avg_retention,
        x="period", y="avg_retention_pct",
        markers=True,
        labels={
            "period":              "Months since First Purchase",
            "avg_retention_pct":   "Avg Retention Rate (%)",
        },
        title="Average Cohort Retention Decay",
        template=TEMPLATE,
        color_discrete_sequence=[PRIMARY],
    )
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

    m1 = avg_full.get(1, float("nan"))
    m6 = avg_full.get(6, float("nan"))
    st.info(
        f"Average **Month-1** retention: **{m1:.2f}%** — "
        f"Month-6 retention: **{m6:.2f}%**.  "
        "The near-zero retention confirms a one-time purchase model driven by "
        "organic/SEO discovery rather than brand loyalty."
    )

    with st.expander("Raw cohort retention table (%)"):
        st.dataframe(ret_pct.style.background_gradient(cmap="YlOrRd_r", vmin=0, vmax=5))
