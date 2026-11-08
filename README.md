# Customer Analytics of a Brazilian E-Commerce Marketplace: An End-to-End Data Analysis Pipeline

## Project Overview

This project presents a comprehensive, reproducible analysis of the **Olist Public Dataset**, a publicly available corpus documenting approximately two years of transaction activity (October 2016 – August 2018) on a Brazilian online marketplace. The dataset integrates nine relational tables spanning roughly 99,000 orders, 112,000 order-line items, 99,000 customers, 3,000 sellers, 32,000 products, and over one million geospatial records, and is widely used as a benchmark for marketplace analytics because it captures the full order lifecycle — from purchase through payment, fulfillment, delivery, and post-purchase review.

The analysis is structured according to the canonical data-science workflow, progressing from problem formulation through data wrangling, exploratory analysis, visualization, and prescriptive recommendation. All findings are operationalized in an interactive **Streamlit dashboard**, ensuring that the analytical conclusions remain inspectable and reproducible rather than trapped in static output.

## Analytical Framework

### 1. Business Question Formulation

Consistent with best practice in applied analytics, the study is driven by five SMART-formulated business questions rather than by undirected data exploration:

1. The evolution of monthly revenue and order volume, and the presence of seasonality;
2. The relationship between category-level revenue generation and customer satisfaction;
3. The variation of delivery performance across Brazilian states and its effect on review scores;
4. The geographic concentration of customers and revenue, and the identification of under-served states;
5. The distribution of payment behaviours and the relationship between installment financing and purchase value.

Each question is explicitly bounded by the observation window and defined in terms of measurable quantities (currency values, rates, score differentials), ensuring that every subsequent analytical step is traceable to a well-posed problem.

### 2. Data Wrangling

The raw tables are consolidated into a single analytical base (`master_df`, 112,650 × 49) through a documented sequence of operations:

- **Entity resolution at the correct grain.** Because reviews, payments, and geolocation records are not unique per order, the pipeline aggregates each to the appropriate level — mean review score per `order_id`, summed payment value with installment maxima per `order_id`, and mean coordinates per zip-code prefix — before joining, thereby avoiding fan-out duplication.
- **Enrichment.** Customers and sellers are augmented with geospatial coordinates; product categories are translated to English via the translation table.
- **Quality assessment.** Data profiling identifies multiple defect classes: 261,831 duplicate geolocation rows, missing values concentrated in delivery timestamps, product attributes, and coordinates, an invalid `not_defined` payment type, and systematic multi-review/multi-payment cardinality.
- **Cleaning.** Defects are remediated with principled strategies — temporal parsing, state-level mean imputation for coordinates, category-level median imputation for product attributes, and exclusion of invalid payment records — each documented with its rationale.

### 3. Exploratory Analysis, Visualization, and Inference

Exploratory analysis proceeds through purpose-built aggregations (monthly revenue, category statistics, state-level delivery and revenue profiles, review-versus-delay distributions) and is given analytical force through paired static (seaborn/matplotlib) and interactive (plotly) visualizations, one or more per business question.

The principal empirical findings are:

- **Revenue concentration and growth.** Revenue grew from a soft launch to a ~BRL 1M monthly run-rate, with a pronounced Black Friday spike (November 2017). However, São Paulo alone accounts for 37.4% of revenue, and the top three states for 62.5%, indicating substantial regional concentration risk.
- **Satisfaction is logistics-mediated.** Late orders average a review score of 2.27 versus 4.29 for on-time orders -- a 2.02-point deficit on a five-point scale. Late-delivery rates reach 21.4% in Alagoas versus a 9.1% national average, identifying delivery reliability as the dominant lever of perceived service quality.
- **Category economics are decoupled from satisfaction.** The highest-revenue category (`health_beauty`) scores respectably (4.19), while `bed_bath_table` (BRL 1.23M revenue) underperforms at 3.92, suggesting that revenue leadership does not guarantee service quality.
- **One-time purchasing dominates.** 96.9% of customers transact exactly once; month-1 cohort retention averages ~0.45%, evidencing an acquisition-driven rather than loyalty-driven business model.
- **Installment financing expands demand.** Credit cards account for 75.3% of payments, and buyers using 12 installments spend approximately 3.3× more than single-payment buyers.

These findings are synthesized into six conclusions and a prioritized set of action items (e.g., SLA enforcement in north-eastern states, re-engagement of the ~22,000-customer "At Risk" RFM segment, loyalty-loop design, and geographic seller recruitment).

### 4. Customer Lifetime Value Modelling: RFM, Cohort, and Cluster Analysis

Beyond descriptive statistics, the project implements two complementary techniques for customer-base diagnosis:

- **RFM segmentation** scores each of 93,357 customers on recency, frequency, and monetary value, yielding actionable tiers — Champions, Loyal Customers, At Risk, and Lost — that translate directly into differentiated retention strategies.
- **Cohort retention analysis** tracks monthly acquisition cohorts over up to 21 periods, masking unobserved future periods as missing rather than zero (a methodological choice that preserves the distinction between "no repeat purchase" and "not yet observable"). Diagnostics restrict inference to cohorts with ≥50 customers at acquisition to avoid small-sample distortion.
- **Unsupervised K-means segmentation** complements the rule-based tiers with data-driven clusters over `log1p`-transformed, `StandardScaler`-standardised RFM features. Cluster count is selected via SSE elbow and silhouette evidence, profiles are interpreted on the raw scale, and cluster-vs-rule agreement is cross-tabulated as a sanity check.

## Technical Implementation

| Component | Technology |
|---|---|
| Analysis pipeline | `notebook.ipynb` -- pandas, numpy, seaborn, matplotlib, plotly |
| Reusable analytics modules | `src/cohort.py`, `src/rfm.py`, `src/cluster.py` -- cohort retention, RFM segmentation, and K-means clustering, dataset-agnostic |
| Data exports | Parquet files under `dashboard/data/` (dtype-preserving, compact) |
| Interactive dashboard | `dashboard/dashboard.py` -- Streamlit + Plotly, cached loading, seven analytical views with date/state/payment filters |
| Reproducibility | Single-command re-execution via `jupyter nbconvert --execute`; dashboard consumes only exported artifacts |

The separation between the analytical notebook (source of truth) and the dashboard (consumption layer) enforces a clean provenance chain: every figure and KPI in the application is derivable from the documented pipeline.

### Project Structure

```
data_analysis_project-dicoding/
├── notebook.ipynb                  # End-to-end analysis: questions → wrangling → EDA → viz → conclusions
├── src/
│   ├── cohort.py                   # CohortAnalyzer: reusable cohort retention module
│   ├── rfm.py                      # RFMAnalyzer: reusable RFM segmentation module
│   └── cluster.py                  # ClusterAnalyzer: reusable K-means segmentation module
├── data/
│   └── E-Commerce_Public_Dataset/  # Raw Olist CSV files (9 tables)
├── dashboard/
│   ├── dashboard.py                # Streamlit application
│   └── data/                       # Parquet exports consumed by the dashboard
│       ├── master_df.parquet, delivered_df.parquet, rfm.parquet,
│       ├── cohort_retention.parquet, cohort_sizes.parquet, ...
│       └── *.png                   # Static figures exported for reference
│                                   # (parquets are committed so the app deploys out-of-the-box)
├── requirements.txt                # Python dependencies (dashboard + core pipeline)
├── requirements-analysis.txt       # Adds scikit-learn for the K-means clustering step
└── README.md
```

### Setup & Reproduction

**1. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-analysis.txt   # full notebook (includes scikit-learn)
# or: pip install -r requirements.txt       # dashboard only
```

**2. Obtain the raw dataset**

Download the [Olist Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the nine CSV files under `data/E-Commerce_Public_Dataset/` (see `notebook.ipynb`, *Gathering Data*, for the expected filenames).

**3. Generate the dashboard data (if `dashboard/data/*.parquet` is missing)**

The Parquet exports are produced by executing the notebook, which performs the full wrangling pipeline and writes every table consumed by the dashboard:

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

This regenerates `master_df`, `delivered_df`, `monthly_revenue`, `category_stats`, `delivery_stats`, `state_revenue`, `review_delivery`, `rfm`, `cohort_retention`, `cohort_sizes`, and the PNG figures under `dashboard/data/`.

**4. Run the dashboard**

```bash
streamlit run dashboard/dashboard.py
```

The application loads the Parquet artifacts relative to `dashboard/dashboard.py` and exposes interactive filters (transaction-month range, customer-state selection, payment-type selection) that recompute the underlying aggregations before rendering each chart.

## Limitations

Several constraints qualify the inference. The observation window (~22 months) is short relative to customer lifetime cycles, so retention estimates are right-censored. Review scores are self-selected -- only a subset of orders is reviewed -- and may not represent the silent majority. Revenue is measured on a delivered-orders basis, excluding cancelled and in-flight orders from trend analysis. Finally, all findings are associational; claims such as "late delivery reduces review scores" describe correlation, not causal effect, absent quasi-experimental variation.

## Future Directions

The current pipeline is deliberately descriptive-prescriptive; several methodological extensions would deepen its analytical reach:

1. **Cluster validation and alternative methods.** The implemented K-means segmentation establishes a baseline; remaining work is *validation* rather than basic clustering: stability assessment via bootstrap resampling and internal indices beyond silhouette (Calinski–Harabasz, Davies–Bouldin), comparison against alternative algorithms (Gaussian mixtures, DBSCAN/HDBSCAN for non-convex structures), and downstream validation of segments against repeat-purchase rates and revenue contribution before operationalization.

2. **Time-series forecasting.** Monthly revenue and order volume exhibit trend, the November seasonal spike, and platform-growth dynamics suitable for decomposition (STL) followed by SARIMAX or state-space (ETS/Prophet-class) models, or gradient-boosted approaches (LightGBM with calendar covariates). Forecasts would enable inventory, logistics-capacity, and promotional-budget planning; backtesting on a held-out tail would be essential given the short series.

3. **Causal inference for delivery interventions.** The late-delivery/review-score relationship invites quasi-experimental designs -- difference-in-differences around logistics improvements, regression discontinuity at estimated-delivery thresholds, or instrumental-variable approaches exploiting exogenous distance/shipping-lane variation -- to move from association to actionable causal estimates of SLA enforcement.

4. **Churn and next-purchase prediction.** With 96.9% one-time buyers, a supervised classifier (e.g., gradient boosting on RFM, category, and payment features) could rank customers by reactivation propensity, converting the descriptive "At Risk" segment into a prioritized outreach queue. Survival analysis (Cox proportional hazards or parametric AFT models) would additionally model *when* customers are likely to return.

5. **Marketplace mechanism design.** Seller-level analysis -- fulfillment latency, review quality, and category specialization -- could support recommendation systems for buyer-seller matching, pricing experiments via uplift modelling, and fairness audits of geographic and category exposure.

6. **NLP on review text.** The dataset includes Portuguese review comments; sentiment analysis and topic modelling (e.g., BERTopic) would localize the drivers of low scores -- damage, sizing, or delays -- sharpening the category-quality recommendations.

Each extension preserves the project's core discipline: business-question-first formulation, documented data provenance, and evaluation against held-out evidence rather than in-sample fit.
