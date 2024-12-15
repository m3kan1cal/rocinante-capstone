# Introduction

This is my Capstone project that attempts to answer the question "how can historical pharmacy claims data be used to accurately forecast a health plan group's drug spend over the next 12-24 months, accounting for utilization patterns, price changes, and member dynamics?". This is a Python data science project that uses historical pharmacy claims data to forecast drug spend. Answering this question will help health plan groups budget for future expenditures and assist in setting more accurate financial guarantees.

## Description of the dataset

The dataset in question is two years of pharmacy claims history. The dataset contains the following data:
  - Claims-level transaction details
  - Drug information (NDC, quantity, days supply, cost)
  - Temporal features (fill dates, claim processing dates)

The actual dataset is found by executing the SQL command shown below.

```sql
-- This query will give you a complete view of drug spend metrics for all health plans in 2023 and 2024,
-- with proper formatting of dates and clear identification of each plan through the concatenated ID.
WITH monthly_claims AS (
    SELECT
        customer_id || '_' || client_id || '_' || client_group_id AS health_plan_id,
        cardholder_id,
        person_code,
        date_trunc('month', to_date(date_filled::text, 'YYYYMMDD'))::date as claim_month,
        COUNT(*) as claim_count,
        SUM(total_amount_due + approved_copay) as total_drug_spend,
        SUM(total_amount_due) as plan_paid_amount,
        SUM(approved_copay) as patient_paid_amount,
        SUM(days_supply) as total_days_supply,
        SUM(metric_dec_quantity) as total_quantity,
        COUNT(DISTINCT ndc) as unique_drugs
    FROM master.rh_rx_transaction_detail
    WHERE date_submitted BETWEEN 20230101 AND 20241231
        AND transaction_status = 'P'
    GROUP BY 1, 2, 3, 4
),
-- Generate month series for the analysis period
month_series AS (
    SELECT generate_series(
        '2023-01-01'::date,
        '2024-12-31'::date,
        '1 month'::interval
    )::date as month_date
),
-- Expand eligibility to monthly records
monthly_eligibility AS (
    SELECT DISTINCT
        ec.customer_code || '_' || ec.client_code || '_' || ec.group_code AS health_plan_id,
        ec.member_id as cardholder_id,
        ec.person_code,
        ms.month_date as eligibility_month,
        -- Calculate age as of each month
        (EXTRACT(YEAR FROM age(ms.month_date, ec.birth_date::date)) * 12 +
         EXTRACT(MONTH FROM age(ms.month_date, ec.birth_date::date)))::int / 12 as age_as_of_month,
        ec.gender,
        ec.zip_code,
        ec.effective_date,
        ec.termination_date
    FROM master.elig_cache ec
    CROSS JOIN month_series ms
    WHERE ec.effective_date <= ms.month_date
        AND ec.termination_date >= ms.month_date
),
-- Combine claims and eligibility
combined_monthly_data AS (
    SELECT
        e.health_plan_id,
        e.eligibility_month as month_date,
        -- Member Demographics
        COUNT(DISTINCT e.cardholder_id || e.person_code) as total_eligible_members,
        CEILING(AVG(e.age_as_of_month)) as avg_member_age,
        COUNT(DISTINCT CASE WHEN e.age_as_of_month >= 65 THEN e.cardholder_id || e.person_code END) as medicare_aged_members,
        -- Utilization Metrics
        COUNT(DISTINCT c.cardholder_id || c.person_code) as utilizing_members,
        COALESCE(SUM(c.claim_count), 0) as total_claims,
        COALESCE(SUM(c.unique_drugs), 0) as total_unique_drugs,
        COALESCE(SUM(c.total_days_supply), 0) as total_days_supply,
        COALESCE(SUM(c.total_quantity), 0) as total_quantity,
        -- Financial Metrics
        COALESCE(SUM(c.total_drug_spend), 0) as total_drug_spend,
        COALESCE(SUM(c.plan_paid_amount), 0) as plan_paid_amount,
        COALESCE(SUM(c.patient_paid_amount), 0) as patient_paid_amount,
        -- Calculated Metrics
        CASE
            WHEN COUNT(DISTINCT e.cardholder_id || e.person_code) > 0
            THEN ROUND((COUNT(DISTINCT c.cardholder_id || c.person_code)::float /
                 COUNT(DISTINCT e.cardholder_id || e.person_code))::NUMERIC, 2)
        END as utilization_rate,
        CASE
            WHEN COUNT(DISTINCT e.cardholder_id || e.person_code) > 0
            THEN ROUND(COALESCE(SUM(c.total_drug_spend) /
                 COUNT(DISTINCT e.cardholder_id || e.person_code), 0)::NUMERIC, 2)
        END as pmpm_total_spend,
        -- Member Changes
        COUNT(DISTINCT CASE
            WHEN e.effective_date >= e.eligibility_month AND
                 e.effective_date < e.eligibility_month + interval '1 month'
            THEN e.cardholder_id || e.person_code
        END) as new_members,
        COUNT(DISTINCT CASE
            WHEN e.termination_date >= e.eligibility_month AND
                 e.termination_date < e.eligibility_month + interval '1 month'
            THEN e.cardholder_id || e.person_code
        END) as termed_members,
        -- Demographic breakdowns
        COUNT(DISTINCT CASE WHEN e.gender = 'M' THEN e.cardholder_id || e.person_code END) as male_members,
        COUNT(DISTINCT CASE WHEN e.gender = 'F' THEN e.cardholder_id || e.person_code END) as female_members,
        COUNT(DISTINCT CASE WHEN e.age_as_of_month < 18 THEN e.cardholder_id || e.person_code END) as pediatric_members,
        COUNT(DISTINCT CASE WHEN e.age_as_of_month >= 18 AND e.age_as_of_month < 65 THEN e.cardholder_id || e.person_code END) as adult_members,
        COUNT(DISTINCT e.zip_code) as unique_zip_codes
    FROM monthly_eligibility e
    LEFT JOIN monthly_claims c
        ON e.health_plan_id = c.health_plan_id
            AND e.cardholder_id = c.cardholder_id
            AND e.person_code = c.person_code
            AND e.eligibility_month = c.claim_month
    GROUP BY 1, 2
)
SELECT
    *,
    -- Calculate total PMPM across all months
    ROUND(COALESCE(AVG(pmpm_total_spend) OVER (
        PARTITION BY health_plan_id
    ), 0), 2) as total_pmpm_all_months,
    -- Add rolling metrics
    ROUND(COALESCE(AVG(total_drug_spend) OVER (
        PARTITION BY health_plan_id
        ORDER BY month_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 0), 2) as rolling_3month_spend,
    ROUND(COALESCE(AVG(pmpm_total_spend) OVER (
        PARTITION BY health_plan_id
        ORDER BY month_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 0), 2) as rolling_3month_pmpm,
    -- Year over year calculations
    COALESCE(LAG(total_drug_spend, 12) OVER (
        PARTITION BY health_plan_id
        ORDER BY month_date
    ), 0) as prior_year_spend,
    COALESCE(LAG(utilizing_members, 12) OVER (
        PARTITION BY health_plan_id
        ORDER BY month_date
    ), 0) as prior_year_utilizing_members,
    -- Calculate month-over-month changes
    COALESCE(total_drug_spend - LAG(total_drug_spend, 1) OVER (
        PARTITION BY health_plan_id
        ORDER BY month_date
    ), 0) as mom_spend_change,
    ROUND((COALESCE((total_eligible_members - LAG(total_eligible_members, 1) OVER (
        PARTITION BY health_plan_id
        ORDER BY month_date
    ))::float / NULLIF(LAG(total_eligible_members, 1) OVER (
        PARTITION BY health_plan_id
        ORDER BY month_date
    ), 0) * 100, 0))::numeric, 1) as mom_membership_change_pct
FROM combined_monthly_data
ORDER BY health_plan_id, month_date;
```

This dataset provides a strong foundation for forecasting health plan drug spend because it combines several critical elements:

Financial Metrics:
- Total drug spend
- Plan paid amounts
- Member paid amounts
- PMPM metrics
- Rolling averages and trends

Population Dynamics:
- Total eligible members vs utilizing members
- Member turnover (new/termed)
- Age and gender distributions
- Geographic distribution via zip codes
- Utilization rates

Utilization Patterns:
- Claims per member
- Unique drugs per member
- Days supply
- Quantities dispensed

Trend Indicators:
- Year-over-year comparisons
- Rolling 3-month averages
- Month-over-month changes
- Seasonality patterns

This comprehensive dataset would allow you to:
- Build baseline forecasts using historical trends
- Adjust for population changes
- Account for seasonality
- Consider demographic shifts
- Factor in utilization pattern changes

The structure of the dataset loaded into a Pandas datafram is as follows:

```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 27702 entries, 0 to 27701
Data columns (total 29 columns):
 #   Column                        Non-Null Count  Dtype
---  ------                        --------------  -----
 0   health_plan_id                27702 non-null  object
 1   month_date                    27702 non-null  object
 2   total_eligible_members        27702 non-null  int64
 3   avg_member_age                27702 non-null  int64
 4   medicare_aged_members         27702 non-null  int64
 5   utilizing_members             27702 non-null  int64
 6   total_claims                  27702 non-null  int64
 7   total_unique_drugs            27702 non-null  int64
 8   total_days_supply             27702 non-null  int64
 9   total_quantity                27702 non-null  float64
 10  total_drug_spend              27702 non-null  float64
 11  plan_paid_amount              27702 non-null  float64
 12  patient_paid_amount           27702 non-null  float64
 13  utilization_rate              27702 non-null  float64
 14  pmpm_total_spend              27702 non-null  float64
 15  new_members                   27702 non-null  int64
 16  termed_members                27702 non-null  int64
 17  male_members                  27702 non-null  int64
 18  female_members                27702 non-null  int64
 19  pediatric_members             27702 non-null  int64
 20  adult_members                 27702 non-null  int64
 21  unique_zip_codes              27702 non-null  int64
 22  total_pmpm_all_months         27702 non-null  float64
 23  rolling_3month_spend          27702 non-null  float64
 24  rolling_3month_pmpm           27702 non-null  float64
 25  prior_year_spend              27702 non-null  float64
 26  prior_year_utilizing_members  27702 non-null  int64
 27  mom_spend_change              27702 non-null  float64
 28  mom_membership_change_pct     27702 non-null  float64
dtypes: float64(12), int64(15), object(2)
```

## Description of the approach

The code implements a `DrugSpendForecast` class that provides a comprehensive framework for predicting and analyzing drug spend using either Random Forest or LightGBM models. Here's a breakdown of the key components and functions:

Core Model Functions:
- `__init__`: Initializes the forecaster with a choice between Random Forest or LightGBM models
- `_get_model`: Helper function that returns the configured model instance
- `train_model`: Handles feature selection, data splitting, scaling, model training, and stores feature importance results

Visualization Functions:
- `plot_actual_vs_predicted_low_spend`: Visualizes predictions for drug spend ≤ $500,000
- `plot_actual_vs_predicted_high_spend`: Visualizes predictions for drug spend > $500,000
- `plot_feature_importance`: Shows top 10 most important features with percentage contributions
- `plot_residuals_distribution`: Displays distribution of prediction errors using box plots
- `plot_residuals_by_actual`: Shows how prediction errors vary with actual spend
- `plot_prediction_intervals`: Creates 80% prediction intervals using gradient boosting

Analysis Functions:
- `get_detailed_metrics`: Calculates comprehensive performance metrics including:
  - Overall metrics (MAE, RMSE, R²)
  - Metrics by spend tier (low/medium/high)
  - Bias metrics (mean bias, % over-predicted)
- `compare_models`: Static method that compares Random Forest vs LightGBM performance
- `plot_model_comparison`: Creates 4 comparison visualizations:
  - Actual vs predicted scatter plots
  - Feature importance comparison
  - Residuals distribution comparison
  - Error metrics comparison

The approach focuses on:
- Robust model evaluation through multiple metrics and visualizations
- Separate handling of low vs high spend predictions
- Feature importance analysis to understand key drivers
- Model comparison to select the best performer
- Uncertainty quantification through prediction intervals
- Detailed error analysis through residual plots
- Bias detection in different spend tiers

This implementation provides a thorough framework for not just predicting drug spend, but also understanding model performance, identifying important features, and quantifying prediction uncertainty.

## Expected results

The expected results are as follows:
  - A predictive model forecasting drug spend with confidence intervals
  - Identification of key spending drivers through feature importance analysis
  - Specific forecasts breaking down:
    - Changes in utilization (quantity/days supply)
    - Impact of price changes
    - Member population dynamics (new/terminated users)
  - Performance metrics (MAPE, R-squared, or something else)
  - Visualization dashboard for monitoring

The goal now is to outline and implement specific forecasting approaches that could be used with this dataset using Python and any needed libraries so I end up with a working model that can be used to forecast drug spend.
