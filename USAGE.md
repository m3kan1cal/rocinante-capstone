# Guide: Using the Drug Spend Forecasting Model for New Health Plans

## Data Requirements

### Required Input Features

1. Member Demographics

    - Total eligible members
    - Gender distribution (male/female members)
    - Age distribution (pediatric/adult/medicare)
    - Geographic distribution (zip codes)

2. Utilization Metrics

    - Number of utilizing members
    - Total claims
    - Unique drugs
    - Days supply
    - Total quantity dispensed

3. Cost Components

    - Approved ingredient costs
    - Acquisition costs
    - AWP costs
    - Submitted/approved fees
    - Submitted ingredient costs

### Data Format Requirements

- Data should be aggregated monthly
- All numeric fields should be properly formatted (no text in numeric fields)
- Missing values should be handled according to the feature engineering pipeline
- Dates should be in YYYY-MM-DD format
- Monetary values should be in dollars (not cents)

## Prediction Process

**Step 1:** Data Preparation

```python
# Load and prepare new health plan data
new_plan_data = pd.read_csv('new_health_plan_data.csv')

# Convert date columns
new_plan_data['month_date'] = pd.to_datetime(new_plan_data['month_date'])

# Initialize the forecaster with the trained model
forecaster = DrugSpendForecast(model_type='linear')  # or 'random_forest' based on needs

# Apply the same feature engineering
processed_data = forecaster._engineer_features(new_plan_data)
```

**Step 2:** Making Predictions

```python
# Generate predictions
predictions = forecaster.predict(processed_data)

# Get prediction intervals if needed
prediction_intervals = forecaster.get_prediction_intervals(processed_data)
```

**Step 3:** Output Processing

```python
# Combine predictions with original data
results = pd.DataFrame({
    'month_date': processed_data['month_date'],
    'health_plan_id': processed_data['health_plan_id'],
    'predicted_spend': predictions,
    'lower_bound': prediction_intervals[:, 0],
    'upper_bound': prediction_intervals[:, 1]
})
```

## Best Practices

### Data Quality Checks

1. Pre-prediction Validation

    - Verify all required features are present
    - Check for appropriate data types
    - Validate value ranges
    - Identify any anomalies

2. Post-prediction Validation

    - Compare predictions to historical averages
    - Check for outliers in predictions
    - Verify prediction intervals are reasonable

### Business Rules

1. Timing

    - Use at least 6 months of historical data
    - Predict no more than 24 months into the future
    - Update predictions monthly with new data

2. Thresholds

    - Flag predictions that deviate >20% from historical trends
    - Set minimum/maximum reasonable values
    - Implement automatic review for outlier predictions

### Model Maintenance

1. Regular Updates

    - Retrain model quarterly with new data
    - Update feature engineering as needed
    - Validate performance metrics

2. Version Control

    - Track model versions
    - Document all changes
    - Maintain prediction history

## Implementation Example

```python
def predict_new_health_plan(plan_data, model_type='linear'):
    """
    Predict drug spend for a new health plan.

    Args:
        plan_data (pd.DataFrame): Historical data for the new health plan
        model_type (str): Type of model to use ('linear' or 'random_forest')

    Returns:
        pd.DataFrame: Predictions with confidence intervals
    """
    # Initialize forecaster
    forecaster = DrugSpendForecast(model_type=model_type)

    # Validation checks
    required_features = [
        'total_eligible_members',
        'total_claims',
        'total_unique_drugs',
        'total_days_supply',
        'total_approved_ingredient_cost',
        'total_acquisition_cost',
        'total_awp_cost'
    ]

    missing_features = [col for col in required_features if col not in plan_data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Process data
    processed_data = forecaster._engineer_features(plan_data)

    # Generate predictions
    predictions = forecaster.predict(processed_data)
    intervals = forecaster.get_prediction_intervals(processed_data)

    # Create results dataframe
    results = pd.DataFrame({
        'month_date': processed_data['month_date'],
        'health_plan_id': processed_data['health_plan_id'],
        'predicted_spend': predictions,
        'lower_bound': intervals[:, 0],
        'upper_bound': intervals[:, 1],
        'prediction_range': intervals[:, 1] - intervals[:, 0]
    })

    # Add validation flags
    results['high_variance'] = results['prediction_range'] > results['predicted_spend'] * 0.4
    results['outlier'] = np.abs(results['predicted_spend'] - results['predicted_spend'].mean()) > 2 * results['predicted_spend'].std()

    return results
```

## Usage Considerations

### When to Use Linear Model

- New health plans with limited historical data
- Plans with stable, predictable patterns
- When interpretability is crucial
- For short-term predictions (6-12 months)

### When to Use Random Forest Model

- Established plans with rich historical data
- Plans with complex utilization patterns
- When capturing non-linear relationships is important
- For longer-term predictions (12-24 months)

### Model Limitations

1. Short-term Limitations

    - May not capture sudden market changes
    - Limited ability to predict new drug launches
    - Sensitive to rapid membership changes

2. Long-term Limitations

    - Accuracy decreases with prediction horizon
    - Cannot predict structural market changes
    - May miss long-term trend shifts

## Monitoring and Maintenance

### Performance Tracking

1. Track Key Metrics

    - Prediction accuracy (MAE, RMSE)
    - Bias metrics
    - Feature importance stability

2. Set Alert Thresholds

    - Prediction deviation limits
    - Data quality metrics
    - Model drift indicators

### Regular Updates

1. Monthly Tasks

    - Update predictions with new data
    - Check prediction accuracy
    - Document any anomalies

2. Quarterly Tasks

    - Retrain models
    - Update feature importance analysis
    - Review and adjust thresholds

3. Annual Tasks

    - Full model review
    - Feature engineering updates
    - Documentation updates
