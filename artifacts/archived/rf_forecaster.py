import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class DrugSpendForecast:
    def __init__(self, df):
        self.df = df.copy()
        self.df['month_date'] = pd.to_datetime(self.df['month_date'])

    def train_model(self):
        # Use existing features that are relevant for forecasting
        feature_columns = [
            'total_eligible_members',
            'avg_member_age',
            'utilizing_members',
            'total_claims',
            'total_unique_drugs',
            'total_days_supply',
            'total_quantity',
            'utilization_rate',
            'new_members',
            'termed_members',
            'pmpm_total_spend',
            'mom_spend_change',
            'mom_membership_change_pct'
        ]

        X = self.df[feature_columns]
        y = self.df['total_drug_spend']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        self.y_pred = self.model.predict(X_test_scaled)
        self.y_test = y_test

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return self.model

    def plot_actual_vs_predicted_low_spend(self):
        """Plot actual vs predicted for drug spend ≤ $500,000"""
        plt.figure(figsize=(10, 6))

        # Filter for low spend
        threshold = 500000
        mask_low = self.y_test <= threshold

        # Create scatter plot
        plt.scatter(self.y_test[mask_low], self.y_pred[mask_low],
                    alpha=0.5, label='Predictions')

        # Add reference line
        min_val = min(self.y_test[mask_low].min(), self.y_pred[mask_low].min())
        max_val = max(self.y_test[mask_low].max(), self.y_pred[mask_low].max())
        plt.plot([min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='Perfect Prediction')

        # Format axes
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.xlabel('Actual Drug Spend')
        plt.ylabel('Predicted Drug Spend')
        plt.title('Actual vs Predicted Drug Spend (≤ $500,000)')
        plt.legend()
        plt.tight_layout()
        return plt

    def plot_actual_vs_predicted_high_spend(self):
        """Plot actual vs predicted for drug spend > $500,000"""
        plt.figure(figsize=(10, 6))

        # Filter for high spend
        threshold = 500000
        mask_high = self.y_test > threshold

        # Create scatter plot
        plt.scatter(self.y_test[mask_high], self.y_pred[mask_high],
                    alpha=0.5, label='Predictions')

        # Add reference line
        min_val = min(self.y_test[mask_high].min(), self.y_pred[mask_high].min())
        max_val = max(self.y_test[mask_high].max(), self.y_pred[mask_high].max())
        plt.plot([min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='Perfect Prediction')

        # Format axes
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.xlabel('Actual Drug Spend')
        plt.ylabel('Predicted Drug Spend')
        plt.title('Actual vs Predicted Drug Spend (> $500,000)')
        plt.legend()
        plt.tight_layout()
        return plt

    def plot_feature_importance(self):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))

        # Create color palette for the number of features we'll show and convert to list
        colors = plt.cm.Set3(np.linspace(0, 1, 10)).tolist()  # Convert to list

        # Create bar plot with different colors
        ax = sns.barplot(data=self.feature_importance.head(10),
                        x='importance',
                        y='feature',
                        hue='feature',
                        palette=colors,
                        legend=False)

        # Format x-axis to show percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Add value labels on the bars
        for i, v in enumerate(self.feature_importance.head(10)['importance']):
            ax.text(v, i, f'{v:.1%}', va='center')

        plt.title('Top 10 Feature Importance for Drug Spend Prediction')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        return plt

    def get_detailed_metrics(self):
        """Calculate detailed performance metrics"""
        # Print sample of actual vs predicted values for validation
        print("\nSample of Actual vs Predicted Values:")
        print("------------------------------------")
        sample_comparison = pd.DataFrame({
            'Sample Row # (from test set)': self.y_test.head().index,
            'Actual': self.y_test.head(),
            'Predicted': self.y_pred[:5]
        })
        print(sample_comparison.to_string(index=False, float_format=lambda x: f'${x:,.2f}'))

        # Overall metrics
        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)

        print("\nValue Ranges:")
        print(f"Actual values range: ${self.y_test.min():,.2f} to ${self.y_test.max():,.2f}")
        print(f"Predicted values range: ${self.y_pred.min():,.2f} to ${self.y_pred.max():,.2f}")

        # Metrics by spend tier
        tiers = {
            'Low (<$100k)': (0, 100000),
            'Medium ($100k-$500k)': (100000, 500000),
            'High (>$500k)': (500000, float('inf'))
        }

        tier_metrics = {}
        for tier_name, (min_val, max_val) in tiers.items():
            mask = (self.y_test >= min_val) & (self.y_test < max_val)
            if mask.any():
                tier_metrics[tier_name] = {
                    'count': mask.sum(),
                    'mae': mean_absolute_error(self.y_test[mask], self.y_pred[mask]),
                    'r2': r2_score(self.y_test[mask], self.y_pred[mask])
                }

        # Calculate bias metrics
        prediction_bias = np.mean(self.y_pred - self.y_test)
        pct_over_predicted = (self.y_pred > self.y_test).mean() * 100

        return {
            'overall_metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            },
            'tier_metrics': tier_metrics,
            'bias_metrics': {
                'mean_bias': prediction_bias,
                'pct_over_predicted': pct_over_predicted
            }
        }

    def plot_residuals_distribution(self):
        """Plot distribution of prediction residuals using a box plot with jittered outliers"""
        residuals = self.y_pred - self.y_test

        plt.figure(figsize=(10, 6))

        # Create box plot without outliers first
        box = plt.boxplot(residuals,
                        vert=False,
                        widths=0.7,
                        patch_artist=True,
                        showfliers=False)    # Hide default outliers

        # Color the box
        plt.setp(box['boxes'], facecolor='skyblue', alpha=0.6)
        plt.setp(box['medians'], color='red')
        plt.setp(box['whiskers'], color='gray')
        plt.setp(box['caps'], color='gray')

        # Calculate outlier points
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = residuals[(residuals < lower_bound) | (residuals > upper_bound)]

        # Plot outliers with vertical jitter
        if len(outliers) > 0:
            # Create random vertical positions for outliers
            y_jitter = np.random.normal(1, 0.1, size=len(outliers))
            plt.scatter(outliers, y_jitter,
                    color='purple',
                    alpha=0.5,
                    marker='o')

        # Add vertical line at zero
        plt.axvline(x=0, color='red', linestyle='--', label='Zero Residual')

        # Add key statistics
        stats_text = (f'Median: ${np.median(residuals):,.0f}\n'
                    f'Mean: ${np.mean(residuals):,.0f}\n'
                    f'Std Dev: ${np.std(residuals):,.0f}')

        plt.text(plt.xlim()[0], 1.3, stats_text, verticalalignment='center')

        plt.title('Distribution of Prediction Residuals')
        plt.xlabel('Residual Amount ($)')
        plt.ylabel('')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.legend()
        plt.tight_layout()
        return plt

    def plot_residuals_by_actual(self):
        """Plot residuals against actual values"""
        residuals = self.y_pred - self.y_test

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
        plt.title('Residuals vs Actual Drug Spend')
        plt.xlabel('Actual Drug Spend')
        plt.ylabel('Residual Amount')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.legend()
        plt.tight_layout()
        return plt

    def plot_prediction_intervals(self):
        """Plot actual vs predicted with prediction intervals"""

        # Train a gradient boosting model for prediction intervals
        quantiles = [0.1, 0.9]  # 80% prediction interval
        predictions = {}

        X = self.df[self.feature_importance['feature'].tolist()]
        X_train, X_test, y_train, y_test = train_test_split(X, self.df['total_drug_spend'],
                                                            test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for quantile in quantiles:
            gbr = GradientBoostingRegressor(loss='quantile', alpha=quantile,
                                            n_estimators=100, random_state=42)
            gbr.fit(X_train_scaled, y_train)
            predictions[quantile] = gbr.predict(X_test_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, self.y_pred, alpha=0.5, label='Predictions')
        plt.fill_between(y_test, predictions[0.1], predictions[0.9],
                        color='pink', alpha=0.5, label='80% Prediction Interval')
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', label='Perfect Prediction')

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.xlabel('Actual Drug Spend')
        plt.ylabel('Predicted Drug Spend')
        plt.title('Actual vs Predicted with 80% Prediction Interval')
        plt.legend()
        plt.tight_layout()
        return plt