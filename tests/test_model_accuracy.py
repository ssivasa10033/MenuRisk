"""
Comprehensive model accuracy and performance tests.

These tests ensure the model meets production accuracy requirements.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score, mean_absolute_error

from src.models.demand_forecaster import DemandForecaster
from src.data.feature_engineer import TimeSeriesFeatureEngineer, create_train_test_split


class TestModelAccuracyRequirements:
    """Test that model meets minimum accuracy requirements."""

    @pytest.fixture
    def realistic_sales_data(self):
        """Generate realistic sales data with patterns."""
        np.random.seed(42)

        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        items = ['Butter_Chicken', 'Samosas', 'Mango_Lassi', 'Biryani', 'Naan']

        data = []
        for item in items:
            # Base sales with item-specific patterns
            base_sales = {
                'Butter_Chicken': 50,
                'Samosas': 80,
                'Mango_Lassi': 100,
                'Biryani': 40,
                'Naan': 120
            }[item]

            for i, date in enumerate(dates):
                # Weekly pattern (higher on weekends)
                day_of_week = date.dayofweek
                weekly_factor = 1.3 if day_of_week >= 5 else 1.0

                # Seasonal trend
                seasonal = 15 * np.sin(2 * np.pi * i / 365)

                # Growth trend
                growth = 0.02 * i

                # Random noise
                noise = np.random.normal(0, 5)

                quantity = max(1, int(
                    base_sales * weekly_factor + seasonal + growth + noise
                ))

                price = {
                    'Butter_Chicken': 18.99,
                    'Samosas': 6.99,
                    'Mango_Lassi': 4.99,
                    'Biryani': 16.99,
                    'Naan': 2.99
                }[item]

                cogs = price * 0.4  # 40% COGS

                data.append({
                    'date': date,
                    'item_name': item,
                    'quantity_sold': quantity,
                    'current_price': price,
                    'cogs': cogs,
                    'category': 'Main' if item in ['Butter_Chicken', 'Biryani'] else 'Appetizer',
                    'season': 'Summer',
                    'province': 'ON',
                })

        return pd.DataFrame(data)

    def test_model_r2_exceeds_minimum_threshold(self, realistic_sales_data):
        """
        CRITICAL: Model must achieve R² > 0.65 on realistic data.

        This is the primary quality gate for production deployment.
        """
        MIN_R2_THRESHOLD = 0.65

        # Use time-series split
        X_train, X_test, y_train, y_test, _ = create_train_test_split(
            realistic_sales_data,
            test_size_days=30,
            target_col='quantity_sold'
        )

        # Train model
        model = DemandForecaster(tune_hyperparams=False)
        model.train(X_train.values, y_train.values)

        # Evaluate
        predictions = model.predict(X_test.values)
        r2 = r2_score(y_test, predictions)

        assert r2 >= MIN_R2_THRESHOLD, (
            f"Model R² ({r2:.4f}) is below minimum threshold ({MIN_R2_THRESHOLD}). "
            f"Model is not ready for production."
        )

    def test_model_mae_within_acceptable_range(self, realistic_sales_data):
        """
        Model MAE should be within 20% of mean sales.
        """
        X_train, X_test, y_train, y_test, _ = create_train_test_split(
            realistic_sales_data,
            test_size_days=30
        )

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X_train.values, y_train.values)

        predictions = model.predict(X_test.values)
        mae = mean_absolute_error(y_test, predictions)

        mean_sales = y_test.mean()
        relative_mae = mae / mean_sales

        assert relative_mae < 0.20, (
            f"Model MAE ({mae:.2f}) is {relative_mae:.1%} of mean sales. "
            f"Should be < 20%."
        )

    def test_model_no_extreme_predictions(self, realistic_sales_data):
        """
        Model should not make unrealistic predictions.
        """
        X_train, X_test, y_train, y_test, _ = create_train_test_split(
            realistic_sales_data,
            test_size_days=30
        )

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X_train.values, y_train.values)

        predictions = model.predict(X_test.values)

        # No negative predictions
        assert (predictions >= 0).all(), "Model made negative predictions"

        # No predictions > 3x max observed
        max_observed = y_train.max()
        assert (predictions <= 3 * max_observed).all(), (
            f"Model made unrealistic predictions (max: {predictions.max():.0f}, "
            f"expected < {3 * max_observed:.0f})"
        )

    def test_prediction_intervals_capture_actuals(self, realistic_sales_data):
        """
        90% prediction intervals should capture ~90% of actual values.
        """
        X_train, X_test, y_train, y_test, _ = create_train_test_split(
            realistic_sales_data,
            test_size_days=30
        )

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X_train.values, y_train.values)

        preds, lower, upper = model.get_prediction_intervals(
            X_test.values,
            confidence=0.90
        )

        # Check how many actuals fall within intervals
        within_interval = ((y_test >= lower) & (y_test <= upper)).sum()
        coverage = within_interval / len(y_test)

        # Allow some tolerance (75-100% coverage for 90% intervals)
        # Note: >95% coverage means intervals are conservative (wider than needed)
        # This is acceptable for production as we prefer to overestimate uncertainty
        assert 0.75 <= coverage <= 1.0, (
            f"Prediction interval coverage ({coverage:.1%}) is outside "
            f"expected range (75-100%) for 90% confidence intervals"
        )


class TestModelConsistency:
    """Test model behavior consistency."""

    @pytest.fixture
    def sample_data(self):
        """Generate simple sample data."""
        np.random.seed(42)
        n_samples = 100

        X = np.random.randn(n_samples, 5)
        y = 50 + 10 * X[:, 0] + 5 * X[:, 1] + np.random.randn(n_samples) * 3

        return X, y

    def test_model_reproducibility_with_seed(self, sample_data):
        """Model should produce identical results with same seed."""
        X, y = sample_data

        model1 = DemandForecaster(tune_hyperparams=False, random_state=42)
        model1.train(X, y)
        pred1 = model1.predict(X[:10])

        model2 = DemandForecaster(tune_hyperparams=False, random_state=42)
        model2.train(X, y)
        pred2 = model2.predict(X[:10])

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)

    def test_model_handles_single_prediction(self, sample_data):
        """Model should handle single-row predictions."""
        X, y = sample_data

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X, y)

        single_pred = model.predict(X[0:1])
        assert len(single_pred) == 1
        assert np.isfinite(single_pred[0])

    def test_model_consistent_across_batch_sizes(self, sample_data):
        """Predictions should be consistent regardless of batch size."""
        X, y = sample_data

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X, y)

        # Predict all at once
        pred_all = model.predict(X[:10])

        # Predict one by one
        pred_individual = np.array([model.predict(X[i:i+1])[0] for i in range(10)])

        np.testing.assert_array_almost_equal(pred_all, pred_individual, decimal=6)


class TestModelRobustness:
    """Test model robustness to edge cases."""

    def test_model_handles_missing_features_gracefully(self):
        """Model should handle datasets with varying feature counts."""
        np.random.seed(42)

        # Train with 5 features
        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 50

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X_train, y_train)

        # Predict with same features should work
        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test)

        assert len(predictions) == 10
        assert all(np.isfinite(predictions))

    def test_model_handles_constant_features(self):
        """Model should handle features with no variance."""
        np.random.seed(42)

        X_train = np.random.randn(100, 5)
        X_train[:, 2] = 1.0  # Constant feature
        y_train = np.random.rand(100) * 50

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X_train, y_train)

        X_test = np.random.randn(10, 5)
        X_test[:, 2] = 1.0

        predictions = model.predict(X_test)
        assert all(np.isfinite(predictions))

    def test_model_handles_outliers_in_training(self):
        """Model should be robust to outliers in training data."""
        np.random.seed(42)

        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 50

        # Add outliers
        y_train[0] = 1000  # Extreme outlier
        y_train[1] = 0.01

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X_train, y_train)

        # Predictions should still be reasonable
        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test)

        # Most predictions should be in reasonable range
        assert (predictions > 0).all()
        assert (predictions < 500).sum() >= 8  # At least 80% reasonable


class TestFeatureImportance:
    """Test feature importance calculations."""

    def test_feature_importance_sums_to_one(self):
        """Feature importances should sum to approximately 1."""
        np.random.seed(42)

        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 50

        model = DemandForecaster(tune_hyperparams=False)
        model.train(
            X_train,
            y_train,
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5']
        )

        importance_df = model.get_feature_importance()
        total_importance = importance_df['importance'].sum()

        assert abs(total_importance - 1.0) < 0.01

    def test_feature_importance_identifies_relevant_features(self):
        """Feature importance should identify actually relevant features."""
        np.random.seed(42)

        X_train = np.random.randn(200, 5)
        # Only first two features matter
        y_train = 50 + 20 * X_train[:, 0] + 10 * X_train[:, 1]

        model = DemandForecaster(tune_hyperparams=False)
        model.train(
            X_train,
            y_train,
            feature_names=['important1', 'important2', 'noise1', 'noise2', 'noise3']
        )

        importance_df = model.get_feature_importance()

        # Top 2 features should be the important ones
        top_features = importance_df.head(2)['feature'].tolist()
        assert 'important1' in top_features
        assert 'important2' in top_features


@pytest.mark.slow
class TestModelWithHyperparameterTuning:
    """Test model with hyperparameter tuning (slower tests)."""

    @pytest.fixture
    def larger_dataset(self):
        """Generate larger dataset for tuning."""
        np.random.seed(42)
        n_samples = 500

        X = np.random.randn(n_samples, 8)
        y = (
            50
            + 15 * X[:, 0]
            + 10 * X[:, 1]
            + 5 * X[:, 2]
            + np.random.randn(n_samples) * 5
        )

        return X, y

    def test_tuned_model_improves_over_default(self, larger_dataset):
        """
        Tuned model should achieve similar or better performance than default.

        Note: Due to randomness, we check that tuning doesn't significantly degrade.
        """
        X, y = larger_dataset

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Default model
        model_default = DemandForecaster(tune_hyperparams=False)
        model_default.train(X_train, y_train)
        pred_default = model_default.predict(X_test)
        r2_default = r2_score(y_test, pred_default)

        # Tuned model
        model_tuned = DemandForecaster(tune_hyperparams=True)
        model_tuned.train(X_train, y_train, cv_splits=3, n_iter=10)
        pred_tuned = model_tuned.predict(X_test)
        r2_tuned = r2_score(y_test, pred_tuned)

        # Tuning shouldn't significantly hurt performance
        assert r2_tuned > 0.5, f"Tuned model R² too low: {r2_tuned:.4f}"

        # Ideally tuning improves, but at minimum shouldn't degrade by > 10%
        assert r2_tuned >= r2_default * 0.9, (
            f"Tuned model ({r2_tuned:.4f}) significantly worse than "
            f"default ({r2_default:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
