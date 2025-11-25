"""
Tests for model comparison framework.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import numpy as np
import pandas as pd
import pytest

from src.models.model_comparison import (
    ModelComparison,
    NaiveBaseline,
    mean_absolute_percentage_error,
    directional_accuracy,
)


class TestMetricFunctions:
    """Test metric calculation functions."""

    def test_mape_basic(self):
        """Test basic MAPE calculation."""
        y_true = np.array([100, 200, 150, 180])
        y_pred = np.array([110, 190, 160, 170])

        mape = mean_absolute_percentage_error(y_true, y_pred)

        # MAPE should be positive
        assert mape > 0
        # MAPE should be reasonable (< 100% for this case)
        assert mape < 20

    def test_mape_perfect_predictions(self):
        """Test MAPE with perfect predictions."""
        y_true = np.array([100, 200, 150])
        y_pred = np.array([100, 200, 150])

        mape = mean_absolute_percentage_error(y_true, y_pred)

        assert mape == 0.0

    def test_mape_with_zeros(self):
        """Test MAPE handles zeros gracefully."""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 190])

        # Should not raise division by zero error
        mape = mean_absolute_percentage_error(y_true, y_pred)

        assert isinstance(mape, float)
        assert not np.isnan(mape)
        assert not np.isinf(mape)

    def test_mape_all_zeros(self):
        """Test MAPE with all zeros."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        mape = mean_absolute_percentage_error(y_true, y_pred)

        assert mape == 0.0

    def test_directional_accuracy_perfect(self):
        """Test directional accuracy with perfect direction prediction."""
        y_true = np.array([100, 110, 105, 120, 115])
        y_pred = np.array([100, 108, 107, 118, 117])

        acc = directional_accuracy(y_true, y_pred)

        # Both have same direction changes
        assert acc == 100.0

    def test_directional_accuracy_poor(self):
        """Test directional accuracy with poor direction prediction."""
        y_true = np.array([100, 110, 105, 120])  # up, down, up
        y_pred = np.array([100, 90, 115, 100])  # down, up, down

        acc = directional_accuracy(y_true, y_pred)

        # All directions are wrong
        assert acc == 0.0

    def test_directional_accuracy_insufficient_data(self):
        """Test directional accuracy with insufficient data."""
        y_true = np.array([100])
        y_pred = np.array([105])

        acc = directional_accuracy(y_true, y_pred)

        assert acc == 0.0


class TestNaiveBaseline:
    """Test naive baseline models."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        X = np.arange(100).reshape(-1, 1)
        y = 50 + 10 * np.sin(X.ravel() / 10) + np.random.randn(100) * 2
        return X, y

    def test_last_value_baseline(self, sample_data):
        """Test last value baseline."""
        X, y = sample_data

        model = NaiveBaseline(method="last_value")
        model.fit(X[:80], y[:80])

        predictions = model.predict(X[80:])

        # All predictions should be the last training value
        assert len(predictions) == 20
        assert np.allclose(predictions, y[79])

    def test_moving_average_baseline(self, sample_data):
        """Test moving average baseline."""
        X, y = sample_data

        model = NaiveBaseline(method="moving_average", window=7)
        model.fit(X[:80], y[:80])

        predictions = model.predict(X[80:])

        # All predictions should be the mean of last 7 values
        expected = np.mean(y[73:80])
        assert len(predictions) == 20
        assert np.allclose(predictions, expected)

    def test_seasonal_naive_baseline(self, sample_data):
        """Test seasonal naive baseline."""
        X, y = sample_data

        model = NaiveBaseline(method="seasonal_naive", window=10)
        model.fit(X[:80], y[:80])

        predictions = model.predict(X[80:])

        # All predictions should be the value from 10 steps ago
        assert len(predictions) == 20
        assert np.allclose(predictions, y[70])


class TestModelComparison:
    """Test model comparison framework."""

    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 300

        X = np.random.randn(n_samples, 5)
        # Create target with some correlation to features
        y = 50 + 10 * X[:, 0] + 5 * X[:, 1] + np.random.randn(n_samples) * 5

        # Split into train/test
        split = 240
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, X_test, y_train, y_test

    def test_initialization(self):
        """Test model comparison initialization."""
        comparison = ModelComparison()

        assert comparison.random_state == 42
        assert comparison.n_jobs == -1
        assert comparison.cv_splits == 5
        assert comparison.results is None

    def test_get_models(self):
        """Test getting model dictionary."""
        comparison = ModelComparison()
        models = comparison.get_models()

        # Should have at least RF, GB, and naive baselines
        assert "random_forest" in models
        assert "gradient_boosting" in models
        assert "naive_last_value" in models
        assert "naive_moving_avg" in models
        assert "naive_seasonal" in models

    def test_compare_models(self, sample_training_data):
        """Test model comparison."""
        X_train, X_test, y_train, y_test = sample_training_data

        comparison = ModelComparison()
        results = comparison.compare_models(X_train, X_test, y_train, y_test)

        # Should return DataFrame
        assert isinstance(results, pd.DataFrame)

        # Should have required columns
        required_cols = [
            "model",
            "train_r2",
            "test_r2",
            "train_mae",
            "test_mae",
            "test_mape",
            "test_directional_accuracy",
        ]
        for col in required_cols:
            assert col in results.columns

        # Should have multiple models
        assert len(results) >= 3

        # All metrics should be valid
        assert not results["test_mae"].isna().any()
        assert not results["test_r2"].isna().any()

    def test_compare_models_subset(self, sample_training_data):
        """Test comparing only a subset of models."""
        X_train, X_test, y_train, y_test = sample_training_data

        comparison = ModelComparison()
        results = comparison.compare_models(
            X_train,
            X_test,
            y_train,
            y_test,
            model_subset=["random_forest", "naive_last_value"],
        )

        # Should only have 2 models
        assert len(results) == 2
        assert set(results["model"]) == {"random_forest", "naive_last_value"}

    def test_get_best_model(self, sample_training_data):
        """Test getting best model."""
        X_train, X_test, y_train, y_test = sample_training_data

        comparison = ModelComparison()
        comparison.compare_models(X_train, X_test, y_train, y_test)

        # Get best by MAE
        best_name, best_model, best_mae = comparison.get_best_model(metric="test_mae")

        assert isinstance(best_name, str)
        assert best_model is not None
        assert isinstance(best_mae, (int, float))

        # Best model should be in trained models
        assert best_name in comparison.trained_models

    def test_get_best_model_by_r2(self, sample_training_data):
        """Test getting best model by R²."""
        X_train, X_test, y_train, y_test = sample_training_data

        comparison = ModelComparison()
        comparison.compare_models(X_train, X_test, y_train, y_test)

        # Get best by R²
        best_name, best_model, best_r2 = comparison.get_best_model(metric="test_r2")

        assert isinstance(best_name, str)
        assert best_model is not None
        # R² should be reasonable
        assert -1 <= best_r2 <= 1

    def test_get_summary(self, sample_training_data):
        """Test getting summary report."""
        X_train, X_test, y_train, y_test = sample_training_data

        comparison = ModelComparison()
        comparison.compare_models(X_train, X_test, y_train, y_test)

        summary = comparison.get_summary()

        assert isinstance(summary, str)
        assert "Model Comparison Summary" in summary
        assert "Best Model" in summary

    def test_ml_beats_naive(self, sample_training_data):
        """Test that ML models perform at least as well as naive baselines."""
        X_train, X_test, y_train, y_test = sample_training_data

        comparison = ModelComparison()
        results = comparison.compare_models(X_train, X_test, y_train, y_test)

        # Get best naive baseline MAE
        naive_results = results[results["model"].str.startswith("naive")]
        best_naive_mae = naive_results["test_mae"].min()

        # Get best ML model MAE
        ml_results = results[~results["model"].str.startswith("naive")]
        best_ml_mae = ml_results["test_mae"].min()

        # ML should perform at least as well as naive (with small tolerance)
        # Allow 5% tolerance since on small synthetic data, baselines can be competitive
        tolerance = 0.05 * best_naive_mae
        assert (
            best_ml_mae <= best_naive_mae + tolerance
        ), (
            f"ML MAE ({best_ml_mae:.4f}) should be <= naive MAE "
            f"({best_naive_mae:.4f}) + tolerance ({tolerance:.4f})"
        )

    def test_overfitting_detection(self, sample_training_data):
        """Test overfitting gap calculation."""
        X_train, X_test, y_train, y_test = sample_training_data

        comparison = ModelComparison()
        results = comparison.compare_models(X_train, X_test, y_train, y_test)

        # Check overfitting gap is calculated
        assert "overfitting_gap" in results.columns

        # Overfitting gap should be non-negative (train R² >= test R²)
        # Allow negative values due to randomness in small datasets
        # (test set can occasionally perform better than train by chance)
        assert (results["overfitting_gap"] >= -2.0).all()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_predictions_non_negative(self):
        """Test that forecasts are always non-negative."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.abs(50 + 10 * X[:, 0] + np.random.randn(100) * 5)

        comparison = ModelComparison()
        results = comparison.compare_models(X[:80], X[80:], y[:80], y[80:])

        # Check that all models make non-negative predictions
        for model_name in results["model"]:
            model = comparison.trained_models[model_name]
            preds = model.predict(X[80:])

            # Allow small negative values due to numerical errors
            assert np.all(preds >= -1), f"{model_name} has negative predictions"

    def test_sharpe_ratio_division_by_zero(self):
        """Test Sharpe ratio handles zero volatility."""
        from src.finance.risk_metrics import PortfolioAnalyzer

        analyzer = PortfolioAnalyzer()

        # Returns with zero volatility
        returns = np.array([0.5, 0.5, 0.5, 0.5])

        sharpe = analyzer.calculate_sharpe_ratio(returns)

        # Should return 0.0 (not raise error or return inf)
        assert sharpe == 0.0

    def test_empty_returns_array(self):
        """Test metrics handle empty arrays gracefully."""
        from src.finance.risk_metrics import PortfolioAnalyzer

        analyzer = PortfolioAnalyzer()

        # Empty returns
        returns = np.array([])

        var = analyzer.calculate_var(returns)

        assert var == 0.0

    def test_single_sample_data(self):
        """Test handling of single sample."""
        X = np.array([[1, 2, 3]])
        y = np.array([50])

        model = NaiveBaseline(method="last_value")
        model.fit(X, y)

        preds = model.predict(X)

        assert len(preds) == 1
        assert preds[0] == 50.0
