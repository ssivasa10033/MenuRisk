"""
Tests for MenuPriceOptimizer model.

Tests model training, prediction, and optimization functionality.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.optimizer import MenuPriceOptimizer, ModelNotTrainedError
from src.data.preprocessor import InsufficientDataError


class TestMenuPriceOptimizer:
    """Test suite for MenuPriceOptimizer."""

    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = MenuPriceOptimizer()

        assert model.model is not None
        assert model.is_trained is False
        assert model.r2_score_ is None

    def test_train_with_sufficient_data(self, sample_data):
        """Test model training with sufficient data."""
        model = MenuPriceOptimizer(random_state=42)
        metrics = model.train(sample_data)

        assert model.is_trained is True
        assert metrics["r2_score"] is not None
        assert metrics["mae"] is not None
        assert metrics["rmse"] is not None
        assert metrics["cv_r2_mean"] is not None
        assert model.feature_importance_ is not None
        assert len(model.feature_importance_) > 0

    def test_train_insufficient_data(self):
        """Test training with insufficient data raises error."""
        model = MenuPriceOptimizer()
        small_data = pd.DataFrame(
            {
                "item_name": ["A", "B", "C"],
                "current_price": [10, 20, 30],
                "cogs": [5, 10, 15],
                "quantity_sold": [100, 200, 300],
            }
        )

        with pytest.raises(InsufficientDataError):
            model.train(small_data)

    def test_train_no_variance(self, sample_data):
        """Test training with no variance in target raises error."""
        model = MenuPriceOptimizer()
        no_variance_data = sample_data.copy()
        no_variance_data["cogs"] = 10
        no_variance_data["current_price"] = 20
        no_variance_data["quantity_sold"] = 100

        with pytest.raises(InsufficientDataError):
            model.train(no_variance_data)

    def test_predict_before_training(self, sample_data):
        """Test prediction before training raises error."""
        model = MenuPriceOptimizer()

        with pytest.raises(ModelNotTrainedError):
            model.predict(sample_data)

    def test_predict_after_training(self, trained_model, sample_data):
        """Test prediction after training."""
        predictions = trained_model.predict(sample_data)

        assert len(predictions) > 0
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        assert np.all(predictions > -1)

    def test_calculate_portfolio_metrics(self, sample_data):
        """Test portfolio metrics calculation."""
        model = MenuPriceOptimizer()
        metrics = model.calculate_portfolio_metrics(sample_data)

        required_keys = [
            "mean_return",
            "volatility",
            "sharpe_ratio",
            "recommendations",
            "num_items",
        ]
        for key in required_keys:
            assert key in metrics

        assert isinstance(metrics["mean_return"], float)
        assert isinstance(metrics["volatility"], float)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert len(metrics["recommendations"]) > 0

    def test_optimize_prices_before_training(self, sample_data):
        """Test price optimization before training raises error."""
        model = MenuPriceOptimizer()

        with pytest.raises(ModelNotTrainedError):
            model.optimize_prices(sample_data)

    def test_optimize_prices_after_training(self, trained_model, sample_data):
        """Test price optimization after training."""
        optimized = trained_model.optimize_prices(sample_data)

        assert "optimal_price" in optimized.columns
        assert "optimal_margin" in optimized.columns
        assert "price_change" in optimized.columns
        assert np.all(optimized["optimal_price"] >= optimized["cogs"] * 1.1)

    def test_model_reproducibility(self, sample_data):
        """Test model produces same results with same random seed."""
        model1 = MenuPriceOptimizer(random_state=42)
        model2 = MenuPriceOptimizer(random_state=42)

        metrics1 = model1.train(sample_data)
        metrics2 = model2.train(sample_data)

        assert abs(metrics1["r2_score"] - metrics2["r2_score"]) < 0.001

    def test_feature_importance_validity(self, trained_model):
        """Test feature importance values are valid."""
        importance_df = trained_model.feature_importance_

        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert np.all(importance_df["importance"] >= 0)
        assert np.all(importance_df["importance"] <= 1)
        assert abs(importance_df["importance"].sum() - 1.0) < 0.01


class TestModelAccuracy:
    """Tests for model accuracy requirements."""

    def test_model_accuracy_above_80_percent(self, high_quality_data):
        """Test model achieves >80% accuracy (R-squared)."""
        model = MenuPriceOptimizer(
            n_estimators=200, max_depth=15, min_samples_split=3, random_state=42
        )

        metrics = model.train(high_quality_data)

        assert (
            metrics["r2_score"] > 0.8
        ), f"R-squared {metrics['r2_score']:.4f} is below 0.8 threshold"

        assert (
            metrics["cv_r2_mean"] > 0.75
        ), f"CV R-squared {metrics['cv_r2_mean']:.4f} is below 0.75"

    def test_cross_validation_stability(self, high_quality_data):
        """Test cross-validation scores are stable."""
        model = MenuPriceOptimizer(n_estimators=100, random_state=42)
        metrics = model.train(high_quality_data)

        # CV std should be reasonably low
        assert (
            metrics["cv_r2_std"] < 0.1
        ), f"CV std {metrics['cv_r2_std']:.4f} indicates unstable model"
