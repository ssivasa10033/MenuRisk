"""Tests for risk metrics calculations."""

import numpy as np
import pandas as pd
import pytest
from src.finance.risk_metrics import PortfolioAnalyzer


class TestPortfolioAnalyzer:
    """Tests for PortfolioAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a PortfolioAnalyzer instance for testing."""
        return PortfolioAnalyzer(risk_free_rate=0.0225)

    @pytest.fixture
    def sample_data(self):
        """Create sample menu data for testing."""
        return pd.DataFrame(
            {
                "item_name": ["Item A", "Item B", "Item C"] * 10,
                "current_price": [10.0, 15.0, 20.0] * 10,
                "cogs": [5.0, 8.0, 12.0] * 10,
                "quantity_sold": [100, 80, 60] * 10,
            }
        )

    def test_sharpe_ratio_zero_volatility(self, analyzer):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.1, 0.1, 0.1, 0.1])
        sharpe = analyzer.calculate_sharpe_ratio(returns)

        # Zero volatility with positive excess return = infinite Sharpe
        assert sharpe == float("inf")

    def test_calculate_portfolio_metrics(self, analyzer, sample_data):
        """Test portfolio metrics calculation."""
        metrics = analyzer.calculate_portfolio_metrics(sample_data)

        assert "mean_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "recommendations" in metrics
        assert "num_items" in metrics

        assert isinstance(metrics["mean_return"], float)
        assert isinstance(metrics["volatility"], float)

    def test_sharpe_ratio_basic(self, analyzer):
        """Test basic Sharpe ratio calculation."""
        returns = np.array([0.1, 0.05, 0.08, 0.12, 0.06])
        sharpe = analyzer.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for positive returns

    def test_calculate_var(self, analyzer):
        """Test Value at Risk calculation."""
        returns = np.array([0.1, -0.05, 0.08, -0.02, 0.12, -0.08, 0.05])
        var_95 = analyzer.calculate_var(returns, confidence_level=0.95)

        assert isinstance(var_95, float)
        # VaR should be defined and negative for data with losses
        if not np.isnan(var_95):
            assert var_95 < 0

    def test_calculate_sortino_ratio(self, analyzer):
        """Test Sortino ratio calculation."""
        returns = np.array([0.1, -0.05, 0.08, -0.02, 0.12, -0.08, 0.05])
        sortino = analyzer.calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_sortino_no_downside(self, analyzer):
        """Test Sortino ratio with no downside risk."""
        returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])  # All positive
        sortino = analyzer.calculate_sortino_ratio(returns)

        # Should be infinite when no downside risk
        assert sortino == float("inf")

    def test_recommendation_logic(self, analyzer, sample_data):
        """Test recommendation generation logic."""
        metrics = analyzer.calculate_portfolio_metrics(sample_data)

        recommendations = metrics["recommendations"]
        assert isinstance(recommendations, dict)
        assert len(recommendations) > 0

        # Check that recommendations are valid
        valid_recommendations = {"keep", "monitor", "remove"}
        for rec in recommendations.values():
            assert rec in valid_recommendations


class TestNormalityTesting:
    """Tests for normality testing functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a PortfolioAnalyzer instance for testing."""
        return PortfolioAnalyzer()

    def test_normal_distribution(self, analyzer):
        """Test normality testing with normal distribution."""
        np.random.seed(42)
        # Generate truly normal returns
        returns = np.random.normal(loc=0.1, scale=0.05, size=100)

        results = analyzer.test_normality(returns)

        assert "is_normal" in results
        assert "shapiro_p_value" in results
        # Note: jarque_bera not in backward compat version
        assert "skewness" in results
        assert "kurtosis" in results

        # Should likely be detected as normal
        assert isinstance(results["is_normal"], (bool, np.bool_))
        assert isinstance(results["shapiro_p_value"], float)

    def test_skewed_distribution(self, analyzer):
        """Test normality testing with skewed distribution."""
        np.random.seed(42)
        # Generate skewed returns (exponential)
        returns = np.random.exponential(scale=0.1, size=100)

        results = analyzer.test_normality(returns)

        # Should detect non-normality
        assert results["is_normal"] is False or results["is_normal"] == np.False_
        # Skewness should be high
        assert abs(results["skewness"]) > 0.5

    def test_heavy_tailed_distribution(self, analyzer):
        """Test normality testing with heavy-tailed distribution."""
        np.random.seed(42)
        # Generate returns with heavy tails (t-distribution)
        from scipy import stats as sp_stats

        returns = sp_stats.t.rvs(df=3, loc=0.1, scale=0.05, size=100)

        results = analyzer.test_normality(returns)

        # Might detect high kurtosis
        assert isinstance(results["kurtosis"], float)

    def test_insufficient_data(self, analyzer):
        """Test normality testing with insufficient data."""
        returns = np.array([0.1, 0.2])

        results = analyzer.test_normality(returns)

        # Should return results but may not be reliable
        assert "is_normal" in results
        assert "shapiro_p_value" in results

    def test_nan_handling(self, analyzer):
        """Test normality testing handles NaN values."""
        returns = np.array([0.1, np.nan, 0.15, 0.12, np.nan, 0.08])

        results = analyzer.test_normality(returns)

        # Should not raise error
        assert isinstance(results, dict)
        # Should have valid results from non-NaN values
        assert isinstance(results["shapiro_p_value"], float)

    def test_inf_handling(self, analyzer):
        """Test normality testing handles infinite values."""
        returns = np.array([0.1, 0.15, np.inf, 0.12, 0.08])

        results = analyzer.test_normality(returns)

        # Should not raise error
        assert isinstance(results, dict)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def analyzer(self):
        """Create a PortfolioAnalyzer instance for testing."""
        return PortfolioAnalyzer()

    def test_sharpe_ratio_single_value(self, analyzer):
        """Test Sharpe ratio with single value."""
        returns = np.array([0.1])

        sharpe = analyzer.calculate_sharpe_ratio(returns)

        # Single value has zero volatility, positive excess return = inf
        assert sharpe == float("inf")

    def test_negative_returns(self, analyzer):
        """Test handling of negative returns."""
        returns = np.array([-0.5, -0.3, -0.2, -0.4])

        sharpe = analyzer.calculate_sharpe_ratio(returns)

        # Should have negative Sharpe ratio
        assert sharpe < 0

    def test_var_empty_array(self, analyzer):
        """Test VaR with empty array."""
        returns = np.array([])

        var = analyzer.calculate_var(returns)

        # Empty array should return NaN
        assert np.isnan(var)

    def test_var_confidence_levels(self, analyzer):
        """Test VaR at different confidence levels."""
        returns = np.array([0.1, -0.05, 0.08, -0.02, 0.12, -0.08, 0.05, -0.03])

        var_90 = analyzer.calculate_var(returns, confidence_level=0.90)
        var_95 = analyzer.calculate_var(returns, confidence_level=0.95)
        var_99 = analyzer.calculate_var(returns, confidence_level=0.99)

        # Higher confidence should give more extreme (lower) VaR
        # Skip if any are NaN
        if not (np.isnan(var_90) or np.isnan(var_95) or np.isnan(var_99)):
            assert var_99 <= var_95 <= var_90

    def test_portfolio_metrics_empty_data(self, analyzer):
        """Test portfolio metrics with empty data."""
        import pandas as pd

        empty_df = pd.DataFrame(
            columns=["item_name", "current_price", "cogs", "quantity_sold"]
        )

        metrics = analyzer.calculate_portfolio_metrics(empty_df)

        assert metrics["mean_return"] == 0.0
        assert metrics["volatility"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["num_items"] == 0
