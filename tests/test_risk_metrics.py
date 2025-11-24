"""
Tests for financial risk metrics module.
"""

import numpy as np
import pytest

from src.finance.risk_metrics import PortfolioAnalyzer


class TestPortfolioAnalyzer:
    """Test suite for PortfolioAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PortfolioAnalyzer instance."""
        return PortfolioAnalyzer(
            risk_free_rate=0.0225, keep_threshold=1.5, monitor_threshold=0.8
        )

    def test_calculate_sharpe_ratio(self, analyzer):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])
        sharpe = analyzer.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive returns should give positive Sharpe

    def test_sharpe_ratio_zero_volatility(self, analyzer):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.1, 0.1, 0.1, 0.1])
        sharpe = analyzer.calculate_sharpe_ratio(returns)

        assert sharpe == 0.0  # Zero volatility returns zero

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
        assert metrics["num_items"] == len(sample_data)

    def test_recommendations_validity(self, analyzer, sample_data):
        """Test recommendation values are valid."""
        metrics = analyzer.calculate_portfolio_metrics(sample_data)
        recommendations = metrics["recommendations"]

        valid_recs = {"keep", "monitor", "remove"}
        for rec in recommendations.values():
            assert rec in valid_recs

    def test_calculate_var(self, analyzer):
        """Test Value at Risk calculation."""
        returns = np.array([0.1, -0.05, 0.08, -0.02, 0.12, -0.08, 0.05])
        var_95 = analyzer.calculate_var(returns, confidence_level=0.95)

        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR should be negative for this data

    def test_calculate_sortino_ratio(self, analyzer):
        """Test Sortino ratio calculation."""
        returns = np.array([0.1, -0.05, 0.08, -0.02, 0.12, -0.08, 0.05])
        sortino = analyzer.calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_sortino_no_downside(self, analyzer):
        """Test Sortino ratio with no downside risk."""
        returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])  # All positive
        sortino = analyzer.calculate_sortino_ratio(returns)

        assert sortino == float("inf")

    def test_efficient_frontier_points(self, analyzer):
        """Test efficient frontier calculation."""
        returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11, 0.09, 0.14])
        vol, ret = analyzer.get_efficient_frontier_points(returns, n_points=50)

        assert len(vol) == 50
        assert len(ret) == 50
        assert np.all(vol >= 0)

    def test_custom_thresholds(self):
        """Test custom threshold settings."""
        analyzer = PortfolioAnalyzer(keep_threshold=2.0, monitor_threshold=1.0)

        assert analyzer.keep_threshold == 2.0
        assert analyzer.monitor_threshold == 1.0

    def test_custom_risk_free_rate(self):
        """Test custom risk-free rate."""
        returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])

        analyzer_low = PortfolioAnalyzer(risk_free_rate=0.01)
        analyzer_high = PortfolioAnalyzer(risk_free_rate=0.05)

        sharpe_low = analyzer_low.calculate_sharpe_ratio(returns)
        sharpe_high = analyzer_high.calculate_sharpe_ratio(returns)

        # Higher risk-free rate should give lower Sharpe
        assert sharpe_low > sharpe_high


class TestNormalityTesting:
    """Test normality testing functionality."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PortfolioAnalyzer instance."""
        return PortfolioAnalyzer()

    def test_normal_distribution(self, analyzer):
        """Test normality testing with normal distribution."""
        np.random.seed(42)
        # Generate truly normal returns
        returns = np.random.normal(loc=0.1, scale=0.05, size=100)

        results = analyzer.test_normality(returns)

        assert "is_normal" in results
        assert "shapiro_p_value" in results
        assert "jarque_bera_p_value" in results
        assert "skewness" in results
        assert "kurtosis" in results
        assert "recommendation" in results

        # Should likely be detected as normal
        assert isinstance(results["is_normal"], bool)
        assert isinstance(results["shapiro_p_value"], float)

    def test_skewed_distribution(self, analyzer):
        """Test normality testing with skewed distribution."""
        np.random.seed(42)
        # Generate skewed returns (exponential)
        returns = np.random.exponential(scale=0.1, size=100)

        results = analyzer.test_normality(returns)

        # Should detect non-normality
        assert results["is_normal"] is False
        # Skewness should be high
        assert abs(results["skewness"]) > 0.5
        # Recommendation should mention skewness
        assert "skewed" in results["recommendation"].lower()

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

        # Should return invalid results
        assert results["is_normal"] is False
        assert np.isnan(results["shapiro_p_value"])
        assert "Insufficient" in results["recommendation"]

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
        """Provide a PortfolioAnalyzer instance."""
        return PortfolioAnalyzer()

    def test_sharpe_ratio_empty_array(self, analyzer):
        """Test Sharpe ratio with empty array."""
        returns = np.array([])

        # Should handle gracefully
        with pytest.raises(Exception):
            analyzer.calculate_sharpe_ratio(returns)

    def test_sharpe_ratio_single_value(self, analyzer):
        """Test Sharpe ratio with single value."""
        returns = np.array([0.1])

        sharpe = analyzer.calculate_sharpe_ratio(returns)

        # With single value, std is 0, so Sharpe should be 0
        assert sharpe == 0.0

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

        assert var == 0.0

    def test_var_confidence_levels(self, analyzer):
        """Test VaR at different confidence levels."""
        returns = np.array([0.1, -0.05, 0.08, -0.02, 0.12, -0.08, 0.05, -0.03])

        var_90 = analyzer.calculate_var(returns, confidence_level=0.90)
        var_95 = analyzer.calculate_var(returns, confidence_level=0.95)
        var_99 = analyzer.calculate_var(returns, confidence_level=0.99)

        # Higher confidence should give more extreme (lower) VaR
        assert var_99 <= var_95 <= var_90

    def test_portfolio_metrics_empty_data(self, analyzer):
        """Test portfolio metrics with empty data."""
        import pandas as pd

        empty_df = pd.DataFrame(
            columns=["item_name", "current_price", "cogs", "quantity_sold"]
        )

        metrics = analyzer.calculate_portfolio_metrics(empty_df)

        assert metrics["num_items"] == 0
        assert metrics["mean_return"] == 0.0

    def test_portfolio_metrics_invalid_cogs(self, analyzer):
        """Test portfolio metrics with zero/negative COGS."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "item_name": ["A", "B", "C"],
                "current_price": [10, 20, 30],
                "cogs": [0, -5, 10],  # Invalid COGS
                "quantity_sold": [100, 200, 300],
            }
        )

        # Should handle gracefully
        metrics = analyzer.calculate_portfolio_metrics(df)

        assert isinstance(metrics, dict)
        assert "mean_return" in metrics
