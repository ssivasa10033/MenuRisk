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
            risk_free_rate=0.0225,
            keep_threshold=1.5,
            monitor_threshold=0.8
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

        assert 'mean_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'recommendations' in metrics
        assert 'num_items' in metrics

        assert isinstance(metrics['mean_return'], float)
        assert isinstance(metrics['volatility'], float)
        assert metrics['num_items'] == len(sample_data)

    def test_recommendations_validity(self, analyzer, sample_data):
        """Test recommendation values are valid."""
        metrics = analyzer.calculate_portfolio_metrics(sample_data)
        recommendations = metrics['recommendations']

        valid_recs = {'keep', 'monitor', 'remove'}
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

        assert sortino == float('inf')

    def test_efficient_frontier_points(self, analyzer):
        """Test efficient frontier calculation."""
        returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11, 0.09, 0.14])
        vol, ret = analyzer.get_efficient_frontier_points(returns, n_points=50)

        assert len(vol) == 50
        assert len(ret) == 50
        assert np.all(vol >= 0)

    def test_custom_thresholds(self):
        """Test custom threshold settings."""
        analyzer = PortfolioAnalyzer(
            keep_threshold=2.0,
            monitor_threshold=1.0
        )

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
