"""
Financial risk metrics with proper time-series calculations.

Implements correct Sharpe ratio and other risk metrics that properly
handle time-series returns and annualization.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Calculate risk-adjusted performance metrics for menu items.

    This class implements proper financial risk metrics:
    - Sharpe ratio with correct annualization
    - Sortino ratio for downside risk
    - Maximum drawdown
    - Value at Risk (VaR) and Conditional VaR
    """

    def __init__(self, risk_free_rate: float = 0.0225):
        """
        Initialize risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.0225 for 2.25%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns_timeseries(
        self,
        df: pd.DataFrame,
        item_col: str = 'item_name',
        date_col: str = 'date',
        revenue_col: str = 'revenue',
        cogs_col: str = 'cogs',
    ) -> pd.DataFrame:
        """
        Calculate time-series of profit margin returns for each item.

        NOTE: Sharpe ratio requires time-series returns, not cross-sectional margins!

        Args:
            df: DataFrame with sales data
            item_col: Column name for item identifier
            date_col: Column name for date
            revenue_col: Column name for revenue
            cogs_col: Column name for COGS

        Returns:
            DataFrame with columns [item_name, date, profit_margin, margin_return]
        """
        df = df.copy()

        # Ensure date is datetime
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

        # Calculate profit margin for each transaction
        # margin = (revenue - cogs) / revenue
        df['profit_margin'] = (df[revenue_col] - df[cogs_col]) / df[revenue_col]

        # Handle edge cases
        df['profit_margin'] = df['profit_margin'].replace([np.inf, -np.inf], np.nan)

        # Group by item and date, calculate daily average margin
        daily_margins = (
            df.groupby([item_col, date_col])['profit_margin']
            .mean()
            .reset_index()
        )

        # Sort by date within each item
        daily_margins = daily_margins.sort_values([item_col, date_col])

        # Calculate period-over-period changes (returns)
        daily_margins['margin_return'] = (
            daily_margins.groupby(item_col)['profit_margin']
            .pct_change()
        )

        # Handle extreme returns
        daily_margins['margin_return'] = daily_margins['margin_return'].clip(-1, 10)

        return daily_margins

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252,
        min_observations: int = 30,
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Formula: (annualized_return - risk_free_rate) / annualized_volatility

        Args:
            returns: Array of returns (decimal form, e.g., 0.05 for 5%)
            periods_per_year: 252 for daily, 12 for monthly, 52 for weekly
            min_observations: Minimum number of observations required

        Returns:
            Sharpe ratio (annualized)
        """
        # Remove NaN values
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        # Check minimum sample size
        if len(returns) < min_observations:
            warnings.warn(
                f"Insufficient data: {len(returns)} observations "
                f"(minimum {min_observations} required)"
            )
            return np.nan

        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # Handle edge cases
        if std_return == 0:
            if mean_return > self.risk_free_rate / periods_per_year:
                return np.inf
            else:
                return -np.inf

        # Annualize
        annual_return = mean_return * periods_per_year
        annual_std = std_return * np.sqrt(periods_per_year)

        # Calculate Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / annual_std

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252,
        target_return: float = 0.0,
        min_observations: int = 30,
    ) -> float:
        """
        Calculate Sortino ratio (better for asymmetric returns).

        Only penalizes downside volatility.

        Args:
            returns: Array of returns
            periods_per_year: Periods per year for annualization
            target_return: Minimum acceptable return (per period)
            min_observations: Minimum observations required

        Returns:
            Sortino ratio (annualized)
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < min_observations:
            return np.nan

        mean_return = np.mean(returns)

        # Calculate downside deviation (only returns below target)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return np.inf

        # Annualize
        annual_return = mean_return * periods_per_year
        annual_downside_std = downside_std * np.sqrt(periods_per_year)

        sortino = (annual_return - self.risk_free_rate) / annual_downside_std

        return float(sortino)

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown (peak-to-trough decline).

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown (negative value)
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return np.nan

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        return float(np.min(drawdown))

    def calculate_value_at_risk(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        min_observations: int = 30,
    ) -> float:
        """
        Calculate Value at Risk (VaR) at given confidence level.

        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            min_observations: Minimum observations required

        Returns:
            VaR (positive value represents loss)
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < min_observations:
            return np.nan

        var = -np.percentile(returns, (1 - confidence) * 100)
        return float(var)

    def calculate_conditional_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        min_observations: int = 30,
    ) -> float:
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall).

        Average of losses beyond VaR threshold.

        Args:
            returns: Array of returns
            confidence: Confidence level
            min_observations: Minimum observations required

        Returns:
            CVaR (positive value represents loss)
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < min_observations:
            return np.nan

        var = -np.percentile(returns, (1 - confidence) * 100)

        # Get returns worse than VaR
        tail_returns = returns[returns < -var]

        if len(tail_returns) == 0:
            return float(var)

        cvar = -np.mean(tail_returns)
        return float(cvar)

    def calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Information Ratio (active return / tracking error).

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Periods per year for annualization

        Returns:
            Information ratio
        """
        returns = np.array(returns)
        benchmark_returns = np.array(benchmark_returns)

        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Remove NaN
        mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
        returns = returns[mask]
        benchmark_returns = benchmark_returns[mask]

        if len(returns) < 30:
            return np.nan

        # Active returns
        active_returns = returns - benchmark_returns

        mean_active = np.mean(active_returns)
        tracking_error = np.std(active_returns, ddof=1)

        if tracking_error == 0:
            return np.inf if mean_active > 0 else -np.inf

        # Annualize
        annual_active = mean_active * periods_per_year
        annual_te = tracking_error * np.sqrt(periods_per_year)

        return float(annual_active / annual_te)

    def calculate_all_metrics(
        self,
        df: pd.DataFrame,
        item_col: str = 'item_name',
        date_col: str = 'date',
        revenue_col: str = 'revenue',
        cogs_col: str = 'cogs',
        periods_per_year: int = 252,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate all risk metrics for each item.

        Args:
            df: DataFrame with sales data
            item_col: Column for item identifier
            date_col: Column for date
            revenue_col: Column for revenue
            cogs_col: Column for COGS
            periods_per_year: Periods per year (252 for daily)

        Returns:
            Dict mapping item_name to dict of metrics
        """
        # Get time-series returns
        returns_df = self.calculate_returns_timeseries(
            df, item_col, date_col, revenue_col, cogs_col
        )

        results = {}

        for item in returns_df[item_col].unique():
            item_returns = returns_df[
                returns_df[item_col] == item
            ]['margin_return'].values

            results[item] = {
                'sharpe_ratio': self.calculate_sharpe_ratio(
                    item_returns, periods_per_year
                ),
                'sortino_ratio': self.calculate_sortino_ratio(
                    item_returns, periods_per_year
                ),
                'max_drawdown': self.calculate_max_drawdown(item_returns),
                'var_95': self.calculate_value_at_risk(item_returns, 0.95),
                'cvar_95': self.calculate_conditional_var(item_returns, 0.95),
                'mean_return': float(np.nanmean(item_returns) * periods_per_year),
                'volatility': float(
                    np.nanstd(item_returns, ddof=1) * np.sqrt(periods_per_year)
                ),
                'num_observations': len(item_returns[~np.isnan(item_returns)]),
            }

        return results

    def get_recommendations(
        self,
        metrics: Dict[str, Dict[str, float]],
        keep_threshold: float = 1.5,
        monitor_threshold: float = 0.8,
    ) -> Dict[str, str]:
        """
        Generate recommendations based on risk metrics.

        Args:
            metrics: Dict from calculate_all_metrics()
            keep_threshold: Sharpe ratio threshold for KEEP
            monitor_threshold: Sharpe ratio threshold for MONITOR

        Returns:
            Dict mapping item_name to recommendation (KEEP/MONITOR/REMOVE)
        """
        recommendations = {}

        for item, item_metrics in metrics.items():
            sharpe = item_metrics.get('sharpe_ratio', np.nan)

            if np.isnan(sharpe):
                recommendations[item] = 'INSUFFICIENT_DATA'
            elif sharpe >= keep_threshold:
                recommendations[item] = 'KEEP'
            elif sharpe >= monitor_threshold:
                recommendations[item] = 'MONITOR'
            else:
                recommendations[item] = 'REMOVE'

        return recommendations


class PortfolioMetrics:
    """
    Calculate portfolio-level metrics across all menu items.
    """

    def __init__(self, risk_metrics: Optional[RiskMetrics] = None):
        """
        Initialize portfolio metrics calculator.

        Args:
            risk_metrics: RiskMetrics instance (creates default if None)
        """
        self.risk_metrics = risk_metrics or RiskMetrics()

    def calculate_portfolio_sharpe(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        revenue_col: str = 'revenue',
        cogs_col: str = 'cogs',
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate portfolio-level Sharpe ratio.

        Aggregates returns across all items by date.

        Args:
            df: DataFrame with sales data
            date_col: Date column
            revenue_col: Revenue column
            cogs_col: COGS column
            periods_per_year: Periods per year

        Returns:
            Portfolio Sharpe ratio
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Aggregate by date
        daily_totals = df.groupby(date_col).agg({
            revenue_col: 'sum',
            cogs_col: 'sum',
        }).reset_index()

        # Calculate daily portfolio margin
        daily_totals['portfolio_margin'] = (
            (daily_totals[revenue_col] - daily_totals[cogs_col]) /
            daily_totals[revenue_col]
        )

        # Calculate daily returns
        daily_totals = daily_totals.sort_values(date_col)
        daily_totals['portfolio_return'] = (
            daily_totals['portfolio_margin'].pct_change()
        )

        returns = daily_totals['portfolio_return'].dropna().values

        return self.risk_metrics.calculate_sharpe_ratio(returns, periods_per_year)

    def calculate_diversification_ratio(
        self,
        item_metrics: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Calculate diversification ratio.

        Higher values indicate better diversification.

        Args:
            item_metrics: Dict from calculate_all_metrics()

        Returns:
            Diversification ratio
        """
        volatilities = [
            m['volatility'] for m in item_metrics.values()
            if not np.isnan(m.get('volatility', np.nan))
        ]

        if len(volatilities) < 2:
            return np.nan

        # Weighted average volatility (assuming equal weights)
        avg_volatility = np.mean(volatilities)

        # Portfolio volatility would be lower if diversified
        # This is a simplified calculation
        portfolio_vol = np.sqrt(np.mean([v**2 for v in volatilities]))

        return float(avg_volatility / portfolio_vol) if portfolio_vol > 0 else np.nan

    def get_portfolio_summary(
        self,
        df: pd.DataFrame,
        item_col: str = 'item_name',
        date_col: str = 'date',
        revenue_col: str = 'revenue',
        cogs_col: str = 'cogs',
        periods_per_year: int = 252,
    ) -> Dict[str, Union[float, int, Dict]]:
        """
        Generate comprehensive portfolio summary.

        Args:
            df: DataFrame with sales data
            item_col: Item identifier column
            date_col: Date column
            revenue_col: Revenue column
            cogs_col: COGS column
            periods_per_year: Periods per year

        Returns:
            Dict with portfolio metrics and per-item breakdown
        """
        # Per-item metrics
        item_metrics = self.risk_metrics.calculate_all_metrics(
            df, item_col, date_col, revenue_col, cogs_col, periods_per_year
        )

        # Recommendations
        recommendations = self.risk_metrics.get_recommendations(item_metrics)

        # Count recommendations
        rec_counts = {
            'keep': sum(1 for r in recommendations.values() if r == 'KEEP'),
            'monitor': sum(1 for r in recommendations.values() if r == 'MONITOR'),
            'remove': sum(1 for r in recommendations.values() if r == 'REMOVE'),
            'insufficient_data': sum(1 for r in recommendations.values() if r == 'INSUFFICIENT_DATA'),
        }

        # Portfolio Sharpe
        portfolio_sharpe = self.calculate_portfolio_sharpe(
            df, date_col, revenue_col, cogs_col, periods_per_year
        )

        # Diversification
        diversification = self.calculate_diversification_ratio(item_metrics)

        return {
            'total_items': len(item_metrics),
            'portfolio_sharpe': portfolio_sharpe,
            'diversification_ratio': diversification,
            'recommendation_counts': rec_counts,
            'item_metrics': item_metrics,
            'recommendations': recommendations,
        }
