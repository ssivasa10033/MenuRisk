"""
Financial risk metrics with proper time-series calculations.

Implements correct Sharpe ratio and other risk metrics that properly
handle time-series returns and annualization.

Key Improvements:
- Proper annualization of Sharpe ratio
- Time-series returns (not static margins)
- Minimum observation requirements
- Comprehensive risk metrics suite

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_RISK_FREE_RATE = 0.0225  # Bank of Canada rate (2025)
DEFAULT_MIN_OBSERVATIONS = 30
DEFAULT_PERIODS_PER_YEAR = 252  # Daily data

# Recommendation thresholds
DEFAULT_KEEP_THRESHOLD = 1.5
DEFAULT_MONITOR_THRESHOLD = 0.8


# =============================================================================
# RISK METRICS CLASS
# =============================================================================


class RiskMetrics:
    """
    Calculate risk-adjusted performance metrics for menu items.

    This class implements proper financial risk metrics:
    - Sharpe ratio with correct annualization
    - Sortino ratio for downside risk
    - Maximum drawdown
    - Value at Risk (VaR) and Conditional VaR (CVaR)
    - Information ratio

    Note on Margin Calculation:
    --------------------------
    This class uses REVENUE-BASED margin: (revenue - cogs) / revenue
    - Range: 0% to 100% (cannot exceed 100%)
    - Example: $20 revenue, $8 COGS → (20-8)/20 = 60% margin

    Alternative (COGS-based): (price - cogs) / cogs
    - Range: 0% to infinity
    - Example: $20 price, $8 COGS → (20-8)/8 = 150% markup

    We use revenue-based for consistency with standard financial metrics.
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    ):
        """
        Initialize risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.0225 for 2.25%)
            min_observations: Minimum observations required for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.min_observations = min_observations

    def calculate_returns_timeseries(
        self,
        df: pd.DataFrame,
        item_col: str = "item_name",
        date_col: str = "date",
        price_col: str = "current_price",
        cogs_col: str = "cogs",
        quantity_col: str = "quantity_sold",
    ) -> pd.DataFrame:
        """
        Calculate time-series of profit margin returns for each item.

        IMPORTANT: Sharpe ratio requires time-series returns (changes over time),
        not cross-sectional margins (static profitability).

        Args:
            df: DataFrame with sales data
            item_col: Column name for item identifier
            date_col: Column name for date
            price_col: Column name for price
            cogs_col: Column name for COGS
            quantity_col: Column name for quantity sold

        Returns:
            DataFrame with columns [item_name, date, profit_margin, margin_return]
        """
        df = df.copy()

        # Ensure date is datetime
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        else:
            logger.warning(f"No '{date_col}' column found. Creating index-based dates.")
            df[date_col] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

        # Calculate revenue
        if "revenue" not in df.columns:
            df["revenue"] = df[price_col] * df[quantity_col]

        # Calculate total COGS
        if "total_cogs" not in df.columns:
            df["total_cogs"] = df[cogs_col] * df[quantity_col]

        # Calculate profit margin (revenue-based)
        # margin = (revenue - cogs) / revenue
        df["profit_margin"] = np.where(
            df["revenue"] > 0,
            (df["revenue"] - df["total_cogs"]) / df["revenue"],
            np.nan,
        )

        # Handle edge cases
        df["profit_margin"] = df["profit_margin"].replace([np.inf, -np.inf], np.nan)

        # Group by item and date, calculate daily average margin
        daily_margins = (
            df.groupby([item_col, date_col])["profit_margin"].mean().reset_index()
        )

        # Sort by date within each item
        daily_margins = daily_margins.sort_values([item_col, date_col])

        # Calculate period-over-period changes (returns)
        # This is the KEY: Sharpe ratio needs CHANGES, not levels
        daily_margins["margin_return"] = daily_margins.groupby(item_col)[
            "profit_margin"
        ].pct_change()

        # Handle extreme returns with logging
        extreme_mask = (daily_margins["margin_return"] < -1) | (
            daily_margins["margin_return"] > 10
        )
        extreme_count = extreme_mask.sum()
        if extreme_count > 0:
            logger.warning(
                f"Clipped {extreme_count} extreme returns (< -100% or > 1000%)"
            )
        daily_margins["margin_return"] = daily_margins["margin_return"].clip(-1, 10)

        return daily_margins

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
        min_observations: Optional[int] = None,
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Formula: (annualized_return - risk_free_rate) / annualized_volatility

        The Sharpe ratio measures risk-adjusted return. A higher Sharpe ratio
        indicates better risk-adjusted performance.

        Typical ranges:
        - < 0: Losing money
        - 0-1: Subpar
        - 1-2: Good
        - 2-3: Very good
        - > 3: Excellent (verify no errors)

        Args:
            returns: Array of periodic returns (decimal form, e.g., 0.05 for 5%)
            periods_per_year: 252 for daily, 52 for weekly, 12 for monthly
            min_observations: Minimum observations required (uses instance default if None)

        Returns:
            Annualized Sharpe ratio
        """
        min_obs = min_observations or self.min_observations

        # Convert and clean
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        # Check minimum sample size
        if len(returns) < min_obs:
            logger.warning(
                f"Insufficient data for Sharpe ratio: {len(returns)} observations "
                f"(minimum {min_obs} required)"
            )
            return np.nan

        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # Handle zero volatility edge case
        if std_return == 0:
            period_rf = self.risk_free_rate / periods_per_year
            if mean_return > period_rf:
                return np.inf  # Perfect positive risk-adjusted return
            elif mean_return < period_rf:
                return -np.inf  # Guaranteed loss vs risk-free
            else:
                return 0.0  # Exactly matches risk-free rate

        # Annualize both components
        annual_return = mean_return * periods_per_year
        annual_std = std_return * np.sqrt(periods_per_year)

        # Calculate Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / annual_std

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
        target_return: float = 0.0,
        min_observations: Optional[int] = None,
    ) -> float:
        """
        Calculate Sortino ratio (downside-risk adjusted return).

        Unlike Sharpe ratio, Sortino only penalizes downside volatility,
        making it more appropriate for asymmetric return distributions.

        Args:
            returns: Array of periodic returns
            periods_per_year: Periods per year for annualization
            target_return: Minimum acceptable return (per period)
            min_observations: Minimum observations required

        Returns:
            Annualized Sortino ratio
        """
        min_obs = min_observations or self.min_observations

        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < min_obs:
            logger.warning(f"Insufficient data for Sortino ratio: {len(returns)} obs")
            return np.nan

        mean_return = np.mean(returns)

        # Calculate downside deviation (only returns below target)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return np.inf  # No downside risk observed

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
        Calculate maximum drawdown (largest peak-to-trough decline).

        Maximum drawdown measures the worst-case scenario loss from
        a peak to a subsequent trough.

        Args:
            returns: Array of periodic returns

        Returns:
            Maximum drawdown (negative value, e.g., -0.25 for 25% drawdown)
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return np.nan

        # Calculate cumulative returns (wealth index)
        cumulative = np.cumprod(1 + returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown at each point
        drawdown = (cumulative - running_max) / running_max

        return float(np.min(drawdown))

    def calculate_value_at_risk(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        min_observations: Optional[int] = None,
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.

        VaR answers: "What is the maximum loss at X% confidence?"

        Args:
            returns: Array of periodic returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            min_observations: Minimum observations required

        Returns:
            VaR as positive value (e.g., 0.05 means 5% loss at risk)
        """
        min_obs = min_observations or self.min_observations

        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < min_obs:
            return np.nan

        # VaR is the negative of the (1-confidence) percentile
        var = -np.percentile(returns, (1 - confidence) * 100)
        return float(var)

    def calculate_conditional_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        min_observations: Optional[int] = None,
    ) -> float:
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall).

        CVaR answers: "What is the expected loss given we exceed VaR?"
        It's the average of all losses worse than VaR.

        Args:
            returns: Array of periodic returns
            confidence: Confidence level
            min_observations: Minimum observations required

        Returns:
            CVaR as positive value
        """
        min_obs = min_observations or self.min_observations

        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < min_obs:
            return np.nan

        var_threshold = np.percentile(returns, (1 - confidence) * 100)

        # Get returns worse than VaR threshold
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return self.calculate_value_at_risk(returns, confidence, min_obs)

        cvar = -np.mean(tail_returns)
        return float(cvar)

    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Useful for evaluating strategies with significant drawdown risk.

        Args:
            returns: Array of periodic returns
            periods_per_year: Periods per year for annualization

        Returns:
            Calmar ratio
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < self.min_observations:
            return np.nan

        annual_return = np.mean(returns) * periods_per_year
        max_dd = self.calculate_max_drawdown(returns)

        if max_dd == 0:
            return np.inf if annual_return > 0 else -np.inf

        # Max drawdown is negative, so we negate it
        calmar = annual_return / abs(max_dd)
        return float(calmar)

    def calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> float:
        """
        Calculate Information Ratio (active return / tracking error).

        Measures risk-adjusted outperformance vs a benchmark.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Periods per year for annualization

        Returns:
            Information ratio
        """
        returns = np.asarray(returns, dtype=np.float64)
        benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Remove NaN pairs
        mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
        returns = returns[mask]
        benchmark_returns = benchmark_returns[mask]

        if len(returns) < self.min_observations:
            return np.nan

        # Active returns (excess over benchmark)
        active_returns = returns - benchmark_returns

        mean_active = np.mean(active_returns)
        tracking_error = np.std(active_returns, ddof=1)

        if tracking_error == 0:
            return np.inf if mean_active > 0 else (-np.inf if mean_active < 0 else 0.0)

        # Annualize
        annual_active = mean_active * periods_per_year
        annual_te = tracking_error * np.sqrt(periods_per_year)

        return float(annual_active / annual_te)

    def test_normality(
        self, returns: np.ndarray, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test if returns are normally distributed.

        Modern Portfolio Theory assumes normally distributed returns.
        If this assumption is violated, consider using Sortino ratio
        or CVaR instead of Sharpe ratio.

        Args:
            returns: Array of returns to test
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results and recommendation
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]

        if len(returns) < 8:
            return {
                "is_normal": False,
                "shapiro_p_value": np.nan,
                "jarque_bera_p_value": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "recommendation": "Insufficient data for normality testing",
            }

        # Shapiro-Wilk test (best for small samples)
        shapiro_stat, shapiro_p = stats.shapiro(returns[:5000])  # Limit for performance

        # Jarque-Bera test (tests skewness and kurtosis)
        jb_stat, jb_p = stats.jarque_bera(returns)

        # Calculate moments
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))  # Excess kurtosis

        # Determine if normal
        is_normal = bool(shapiro_p > alpha and jb_p > alpha)

        # Generate recommendation
        if is_normal:
            recommendation = "Returns appear normally distributed. Sharpe ratio is appropriate."
        elif abs(skewness) > 1:
            direction = "positively" if skewness > 0 else "negatively"
            recommendation = (
                f"Returns are {direction} skewed (skew={skewness:.2f}). "
                "Consider using Sortino ratio for downside-focused analysis."
            )
        elif abs(kurtosis) > 3:
            tail_type = "heavy" if kurtosis > 0 else "light"
            recommendation = (
                f"Returns have {tail_type} tails (kurtosis={kurtosis:.2f}). "
                "Consider using CVaR instead of VaR for risk measurement."
            )
        else:
            recommendation = (
                "Returns deviate from normality. Consider robust risk measures."
            )

        return {
            "is_normal": is_normal,
            "shapiro_p_value": float(shapiro_p),
            "jarque_bera_p_value": float(jb_p),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "recommendation": recommendation,
        }

    def calculate_all_metrics(
        self,
        df: pd.DataFrame,
        item_col: str = "item_name",
        date_col: str = "date",
        price_col: str = "current_price",
        cogs_col: str = "cogs",
        quantity_col: str = "quantity_sold",
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate all risk metrics for each item.

        Args:
            df: DataFrame with sales data
            item_col: Column for item identifier
            date_col: Column for date
            price_col: Column for price
            cogs_col: Column for COGS
            quantity_col: Column for quantity
            periods_per_year: Periods per year (252 for daily)

        Returns:
            Dict mapping item_name to dict of metrics
        """
        # Get time-series returns
        returns_df = self.calculate_returns_timeseries(
            df, item_col, date_col, price_col, cogs_col, quantity_col
        )

        results = {}

        for item in returns_df[item_col].unique():
            item_data = returns_df[returns_df[item_col] == item]
            item_returns = item_data["margin_return"].values
            clean_returns = item_returns[~np.isnan(item_returns)]

            # Calculate all metrics
            sharpe = self.calculate_sharpe_ratio(item_returns, periods_per_year)
            sortino = self.calculate_sortino_ratio(item_returns, periods_per_year)
            max_dd = self.calculate_max_drawdown(item_returns)
            var_95 = self.calculate_value_at_risk(item_returns, 0.95)
            cvar_95 = self.calculate_conditional_var(item_returns, 0.95)
            calmar = self.calculate_calmar_ratio(item_returns, periods_per_year)

            # Basic statistics
            if len(clean_returns) > 0:
                mean_return = float(np.mean(clean_returns) * periods_per_year)
                volatility = float(np.std(clean_returns, ddof=1) * np.sqrt(periods_per_year))
                avg_margin = float(item_data["profit_margin"].mean())
            else:
                mean_return = np.nan
                volatility = np.nan
                avg_margin = np.nan

            results[item] = {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_dd,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "calmar_ratio": calmar,
                "annualized_return": mean_return,
                "annualized_volatility": volatility,
                "average_margin": avg_margin,
                "num_observations": len(clean_returns),
            }

        return results

    def get_recommendations(
        self,
        metrics: Dict[str, Dict[str, float]],
        keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
        monitor_threshold: float = DEFAULT_MONITOR_THRESHOLD,
    ) -> Dict[str, str]:
        """
        Generate recommendations based on risk metrics.

        Recommendation Logic:
        - KEEP: Sharpe >= 1.5 (excellent risk-adjusted return)
        - MONITOR: 0.8 <= Sharpe < 1.5 (acceptable but watch)
        - REMOVE: Sharpe < 0.8 (poor risk-adjusted return)
        - INSUFFICIENT_DATA: Not enough observations

        Args:
            metrics: Dict from calculate_all_metrics()
            keep_threshold: Sharpe ratio threshold for KEEP
            monitor_threshold: Sharpe ratio threshold for MONITOR

        Returns:
            Dict mapping item_name to recommendation
        """
        recommendations = {}

        for item, item_metrics in metrics.items():
            sharpe = item_metrics.get("sharpe_ratio", np.nan)
            num_obs = item_metrics.get("num_observations", 0)

            if num_obs < self.min_observations or np.isnan(sharpe):
                recommendations[item] = "INSUFFICIENT_DATA"
            elif np.isinf(sharpe) and sharpe > 0:
                recommendations[item] = "KEEP"  # Infinite positive Sharpe
            elif sharpe >= keep_threshold:
                recommendations[item] = "KEEP"
            elif sharpe >= monitor_threshold:
                recommendations[item] = "MONITOR"
            else:
                recommendations[item] = "REMOVE"

        return recommendations


# =============================================================================
# PORTFOLIO METRICS CLASS
# =============================================================================


class PortfolioMetrics:
    """
    Calculate portfolio-level metrics across all menu items.

    Provides aggregate analysis and diversification metrics.
    """

    def __init__(self, risk_metrics: Optional[RiskMetrics] = None):
        """
        Initialize portfolio metrics calculator.

        Args:
            risk_metrics: RiskMetrics instance (creates default if None)
        """
        self.risk_metrics = risk_metrics or RiskMetrics()

    def calculate_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        item_col: str = "item_name",
        date_col: str = "date",
        return_col: str = "margin_return",
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between item returns.

        Useful for identifying:
        - Complementary items (negative correlation)
        - Redundant items (high positive correlation)
        - Diversification opportunities

        Args:
            returns_df: DataFrame from calculate_returns_timeseries()
            item_col: Item identifier column
            date_col: Date column
            return_col: Return column

        Returns:
            Correlation matrix as DataFrame
        """
        # Pivot to get items as columns, dates as rows
        pivot = returns_df.pivot(index=date_col, columns=item_col, values=return_col)

        # Calculate correlation matrix
        return pivot.corr()

    def calculate_covariance_matrix(
        self,
        returns_df: pd.DataFrame,
        item_col: str = "item_name",
        date_col: str = "date",
        return_col: str = "margin_return",
        annualize: bool = True,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix between item returns.

        Args:
            returns_df: DataFrame from calculate_returns_timeseries()
            item_col: Item identifier column
            date_col: Date column
            return_col: Return column
            annualize: Whether to annualize the covariance
            periods_per_year: Periods per year for annualization

        Returns:
            Covariance matrix as DataFrame
        """
        pivot = returns_df.pivot(index=date_col, columns=item_col, values=return_col)
        cov = pivot.cov()

        if annualize:
            cov = cov * periods_per_year

        return cov

    def calculate_diversification_ratio(
        self,
        returns_df: pd.DataFrame,
        item_col: str = "item_name",
        date_col: str = "date",
        return_col: str = "margin_return",
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate diversification ratio.

        Diversification Ratio = Weighted Avg Volatility / Portfolio Volatility

        A ratio > 1 indicates diversification benefit.
        Higher values = better diversification.

        Args:
            returns_df: DataFrame from calculate_returns_timeseries()
            item_col: Item identifier column
            date_col: Date column
            return_col: Return column
            weights: Portfolio weights (equal weights if None)

        Returns:
            Diversification ratio
        """
        # Pivot to get items as columns
        pivot = returns_df.pivot(index=date_col, columns=item_col, values=return_col)
        pivot = pivot.dropna()

        if pivot.shape[1] < 2:
            return np.nan

        n_items = pivot.shape[1]

        # Use equal weights if not specified
        if weights is None:
            weights = np.ones(n_items) / n_items
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

        # Individual volatilities
        vols = pivot.std().values

        # Weighted average volatility
        weighted_avg_vol = np.dot(weights, vols)

        # Portfolio volatility (accounts for correlations)
        cov_matrix = pivot.cov().values
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)

        if portfolio_vol == 0:
            return np.nan

        return float(weighted_avg_vol / portfolio_vol)

    def calculate_portfolio_sharpe(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        price_col: str = "current_price",
        cogs_col: str = "cogs",
        quantity_col: str = "quantity_sold",
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> float:
        """
        Calculate portfolio-level Sharpe ratio.

        Aggregates all items by date to get portfolio returns.

        Args:
            df: DataFrame with sales data
            date_col: Date column
            price_col: Price column
            cogs_col: COGS column
            quantity_col: Quantity column
            periods_per_year: Periods per year

        Returns:
            Portfolio Sharpe ratio
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Calculate revenue and total COGS
        df["revenue"] = df[price_col] * df[quantity_col]
        df["total_cogs"] = df[cogs_col] * df[quantity_col]

        # Aggregate by date
        daily_totals = (
            df.groupby(date_col)
            .agg({"revenue": "sum", "total_cogs": "sum"})
            .reset_index()
        )

        # Calculate daily portfolio margin
        daily_totals["portfolio_margin"] = np.where(
            daily_totals["revenue"] > 0,
            (daily_totals["revenue"] - daily_totals["total_cogs"])
            / daily_totals["revenue"],
            np.nan,
        )

        # Calculate daily returns
        daily_totals = daily_totals.sort_values(date_col)
        daily_totals["portfolio_return"] = daily_totals["portfolio_margin"].pct_change()

        returns = daily_totals["portfolio_return"].dropna().values

        return self.risk_metrics.calculate_sharpe_ratio(returns, periods_per_year)

    def get_portfolio_summary(
        self,
        df: pd.DataFrame,
        item_col: str = "item_name",
        date_col: str = "date",
        price_col: str = "current_price",
        cogs_col: str = "cogs",
        quantity_col: str = "quantity_sold",
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio summary.

        Args:
            df: DataFrame with sales data
            item_col: Item identifier column
            date_col: Date column
            price_col: Price column
            cogs_col: COGS column
            quantity_col: Quantity column
            periods_per_year: Periods per year

        Returns:
            Dict with portfolio metrics and per-item breakdown
        """
        # Calculate time-series returns
        returns_df = self.risk_metrics.calculate_returns_timeseries(
            df, item_col, date_col, price_col, cogs_col, quantity_col
        )

        # Per-item metrics
        item_metrics = self.risk_metrics.calculate_all_metrics(
            df, item_col, date_col, price_col, cogs_col, quantity_col, periods_per_year
        )

        # Recommendations
        recommendations = self.risk_metrics.get_recommendations(item_metrics)

        # Count recommendations
        rec_counts = {
            "keep": sum(1 for r in recommendations.values() if r == "KEEP"),
            "monitor": sum(1 for r in recommendations.values() if r == "MONITOR"),
            "remove": sum(1 for r in recommendations.values() if r == "REMOVE"),
            "insufficient_data": sum(
                1 for r in recommendations.values() if r == "INSUFFICIENT_DATA"
            ),
        }

        # Portfolio-level metrics
        portfolio_sharpe = self.calculate_portfolio_sharpe(
            df, date_col, price_col, cogs_col, quantity_col, periods_per_year
        )

        # Diversification ratio
        diversification = self.calculate_diversification_ratio(
            returns_df, item_col, date_col, "margin_return"
        )

        # Correlation matrix
        try:
            corr_matrix = self.calculate_correlation_matrix(
                returns_df, item_col, date_col, "margin_return"
            )
            avg_correlation = corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k=1)
            ].mean()
        except Exception:
            avg_correlation = np.nan

        return {
            "total_items": len(item_metrics),
            "portfolio_sharpe_ratio": portfolio_sharpe,
            "diversification_ratio": diversification,
            "average_item_correlation": float(avg_correlation)
            if not np.isnan(avg_correlation)
            else None,
            "recommendation_counts": rec_counts,
            "item_metrics": item_metrics,
            "recommendations": recommendations,
        }


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================


class PortfolioAnalyzer:
    """
    Backward-compatible wrapper for the original PortfolioAnalyzer interface.

    This class maintains API compatibility with the original risk_metrics.py
    while using the improved calculations internally.
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
        monitor_threshold: float = DEFAULT_MONITOR_THRESHOLD,
    ):
        """Initialize with backward-compatible parameters."""
        self.risk_free_rate = risk_free_rate
        self.keep_threshold = keep_threshold
        self.monitor_threshold = monitor_threshold
        self._risk_metrics = RiskMetrics(risk_free_rate=risk_free_rate)
        self._portfolio_metrics = PortfolioMetrics(self._risk_metrics)

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate Sharpe ratio (backward compatible).

        NOTE: This method assumes returns are already in the correct frequency.
        For proper annualization, use RiskMetrics.calculate_sharpe_ratio()
        with the periods_per_year parameter.

        Args:
            returns: Array of returns
            risk_free_rate: Override default risk-free rate

        Returns:
            Sharpe ratio (assumes daily returns, annualized)
        """
        if risk_free_rate is not None:
            temp_rm = RiskMetrics(risk_free_rate=risk_free_rate)
            return temp_rm.calculate_sharpe_ratio(returns, periods_per_year=252)
        return self._risk_metrics.calculate_sharpe_ratio(returns, periods_per_year=252)

    def calculate_portfolio_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate portfolio metrics (backward compatible).

        This method maintains the original interface while using
        improved calculations internally.

        Args:
            df: DataFrame with menu item data

        Returns:
            Dictionary with portfolio metrics
        """
        df = df.copy()

        # Handle missing date column (original behavior)
        if "date" not in df.columns:
            # Generate synthetic dates for cross-sectional analysis
            df["date"] = pd.date_range(
                start="2024-01-01", periods=len(df), freq="D"
            )
            logger.warning(
                "No 'date' column found. Using synthetic dates. "
                "For proper time-series analysis, include a date column."
            )

        # Calculate revenue if not present
        if "revenue" not in df.columns:
            df["revenue"] = df["current_price"] * df["quantity_sold"]

        # Use original margin calculation for backward compatibility
        df["profit_margin"] = np.where(
            df["cogs"] > 0, (df["current_price"] - df["cogs"]) / df["cogs"], 0
        )

        # Get valid returns
        returns = df["profit_margin"].values
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        returns = returns[returns > -1]

        if len(returns) == 0:
            return self._empty_metrics()

        # Calculate basic statistics
        mean_return = float(np.mean(returns))
        volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

        # For backward compatibility, use non-annualized Sharpe
        # (original behavior for cross-sectional data)
        if volatility > 0:
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(df, volatility)

        return {
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_ratio": float(sharpe_ratio),
            "recommendations": recommendations,
            "num_items": len(df),
        }

    def _generate_recommendations(
        self, df: pd.DataFrame, portfolio_volatility: float
    ) -> Dict[str, str]:
        """Generate per-item recommendations (backward compatible)."""
        recommendations = {}

        for idx, row in df.iterrows():
            item_name = row.get("item_name", f"item_{idx}")
            item_return = row.get("profit_margin", 0)

            if np.isnan(item_return) or np.isinf(item_return) or item_return < -1:
                recommendations[item_name] = "remove"
                continue

            if portfolio_volatility > 0:
                item_sharpe = (item_return - self.risk_free_rate) / portfolio_volatility
            else:
                item_sharpe = item_return - self.risk_free_rate

            if item_sharpe >= self.keep_threshold:
                recommendations[item_name] = "keep"
            elif item_sharpe >= self.monitor_threshold:
                recommendations[item_name] = "monitor"
            else:
                recommendations[item_name] = "remove"

        return recommendations

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            "mean_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "recommendations": {},
            "num_items": 0,
        }

    def calculate_var(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk (backward compatible)."""
        return self._risk_metrics.calculate_value_at_risk(returns, confidence_level)

    def calculate_sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: Optional[float] = None
    ) -> float:
        """Calculate Sortino ratio (backward compatible)."""
        if risk_free_rate is not None:
            temp_rm = RiskMetrics(risk_free_rate=risk_free_rate)
            return temp_rm.calculate_sortino_ratio(returns)
        return self._risk_metrics.calculate_sortino_ratio(returns)

    def test_normality(
        self, returns: np.ndarray, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Test normality of returns (backward compatible)."""
        return self._risk_metrics.test_normality(returns, alpha)
