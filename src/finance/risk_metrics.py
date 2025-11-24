"""
Financial risk metrics and portfolio analysis.

Implements Modern Portfolio Theory concepts for menu optimization:
- Sharpe Ratio calculations
- Portfolio volatility and returns
- Risk-adjusted recommendations

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_RISK_FREE_RATE = 0.0225  # Bank of Canada rate (2025)
DEFAULT_KEEP_THRESHOLD = 1.5
DEFAULT_MONITOR_THRESHOLD = 0.8


class PortfolioAnalyzer:
    """
    Analyzes menu items using portfolio theory concepts.

    Treats menu items as financial assets with risk-return profiles,
    calculating Sharpe ratios and generating recommendations.
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
        monitor_threshold: float = DEFAULT_MONITOR_THRESHOLD,
    ):
        """
        Initialize the portfolio analyzer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
            keep_threshold: Sharpe ratio threshold for 'keep' recommendation
            monitor_threshold: Sharpe ratio threshold for 'monitor' recommendation
        """
        self.risk_free_rate = risk_free_rate
        self.keep_threshold = keep_threshold
        self.monitor_threshold = monitor_threshold

    def calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate the Sharpe ratio.

        Sharpe Ratio = (E[R] - Rf) / sigma

        Args:
            returns: Array of returns (profit margins)
            risk_free_rate: Override default risk-free rate

        Returns:
            Sharpe ratio value
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0

        if volatility == 0:
            return 0.0

        sharpe = (mean_return - rf) / volatility
        return float(sharpe)

    def calculate_portfolio_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive portfolio metrics.

        Args:
            df: DataFrame with menu item data (requires current_price, cogs, quantity_sold)

        Returns:
            Dictionary containing:
            - mean_return: Average profit margin
            - volatility: Standard deviation of margins
            - sharpe_ratio: Risk-adjusted return metric
            - recommendations: Per-item recommendations
            - num_items: Number of items analyzed
        """
        logger.debug("Calculating portfolio metrics")

        df = df.copy()

        # Calculate profit margins
        df["revenue"] = df["current_price"] * df["quantity_sold"]
        df["profit"] = df["revenue"] - (df["cogs"] * df["quantity_sold"])
        df["profit_margin"] = np.where(
            df["cogs"] > 0, (df["current_price"] - df["cogs"]) / df["cogs"], 0
        )

        # Get valid returns
        returns = df["profit_margin"].values
        returns = returns[~np.isnan(returns)]
        returns = returns[returns != np.inf]
        returns = returns[returns > -1]

        if len(returns) == 0:
            logger.warning("No valid returns found")
            return self._empty_metrics()

        # Calculate portfolio-level metrics
        mean_return = float(np.mean(returns))
        volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe_ratio = self.calculate_sharpe_ratio(returns)

        logger.info(
            f"Portfolio - Return: {mean_return:.4f}, "
            f"Volatility: {volatility:.4f}, Sharpe: {sharpe_ratio:.4f}"
        )

        # Generate per-item recommendations
        recommendations = self._generate_recommendations(df, volatility)

        return {
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "recommendations": recommendations,
            "num_items": len(df),
        }

    def _generate_recommendations(
        self, df: pd.DataFrame, portfolio_volatility: float
    ) -> Dict[str, str]:
        """
        Generate recommendations for each menu item.

        Args:
            df: DataFrame with profit_margin column
            portfolio_volatility: Overall portfolio volatility

        Returns:
            Dictionary mapping item names to recommendations
        """
        recommendations = {}

        for idx, row in df.iterrows():
            item_name = row.get("item_name", f"item_{idx}")
            item_return = row["profit_margin"]

            # Handle invalid returns
            if np.isnan(item_return) or item_return == np.inf or item_return < -1:
                recommendations[item_name] = "remove"
                continue

            # Calculate item Sharpe using portfolio volatility
            if portfolio_volatility > 0:
                item_sharpe = (item_return - self.risk_free_rate) / portfolio_volatility
            else:
                item_sharpe = item_return - self.risk_free_rate

            # Apply thresholds
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
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of returns
            confidence_level: Confidence level (default: 95%)

        Returns:
            VaR value (potential loss at confidence level)
        """
        if len(returns) == 0:
            return 0.0

        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, percentile)
        return float(var)

    def calculate_sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate the Sortino ratio (downside risk-adjusted return).

        Args:
            returns: Array of returns
            risk_free_rate: Override default risk-free rate

        Returns:
            Sortino ratio value
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        mean_return = np.mean(returns)

        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < rf]
        if len(negative_returns) == 0:
            return float("inf")  # No downside risk

        downside_std = np.std(negative_returns, ddof=1)
        if downside_std == 0:
            return float("inf")

        sortino = (mean_return - rf) / downside_std
        return float(sortino)

    def get_efficient_frontier_points(
        self, returns: np.ndarray, n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier points (simplified for single-asset analysis).

        For a proper multi-asset efficient frontier, use scipy.optimize.

        Args:
            returns: Array of returns for analysis
            n_points: Number of points to generate

        Returns:
            Tuple of (volatility array, return array)
        """
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)

        # Generate points around the current portfolio
        vol_range = np.linspace(max(0, volatility * 0.5), volatility * 1.5, n_points)

        # Simplified: assume linear risk-return relationship
        return_range = self.risk_free_rate + (
            (mean_return - self.risk_free_rate) / volatility * vol_range
            if volatility > 0
            else np.full(n_points, mean_return)
        )

        return vol_range, return_range

    def test_normality(
        self, returns: np.ndarray, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test if returns are normally distributed using multiple tests.

        Modern Portfolio Theory assumes normally distributed returns.
        If returns are not normal, consider:
        - Log-returns instead of simple returns
        - Alternative risk measures (VaR, CVaR)
        - Robust portfolio optimization methods

        Args:
            returns: Array of returns to test
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results:
            - shapiro_stat: Shapiro-Wilk test statistic
            - shapiro_p_value: Shapiro-Wilk p-value
            - is_normal: Whether returns are normal at given alpha
            - skewness: Distribution skewness
            - kurtosis: Distribution kurtosis (excess)
            - recommendation: String recommendation based on results
        """
        if len(returns) < 3:
            logger.warning("Insufficient data for normality testing (need >= 3 samples)")
            return {
                "shapiro_stat": np.nan,
                "shapiro_p_value": np.nan,
                "jarque_bera_stat": np.nan,
                "jarque_bera_p_value": np.nan,
                "is_normal": False,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "recommendation": "Insufficient data for testing",
            }

        # Clean returns (remove NaN, inf, extreme outliers)
        clean_returns = returns[~np.isnan(returns)]
        clean_returns = clean_returns[~np.isinf(clean_returns)]

        if len(clean_returns) < 3:
            logger.warning("Insufficient valid data after cleaning")
            return {
                "shapiro_stat": np.nan,
                "shapiro_p_value": np.nan,
                "jarque_bera_stat": np.nan,
                "jarque_bera_p_value": np.nan,
                "is_normal": False,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "recommendation": "Insufficient valid data",
            }

        # Shapiro-Wilk test (most powerful for small samples)
        shapiro_stat, shapiro_p = stats.shapiro(clean_returns)

        # Jarque-Bera test (tests skewness and kurtosis)
        jarque_bera_stat, jarque_bera_p = stats.jarque_bera(clean_returns)

        # Calculate skewness and kurtosis
        skewness = float(stats.skew(clean_returns))
        kurtosis = float(stats.kurtosis(clean_returns))  # Excess kurtosis

        # Determine if normal
        is_normal = shapiro_p > alpha and jarque_bera_p > alpha

        # Generate recommendation
        if is_normal:
            recommendation = (
                "Returns appear normally distributed. "
                "MPT assumptions are satisfied."
            )
        else:
            if abs(skewness) > 1:
                recommendation = (
                    f"Returns are {'positively' if skewness > 0 else 'negatively'} "
                    f"skewed (skew={skewness:.2f}). Consider using log-returns "
                    "or robust portfolio methods."
                )
            elif abs(kurtosis) > 3:
                recommendation = (
                    f"Returns have {'heavy' if kurtosis > 0 else 'light'} tails "
                    f"(kurtosis={kurtosis:.2f}). Consider using VaR/CVaR "
                    "instead of volatility for risk measurement."
                )
            else:
                recommendation = (
                    "Returns deviate from normality. MPT assumptions may not hold. "
                    "Consider alternative risk measures."
                )

        logger.info(
            f"Normality test: Shapiro p={shapiro_p:.4f}, "
            f"JB p={jarque_bera_p:.4f}, Normal={is_normal}"
        )

        return {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p_value": float(shapiro_p),
            "jarque_bera_stat": float(jarque_bera_stat),
            "jarque_bera_p_value": float(jarque_bera_p),
            "is_normal": is_normal,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "recommendation": recommendation,
        }
