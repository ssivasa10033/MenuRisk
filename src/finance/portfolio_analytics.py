"""
Portfolio Analytics for Menu Analysis using Time-Series Returns.

Correctly applies Modern Portfolio Theory using time-series of returns,
not cross-sectional comparisons.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """
    Calculates risk-adjusted returns for menu items over time.

    KEY DIFFERENCE from typical implementations:
    - Uses TIME-SERIES of returns (day-to-day profit changes)
    - NOT cross-sectional comparison of different items
    - Sharpe ratio measures consistency of profits over time

    This is the CORRECT way to apply MPT to operational data.
    """

    def __init__(self, risk_free_rate: float = 0.0225):
        """
        Initialize portfolio analyzer.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2.25% - Bank of Canada)
        """
        self.risk_free_rate = risk_free_rate
        # Convert annual rate to daily
        self.daily_rfr = (1 + risk_free_rate) ** (1/365) - 1

        logger.info(f"PortfolioAnalyzer initialized (RFR: {risk_free_rate:.4f} annual)")

    def calculate_daily_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns for each menu item.

        Returns are calculated as percentage change in daily profit.

        Args:
            df: DataFrame with columns: date, item_name, quantity_sold, current_price, cogs
                Must have multiple days of data per item

        Returns:
            DataFrame with daily_profit and daily_return columns added
        """
        logger.info("Calculating daily returns")

        df = df.copy()
        df = df.sort_values(['item_name', 'date'])

        # Calculate daily profit for each item
        df['daily_profit'] = df['quantity_sold'] * (df['current_price'] - df['cogs'])

        # Calculate return as % change in profit
        df['daily_return'] = df.groupby('item_name')['daily_profit'].pct_change()

        # Handle infinite/NaN returns
        df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], np.nan)

        # Log some stats
        valid_returns = df['daily_return'].dropna()
        if len(valid_returns) > 0:
            logger.info(
                f"Daily returns calculated: "
                f"mean={valid_returns.mean():.4f}, "
                f"std={valid_returns.std():.4f}"
            )

        return df

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        freq: str = 'daily'
    ) -> float:
        """
        Calculate Sharpe ratio from time-series of returns.

        Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns

        The ratio is annualized for comparability.

        Args:
            returns: Array of returns over time (NOT across items!)
            freq: Frequency of returns ('daily', 'weekly', 'monthly')

        Returns:
            Annualized Sharpe ratio
        """
        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            logger.warning("Insufficient returns data for Sharpe ratio")
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            logger.warning("Zero volatility - returning 0 Sharpe ratio")
            return 0.0

        # Adjust risk-free rate and annualization for frequency
        if freq == 'daily':
            rfr = self.daily_rfr
            annualization_factor = np.sqrt(252)  # 252 trading days
        elif freq == 'weekly':
            rfr = (1 + self.risk_free_rate) ** (1/52) - 1
            annualization_factor = np.sqrt(52)
        elif freq == 'monthly':
            rfr = (1 + self.risk_free_rate) ** (1/12) - 1
            annualization_factor = np.sqrt(12)
        else:
            raise ValueError(f"Unknown frequency: {freq}")

        # Calculate Sharpe ratio
        sharpe = (mean_return - rfr) / std_return * annualization_factor

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: Optional[float] = None
    ) -> float:
        """
        Calculate Sortino ratio (only penalizes downside volatility).

        Sortino Ratio = (Mean Return - Target Return) / Downside Deviation

        Args:
            returns: Time-series of returns
            target_return: Minimum acceptable return (default: risk-free rate)

        Returns:
            Annualized Sortino ratio
        """
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        target = target_return if target_return is not None else self.daily_rfr

        # Downside deviation (only returns below target)
        downside_returns = returns[returns < target]

        if len(downside_returns) == 0:
            return float('inf')  # No downside risk!

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        # Annualized Sortino ratio
        sortino = (mean_return - target) / downside_std * np.sqrt(252)

        return float(sortino)

    def calculate_value_at_risk(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) - potential loss at given confidence level.

        VaR answers: "What is the maximum loss we can expect 95% of the time?"

        Args:
            returns: Time-series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            VaR value (negative number representing potential loss)
        """
        returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            logger.warning("Insufficient data for VaR calculation")
            return 0.0

        # Historical VaR (percentile method)
        var = np.percentile(returns, (1 - confidence_level) * 100)

        return float(var)

    def calculate_conditional_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall.

        CVaR answers: "If we exceed VaR, what's the average loss?"

        Args:
            returns: Time-series of returns
            confidence_level: Confidence level

        Returns:
            CVaR value (average loss beyond VaR threshold)
        """
        returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            return 0.0

        var = self.calculate_value_at_risk(returns, confidence_level)

        # CVaR is average of all returns worse than VaR
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        cvar = np.mean(tail_returns)

        return float(cvar)

    def test_return_normality(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Test if returns satisfy MPT normality assumption.

        Modern Portfolio Theory assumes normally distributed returns.
        This tests that assumption.

        Args:
            returns: Time-series of returns

        Returns:
            Dictionary with test results and recommendations
        """
        returns = returns[~np.isnan(returns)]

        if len(returns) < 20:
            return {
                'sufficient_data': False,
                'is_normal': None,
                'recommendation': 'Need at least 20 observations for normality testing'
            }

        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = stats.shapiro(returns)

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(returns)

        # Calculate skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Determine if normal (p > 0.05 means we don't reject normality)
        is_normal = shapiro_p > 0.05 and jb_p > 0.05

        result = {
            'sufficient_data': True,
            'is_normal': is_normal,
            'shapiro_p_value': float(shapiro_p),
            'jarque_bera_p_value': float(jb_p),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'sample_size': len(returns)
        }

        # Generate recommendation
        if not is_normal:
            if abs(skewness) > 1:
                result['recommendation'] = (
                    "Returns are significantly skewed. Consider using "
                    "log-returns or Sortino ratio instead of Sharpe ratio."
                )
            elif abs(kurtosis) > 3:
                result['recommendation'] = (
                    "Returns have fat tails (high kurtosis). Consider using "
                    "VaR/CVaR for risk measurement instead of standard deviation."
                )
            else:
                result['recommendation'] = (
                    "Returns are not normally distributed. MPT assumptions "
                    "may not hold. Consider robust alternatives."
                )
        else:
            result['recommendation'] = (
                "Returns appear normally distributed. MPT analysis is appropriate."
            )

        logger.info(
            f"Normality test: Shapiro p={shapiro_p:.4f}, "
            f"is_normal={is_normal}"
        )

        return result

    def analyze_item(
        self,
        df: pd.DataFrame,
        item_name: str
    ) -> Dict[str, Any]:
        """
        Complete risk-return analysis for a single menu item.

        Args:
            df: DataFrame with time-series sales data (must include date, item_name, etc.)
            item_name: Name of item to analyze

        Returns:
            Dictionary with all risk metrics
        """
        logger.info(f"Analyzing portfolio metrics for {item_name}")

        # Filter to this item
        item_df = df[df['item_name'] == item_name].copy()

        if len(item_df) < 14:
            logger.warning(f"{item_name} has <14 days of data, metrics may be unreliable")

        # Calculate returns
        item_df = self.calculate_daily_returns(item_df)
        returns = item_df['daily_return'].dropna().values

        if len(returns) < 2:
            return {
                'item_name': item_name,
                'error': 'Insufficient return data',
                'days_of_data': len(item_df)
            }

        # Calculate all metrics
        metrics = {
            'item_name': item_name,
            'days_of_data': len(item_df),
            'mean_daily_return': float(np.mean(returns)),
            'volatility_daily': float(np.std(returns, ddof=1)),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'var_95': self.calculate_value_at_risk(returns, 0.95),
            'cvar_95': self.calculate_conditional_var(returns, 0.95),
            'max_drawdown': float(np.min(returns)),
            'best_day': float(np.max(returns)),
        }

        # Normality test
        if len(returns) >= 20:
            metrics['normality_test'] = self.test_return_normality(returns)

        # Generate recommendation based on Sharpe ratio
        sharpe = metrics['sharpe_ratio']

        if sharpe >= 1.5:
            metrics['recommendation'] = 'KEEP - Excellent risk-adjusted returns'
        elif sharpe >= 0.8:
            metrics['recommendation'] = 'MONITOR - Moderate risk-adjusted returns'
        else:
            metrics['recommendation'] = 'REMOVE - Poor risk-adjusted returns'

        logger.info(
            f"{item_name}: Sharpe={sharpe:.2f}, "
            f"Recommendation={metrics['recommendation']}"
        )

        return metrics

    def analyze_menu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze entire menu portfolio.

        Args:
            df: DataFrame with time-series sales data for all items

        Returns:
            DataFrame with risk metrics for each item, sorted by Sharpe ratio
        """
        logger.info(f"Analyzing menu portfolio")

        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column for time-series analysis")

        results = []

        for item_name in df['item_name'].unique():
            try:
                metrics = self.analyze_item(df, item_name)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Failed to analyze {item_name}: {e}")

        results_df = pd.DataFrame(results)

        # Sort by Sharpe ratio (descending)
        if 'sharpe_ratio' in results_df.columns:
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        logger.info(f"Portfolio analysis complete for {len(results_df)} items")

        return results_df

    def calculate_portfolio_return_and_risk(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate overall portfolio return and risk.

        If weights not provided, uses equal weighting.

        Args:
            df: DataFrame with time-series data for all items
            weights: Dict mapping item_name to weight (should sum to 1.0)

        Returns:
            Dictionary with portfolio_return, portfolio_volatility, portfolio_sharpe
        """
        logger.info("Calculating portfolio-level metrics")

        # Calculate returns for all items
        df_with_returns = self.calculate_daily_returns(df)

        # Pivot to get returns matrix (dates x items)
        returns_pivot = df_with_returns.pivot_table(
            index='date',
            columns='item_name',
            values='daily_return'
        )

        # Handle missing data
        returns_pivot = returns_pivot.fillna(0)

        # Set weights
        items = returns_pivot.columns.tolist()

        if weights is None:
            # Equal weighting
            weights = {item: 1.0 / len(items) for item in items}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}

        # Calculate weighted returns
        weight_array = np.array([weights.get(item, 0) for item in items])
        portfolio_returns = returns_pivot.values @ weight_array

        # Calculate metrics
        portfolio_return = float(np.mean(portfolio_returns))
        portfolio_volatility = float(np.std(portfolio_returns, ddof=1))
        portfolio_sharpe = self.calculate_sharpe_ratio(portfolio_returns)

        logger.info(
            f"Portfolio: Return={portfolio_return:.4f}, "
            f"Volatility={portfolio_volatility:.4f}, "
            f"Sharpe={portfolio_sharpe:.2f}"
        )

        return {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_sharpe': portfolio_sharpe,
            'num_items': len(items)
        }
