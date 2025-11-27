"""
Price Optimization Engine using Demand Forecasts.

Uses ML-predicted demand curves to find profit-maximizing prices.
Accounts for price elasticity and demand response.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class PriceOptimizer:
    """
    Finds optimal prices that maximize profit given demand forecasts.

    Key features:
    - Price elasticity estimation
    - Constrained optimization (min/max prices, min margins)
    - Multi-objective optimization (profit vs. volume)
    - Competitive pricing constraints

    The optimization problem:
        maximize: profit(price) = demand(price) × (price - cogs)
        subject to: price >= cogs × (1 + min_margin)
                   price <= max_price (if specified)
    """

    def __init__(self, demand_forecaster, feature_engineer=None):
        """
        Initialize price optimizer.

        Args:
            demand_forecaster: Trained DemandForecaster instance
            feature_engineer: TimeSeriesFeatureEngineer instance (optional)
        """
        if not demand_forecaster.is_trained:
            raise ValueError("DemandForecaster must be trained before use")

        self.forecaster = demand_forecaster
        self.feature_engineer = feature_engineer
        logger.info("PriceOptimizer initialized with trained forecaster")

    def predict_demand_at_price(
        self,
        df_historical: pd.DataFrame,
        item_name: str,
        price: float,
        prediction_date: Optional[pd.Timestamp] = None,
    ) -> float:
        """
        Predict demand for an item at a specific price point.

        Args:
            df_historical: Historical data for feature engineering
            item_name: Name of the item
            price: Price point to test
            prediction_date: Date for prediction (default: last date + 1)

        Returns:
            Predicted demand (quantity)
        """
        if prediction_date is None:
            prediction_date = df_historical["date"].max() + pd.Timedelta(days=1)

        # Get item's historical data
        item_df = df_historical[df_historical["item_name"] == item_name].copy()

        if len(item_df) == 0:
            logger.warning(f"No historical data for {item_name}")
            return 0.0

        # Create prediction row
        last_row = item_df.iloc[-1].copy()
        pred_row = pd.DataFrame(
            [
                {
                    "date": prediction_date,
                    "item_name": item_name,
                    "current_price": price,
                    "cogs": last_row["cogs"],
                    "quantity_sold": 0,  # Will be predicted
                    "category": last_row.get("category", "Unknown"),
                    "season": last_row.get("season", "Summer"),
                    "province": last_row.get("province", "ON"),
                }
            ]
        )

        # Combine with historical data for feature engineering
        combined_df = pd.concat([item_df, pred_row], ignore_index=True)

        # Engineer features
        if self.feature_engineer:
            combined_df = self.feature_engineer.transform(combined_df)
        else:
            # Basic feature engineering if no feature engineer provided
            combined_df = self._basic_feature_engineering(combined_df)

        # Get the last row (prediction row) features
        pred_features = combined_df.iloc[-1:]

        # Get feature columns (exclude metadata and target)
        exclude_cols = ["date", "item_name", "quantity_sold", "profit_margin"]
        feature_cols = [col for col in pred_features.columns if col not in exclude_cols]

        # Filter to numeric columns only
        numeric_cols = [
            col
            for col in feature_cols
            if pd.api.types.is_numeric_dtype(pred_features[col])
        ]

        X_pred = pred_features[numeric_cols].fillna(0)

        # Make prediction
        try:
            demand = self.forecaster.predict(X_pred)[0]
            return max(0, demand)  # Ensure non-negative
        except Exception as e:
            logger.error(f"Prediction failed for {item_name} at ${price:.2f}: {e}")
            return 0.0

    def _basic_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic feature engineering if no feature engineer is provided.

        Args:
            df: DataFrame with basic columns

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        # Sort by date
        df = df.sort_values("date")

        # Add lag features
        for lag in [1, 3, 7]:
            df[f"lag_{lag}d"] = df.groupby("item_name")["quantity_sold"].shift(lag)

        # Add rolling averages
        for window in [7, 14]:
            df[f"rolling_avg_{window}d"] = df.groupby("item_name")[
                "quantity_sold"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Add temporal features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Price features
        df["price_to_cogs"] = df["current_price"] / df["cogs"]

        # Category encoding
        if "category" in df.columns:
            df = pd.get_dummies(df, columns=["category"], prefix="category")

        return df

    def estimate_price_elasticity(
        self,
        df_historical: pd.DataFrame,
        item_name: str,
        current_price: float,
        price_change_pct: float = 0.01,
    ) -> float:
        """
        Estimate price elasticity of demand.

        Elasticity = (% change in quantity) / (% change in price)

        Interpretation:
        - e < -1: Elastic (demand very sensitive to price)
        - -1 < e < 0: Inelastic (demand less sensitive)
        - e = -1: Unit elastic

        Args:
            df_historical: Historical sales data
            item_name: Item to analyze
            current_price: Current price point
            price_change_pct: Percentage price change to test (default: 1%)

        Returns:
            Price elasticity coefficient
        """
        # Test small price changes
        price_high = current_price * (1 + price_change_pct)
        price_low = current_price * (1 - price_change_pct)

        # Predict demand at different prices
        q_current = self.predict_demand_at_price(
            df_historical, item_name, current_price
        )
        q_high = self.predict_demand_at_price(df_historical, item_name, price_high)
        q_low = self.predict_demand_at_price(df_historical, item_name, price_low)

        if q_current == 0:
            return 0.0

        # Calculate elasticity using midpoint method
        pct_price_change = (price_high - price_low) / current_price
        pct_quantity_change = (q_high - q_low) / q_current

        if pct_price_change == 0:
            return 0.0

        elasticity = pct_quantity_change / pct_price_change

        logger.debug(
            f"{item_name} elasticity at ${current_price:.2f}: {elasticity:.2f}"
        )

        return elasticity

    def optimize_price_single_item(
        self,
        df_historical: pd.DataFrame,
        item_name: str,
        cogs: float,
        min_margin: float = 0.10,
        max_price: Optional[float] = None,
        prediction_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Find profit-maximizing price for a single item.

        Args:
            df_historical: Historical sales data
            item_name: Item to optimize
            cogs: Cost of goods sold
            min_margin: Minimum profit margin (default: 10%)
            max_price: Maximum allowed price (optional)
            prediction_date: Date for prediction

        Returns:
            Dictionary with:
            - item_name: Item name
            - optimal_price: Profit-maximizing price
            - expected_demand: Expected quantity at optimal price
            - expected_profit: Expected daily profit
            - price_elasticity: Elasticity at optimal price
            - margin: Profit margin (as ratio)
            - margin_pct: Profit margin (as percentage)
        """
        logger.info(f"Optimizing price for {item_name} (COGS: ${cogs:.2f})")

        # Define profit function (negative for minimization)
        def negative_profit(price: float) -> float:
            """Negative profit for scipy minimization."""
            if price < cogs * (1 + min_margin):
                return 1e10  # Heavy penalty for violating margin constraint

            if max_price and price > max_price:
                return 1e10  # Heavy penalty for exceeding max price

            demand = self.predict_demand_at_price(
                df_historical, item_name, price, prediction_date
            )

            profit = demand * (price - cogs)
            return -profit  # Negative for minimization

        # Set bounds
        lower_bound = cogs * (1 + min_margin)
        upper_bound = (
            max_price if max_price else cogs * 5.0
        )  # Max 5x markup if no limit

        # Optimize using bounded scalar minimization
        try:
            result = minimize_scalar(
                negative_profit,
                bounds=(lower_bound, upper_bound),
                method="bounded",
                options={"xatol": 0.01},  # Price tolerance of 1 cent
            )

            optimal_price = result.x

        except Exception as e:
            logger.error(f"Optimization failed for {item_name}: {e}")
            # Fallback: use 2x markup
            optimal_price = cogs * 2.0

        # Calculate metrics at optimal price
        optimal_demand = self.predict_demand_at_price(
            df_historical, item_name, optimal_price, prediction_date
        )
        optimal_profit = optimal_demand * (optimal_price - cogs)

        # Estimate elasticity at optimal price
        elasticity = self.estimate_price_elasticity(
            df_historical, item_name, optimal_price
        )

        margin = (optimal_price - cogs) / cogs if cogs > 0 else 0
        margin_pct = margin * 100

        logger.info(
            f"Optimal price: ${optimal_price:.2f}, "
            f"Expected demand: {optimal_demand:.1f}, "
            f"Expected profit: ${optimal_profit:.2f}/day"
        )

        return {
            "item_name": item_name,
            "optimal_price": optimal_price,
            "expected_demand": optimal_demand,
            "expected_profit": optimal_profit,
            "price_elasticity": elasticity,
            "cogs": cogs,
            "margin": margin,
            "margin_pct": margin_pct,
        }

    def optimize_menu(
        self,
        df_historical: pd.DataFrame,
        min_margin: float = 0.10,
        max_price_multiplier: float = 5.0,
        prediction_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Optimize prices for entire menu.

        Args:
            df_historical: Historical sales data with all items
            min_margin: Minimum profit margin (default: 10%)
            max_price_multiplier: Maximum price as multiple of COGS
            prediction_date: Date for prediction

        Returns:
            DataFrame with optimization results for each item
        """
        logger.info(
            f"Optimizing menu with {df_historical['item_name'].nunique()} items"
        )

        results = []

        # Get unique items with their latest COGS
        items_df = (
            df_historical.groupby("item_name")
            .agg({"cogs": "last", "current_price": "last", "category": "last"})
            .reset_index()
        )

        for _, row in items_df.iterrows():
            item_name = row["item_name"]
            cogs = row["cogs"]
            current_price = row["current_price"]
            max_price = cogs * max_price_multiplier

            try:
                result = self.optimize_price_single_item(
                    df_historical=df_historical,
                    item_name=item_name,
                    cogs=cogs,
                    min_margin=min_margin,
                    max_price=max_price,
                    prediction_date=prediction_date,
                )

                # Add current price for comparison
                result["current_price"] = current_price
                result["price_change"] = result["optimal_price"] - current_price
                result["price_change_pct"] = (
                    (result["price_change"] / current_price * 100)
                    if current_price > 0
                    else 0
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to optimize {item_name}: {e}")

                # Add default result
                results.append(
                    {
                        "item_name": item_name,
                        "optimal_price": cogs * (1 + min_margin * 2),
                        "expected_demand": 0,
                        "expected_profit": 0,
                        "price_elasticity": 0,
                        "cogs": cogs,
                        "margin": min_margin * 2,
                        "margin_pct": min_margin * 200,
                        "current_price": current_price,
                        "price_change": 0,
                        "price_change_pct": 0,
                    }
                )

        results_df = pd.DataFrame(results)

        # Sort by expected profit
        results_df = results_df.sort_values("expected_profit", ascending=False)

        logger.info(
            f"Menu optimization complete. "
            f"Avg price change: {results_df['price_change_pct'].mean():.2f}%"
        )

        return results_df

    def compare_scenarios(
        self,
        df_historical: pd.DataFrame,
        item_name: str,
        cogs: float,
        price_range: Tuple[float, float],
        n_points: int = 20,
    ) -> pd.DataFrame:
        """
        Compare demand and profit across different price points.

        Useful for visualizing the demand curve and price sensitivity.

        Args:
            df_historical: Historical sales data
            item_name: Item to analyze
            cogs: Cost of goods sold
            price_range: (min_price, max_price) to test
            n_points: Number of price points to test

        Returns:
            DataFrame with columns: price, demand, profit, margin
        """
        logger.info(f"Comparing {n_points} price scenarios for {item_name}")

        prices = np.linspace(price_range[0], price_range[1], n_points)
        scenarios = []

        for price in prices:
            demand = self.predict_demand_at_price(df_historical, item_name, price)
            profit = demand * (price - cogs)
            margin = ((price - cogs) / cogs * 100) if cogs > 0 else 0

            scenarios.append(
                {
                    "price": price,
                    "demand": demand,
                    "profit": profit,
                    "margin_pct": margin,
                }
            )

        scenarios_df = pd.DataFrame(scenarios)

        # Find optimal in this range
        optimal_idx = scenarios_df["profit"].idxmax()
        optimal_price = scenarios_df.loc[optimal_idx, "price"]

        logger.info(f"Optimal price in range: ${optimal_price:.2f}")

        return scenarios_df
