"""
Menu Price Optimizer using Random Forest Regression.

Applies machine learning to predict optimal menu prices
based on historical performance data.

Updates:
- Uses temporal train-test split (not random) for time-series data
- Integrates with improved risk metrics (v2)
- Better handling of edge cases

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocessor import DataPreprocessor, InsufficientDataError

# Import risk metrics
from src.finance.risk_metrics import (
    RiskMetrics,
    PortfolioMetrics,
    PortfolioAnalyzer,
)

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Base exception for model errors."""

    pass


class ModelNotTrainedError(ModelError):
    """Raised when trying to predict without training."""

    pass


class MenuPriceOptimizer:
    """
    Menu Price Optimization Model using Random Forest Regression.

    Applies Modern Portfolio Theory concepts to menu optimization,
    treating menu items as assets with risk-return profiles.

    Attributes:
        model: RandomForestRegressor instance
        preprocessor: DataPreprocessor for feature engineering
        portfolio_analyzer: PortfolioAnalyzer for risk metrics
        is_trained: Boolean indicating if model has been trained
        feature_importance_: DataFrame of feature importances
        r2_score_: Model R-squared score on test set
        mae_: Mean Absolute Error on test set
        rmse_: Root Mean Squared Error on test set
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,  # Added for regularization
        random_state: int = 42,
        risk_free_rate: float = 0.0225,
        use_temporal_split: bool = True,  # New parameter
    ):
        """
        Initialize the optimizer.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in a leaf node
            random_state: Random seed for reproducibility
            risk_free_rate: Risk-free rate for Sharpe calculations
            use_temporal_split: Use time-based split instead of random
        """
        logger.info("Initializing MenuPriceOptimizer")

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )

        self.preprocessor = DataPreprocessor()

        # Use the improved risk metrics
        self._risk_metrics = RiskMetrics(risk_free_rate=risk_free_rate)
        self._portfolio_metrics = PortfolioMetrics(self._risk_metrics)

        # Keep legacy analyzer for backward compatibility
        self.portfolio_analyzer = PortfolioAnalyzer(risk_free_rate=risk_free_rate)

        self.is_trained = False
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.r2_score_: Optional[float] = None
        self.mae_: Optional[float] = None
        self.rmse_: Optional[float] = None
        self._random_state = random_state
        self._use_temporal_split = use_temporal_split

        logger.info("Model initialized successfully")

    def _temporal_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform temporal train-test split for time-series data.

        Args:
            X: Feature matrix
            y: Target variable
            df: Original DataFrame (for date column)
            test_size: Fraction of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))

        # If we have a date column, sort by it first
        if "date" in df.columns:
            # Create index mapping based on date sort
            sorted_indices = df.sort_values("date").index

            # Map original indices to sorted positions
            idx_mapping = {orig: pos for pos, orig in enumerate(sorted_indices)}

            # Get sorted positions for our current data
            current_indices = df.index[:n_samples]
            sorted_positions = [
                idx_mapping.get(idx, i) for i, idx in enumerate(current_indices)
            ]

            # Sort everything by date
            sort_order = np.argsort(sorted_positions)
            X = X[sort_order]
            y = y[sort_order]

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        return X_train, X_test, y_train, y_test

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the model on menu data.

        Args:
            df: DataFrame with menu item data
            test_size: Fraction of data for testing (default: 0.2)

        Returns:
            Dictionary with training metrics:
            - r2_score: R-squared score
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - cv_r2_mean: Cross-validation R-squared mean
            - cv_r2_std: Cross-validation R-squared std
            - feature_importance: DataFrame of feature importances

        Raises:
            InsufficientDataError: If insufficient data for training
        """
        logger.info(f"Starting model training with {len(df)} samples")

        # Store original DataFrame for temporal split
        original_df = df.copy()

        # Prepare features
        X, y, feature_names = self.preprocessor.prepare_features(df, fit_scaler=True)

        # Check for sufficient variance
        if len(np.unique(y)) < 2:
            raise InsufficientDataError(
                "Target variable has insufficient variance. "
                "All profit margins are identical."
            )

        # Train-test split (temporal or random based on setting)
        if self._use_temporal_split and "date" in original_df.columns:
            logger.info("Using temporal train-test split")
            X_train, X_test, y_train, y_test = self._temporal_train_test_split(
                X, y, original_df, test_size=test_size
            )
        else:
            # Fall back to random split if no date column
            from sklearn.model_selection import train_test_split

            logger.info("Using random train-test split (no date column found)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self._random_state
            )

        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        # Train model
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)

        # Predictions
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        self.r2_score_ = float(r2_score(y_test, y_test_pred))
        self.mae_ = float(mean_absolute_error(y_test, y_test_pred))
        self.rmse_ = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

        # Feature importance
        self.feature_importance_ = pd.DataFrame(
            {"feature": feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        self.is_trained = True

        # Cross-validation with TimeSeriesSplit
        logger.info("Running cross-validation...")
        if self._use_temporal_split:
            cv = TimeSeriesSplit(n_splits=5)
        else:
            cv = 5  # Standard K-Fold

        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="r2")

        # Calculate overfitting gap
        y_train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        overfitting_gap = train_r2 - self.r2_score_

        metrics = {
            "r2_score": self.r2_score_,
            "mae": self.mae_,
            "rmse": self.rmse_,
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "train_r2": float(train_r2),
            "overfitting_gap": float(overfitting_gap),
            "feature_importance": self.feature_importance_,
        }

        # Log warning if overfitting detected
        if overfitting_gap > 0.1:
            logger.warning(
                f"Potential overfitting detected: Train R²={train_r2:.4f}, "
                f"Test R²={self.r2_score_:.4f}, Gap={overfitting_gap:.4f}"
            )

        logger.info(
            f"Training complete. R²: {self.r2_score_:.4f}, MAE: {self.mae_:.4f}"
        )

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal profit margins.

        Args:
            df: DataFrame with menu item data

        Returns:
            Array of predicted profit margins

        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before predictions. Call train() first."
            )

        logger.debug(f"Predicting for {len(df)} items")

        X, _ = self.preprocessor.transform(df)
        predictions = self.model.predict(X)

        return predictions

    def predict_with_intervals(
        self, df: pd.DataFrame, confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals using tree ensemble variance.

        Args:
            df: DataFrame with menu item data
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before predictions. Call train() first."
            )

        X, _ = self.preprocessor.transform(df)

        # Get predictions from all trees
        tree_predictions = np.array(
            [tree.predict(X) for tree in self.model.estimators_]
        )

        predictions = tree_predictions.mean(axis=0)

        alpha = (1 - confidence) / 2
        lower = np.percentile(tree_predictions, alpha * 100, axis=0)
        upper = np.percentile(tree_predictions, (1 - alpha) * 100, axis=0)

        return predictions, lower, upper

    def calculate_portfolio_metrics(
        self, df: pd.DataFrame, use_timeseries: bool = True
    ) -> Dict:
        """
        Calculate portfolio metrics for menu items.

        Args:
            df: DataFrame with menu item data
            use_timeseries: Use time-series based metrics (recommended)

        Returns:
            Dictionary with portfolio metrics
        """
        if use_timeseries and "date" in df.columns:
            # Use improved time-series metrics
            return self._portfolio_metrics.get_portfolio_summary(df)
        else:
            # Fall back to legacy cross-sectional metrics
            return self.portfolio_analyzer.calculate_portfolio_metrics(df)

    def optimize_prices(
        self, df: pd.DataFrame, target_sharpe: float = 1.5, min_margin: float = 0.1
    ) -> pd.DataFrame:
        """
        Optimize menu prices to achieve target Sharpe ratio.

        Args:
            df: DataFrame with menu item data
            target_sharpe: Target Sharpe ratio (default: 1.5)
            min_margin: Minimum acceptable profit margin (default: 10%)

        Returns:
            DataFrame with optimized prices

        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before optimization. Call train() first."
            )

        logger.info(f"Optimizing prices with target Sharpe: {target_sharpe}")

        df = df.copy()

        # Predict optimal margins
        optimal_margins = self.predict(df)

        # Calculate optimal prices
        df["optimal_margin"] = optimal_margins
        df["optimal_price"] = df["cogs"] * (1 + df["optimal_margin"])

        # Ensure minimum margin
        min_price = df["cogs"] * (1 + min_margin)
        df["optimal_price"] = np.maximum(df["optimal_price"], min_price)

        # Handle invalid prices
        df["optimal_price"] = np.where(
            (df["optimal_price"] > 0) & (df["cogs"] > 0),
            df["optimal_price"],
            df["cogs"] * (1 + min_margin * 2),
        )

        # Calculate price change
        df["price_change"] = df["optimal_price"] - df["current_price"]
        df["price_change_pct"] = np.where(
            df["current_price"] > 0, (df["price_change"] / df["current_price"]) * 100, 0
        )

        logger.info(
            f"Optimization complete. Avg change: {df['price_change_pct'].mean():.2f}%"
        )

        return df

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance DataFrame."""
        return self.feature_importance_

    def get_metrics(self) -> Dict:
        """Get current model metrics."""
        if not self.is_trained:
            return {"error": "Model not trained"}

        return {
            "r2_score": self.r2_score_,
            "mae": self.mae_,
            "rmse": self.rmse_,
            "is_trained": self.is_trained,
        }

    def get_risk_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive risk analysis for all menu items.

        Args:
            df: DataFrame with sales data (must include date column)

        Returns:
            Dictionary with per-item risk metrics and recommendations
        """
        if "date" not in df.columns:
            logger.warning(
                "No date column found. Risk analysis requires time-series data."
            )
            return {"error": "Date column required for risk analysis"}

        # Calculate all metrics
        item_metrics = self._risk_metrics.calculate_all_metrics(df)
        recommendations = self._risk_metrics.get_recommendations(item_metrics)

        # Test normality of portfolio returns
        returns_df = self._risk_metrics.calculate_returns_timeseries(df)
        all_returns = returns_df["margin_return"].dropna().values
        normality = self.portfolio_analyzer.test_normality(all_returns)

        return {
            "item_metrics": item_metrics,
            "recommendations": recommendations,
            "normality_test": normality,
            "total_items": len(item_metrics),
        }
