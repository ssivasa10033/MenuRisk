"""
Menu Price Optimizer using Random Forest Regression.

Applies machine learning to predict optimal menu prices
based on historical performance data.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocessor import DataPreprocessor, InsufficientDataError
from src.finance.risk_metrics import PortfolioAnalyzer

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
        random_state: int = 42,
        risk_free_rate: float = 0.0225,
    ):
        """
        Initialize the optimizer.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random seed for reproducibility
            risk_free_rate: Risk-free rate for Sharpe calculations
        """
        logger.info("Initializing MenuPriceOptimizer")

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )

        self.preprocessor = DataPreprocessor()
        self.portfolio_analyzer = PortfolioAnalyzer(risk_free_rate=risk_free_rate)

        self.is_trained = False
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.r2_score_: Optional[float] = None
        self.mae_: Optional[float] = None
        self.rmse_: Optional[float] = None
        self._random_state = random_state

        logger.info("Model initialized successfully")

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

        # Prepare features
        X, y, feature_names = self.preprocessor.prepare_features(df, fit_scaler=True)

        # Check for sufficient variance
        if len(np.unique(y)) < 2:
            raise InsufficientDataError(
                "Target variable has insufficient variance. "
                "All profit margins are identical."
            )

        # Train-test split
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

        # Cross-validation
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="r2")

        metrics = {
            "r2_score": self.r2_score_,
            "mae": self.mae_,
            "rmse": self.rmse_,
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "feature_importance": self.feature_importance_,
        }

        logger.info(
            f"Training complete. R2: {self.r2_score_:.4f}, MAE: {self.mae_:.4f}"
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

    def calculate_portfolio_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate portfolio metrics for menu items.

        Delegates to PortfolioAnalyzer.

        Args:
            df: DataFrame with menu item data

        Returns:
            Dictionary with portfolio metrics
        """
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
