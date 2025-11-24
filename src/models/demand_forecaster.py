"""
Demand Forecasting Model with proper hyperparameter tuning.

Implements Random Forest regression with TimeSeriesSplit CV
and optional hyperparameter optimization.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value as percentage (0-100)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle zero values by using epsilon
    epsilon = 1e-10
    mask = np.abs(y_true) > epsilon

    if not np.any(mask):
        return 0.0

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(mape)


def directional_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0
) -> float:
    """
    Calculate directional accuracy (% of times prediction captures trend direction).

    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Threshold for considering direction change

    Returns:
        Directional accuracy as percentage (0-100)
    """
    if len(y_true) < 2:
        return 0.0

    # Calculate period-over-period changes
    true_direction = np.diff(y_true) > threshold
    pred_direction = np.diff(y_pred) > threshold

    accuracy = np.mean(true_direction == pred_direction) * 100
    return float(accuracy)


logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    Demand forecasting model with proper hyperparameter tuning.

    Features:
    - TimeSeriesSplit for proper time-series cross-validation
    - RandomizedSearchCV for efficient hyperparameter tuning
    - Prediction intervals using tree ensemble variance
    - Out-of-bag scoring for additional validation
    """

    def __init__(
        self,
        tune_hyperparams: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize the demand forecaster.

        Args:
            tune_hyperparams: Whether to tune hyperparameters during training
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.tune_hyperparams = tune_hyperparams
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: Optional[RandomForestRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained: bool = False

        # Metrics storage
        self.train_metrics: Dict[str, float] = {}
        self.cv_metrics: Dict[str, float] = {}

        logger.info(
            f"Initialized DemandForecaster (tune_hyperparams={tune_hyperparams})"
        )

    def get_param_distributions(self) -> Dict[str, List]:
        """
        Define hyperparameter search space based on best practices.

        Returns:
            Dictionary of parameter names to possible values
        """
        return {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [15, 20, 25, None],  # Deeper trees for RF
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            "bootstrap": [True],
            "max_samples": [0.8, 0.9, None],  # Bootstrap sample size
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get sensible default parameters if not tuning.

        Returns:
            Dictionary of default parameters
        """
        return {
            "n_estimators": 300,
            "max_depth": None,  # Let trees grow fully
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "max_samples": 0.9,
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_splits: int = 5,
        n_iter: int = 50,
    ) -> Dict[str, Any]:
        """
        Train model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features for importance tracking
            cv_splits: Number of time-series CV splits
            n_iter: Number of random search iterations

        Returns:
            Dictionary with training metrics and best parameters
        """
        logger.info(f"Starting training with {len(X_train)} samples")

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X_train, "columns"):
            self.feature_names = list(X_train.columns)

        # Convert to numpy if DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        if self.tune_hyperparams and len(X_train) > 50:
            self._train_with_tuning(X_train, y_train, cv_splits, n_iter)
        else:
            self._train_default(X_train, y_train)

        self.is_trained = True

        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        self.train_metrics = {
            "r2_score": r2_score(y_train, y_train_pred),
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        }

        # Cross-validation metrics
        if len(X_train) > cv_splits * 10:
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=tscv, scoring="r2"
            )
            self.cv_metrics = {
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std()),
            }
        else:
            self.cv_metrics = {"cv_r2_mean": np.nan, "cv_r2_std": np.nan}

        logger.info(
            f"Training complete. R²: {self.train_metrics['r2_score']:.4f}, "
            f"MAE: {self.train_metrics['mae']:.4f}"
        )

        return {
            **self.train_metrics,
            **self.cv_metrics,
            "best_params": self.best_params,
            "feature_importance": self.get_feature_importance(),
        }

    def _train_with_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_splits: int,
        n_iter: int,
    ) -> None:
        """Train with hyperparameter tuning."""
        logger.info("Starting hyperparameter tuning...")

        # Use TimeSeriesSplit for proper CV
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        base_model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True,
        )

        # Randomized search (faster than GridSearch)
        search = RandomizedSearchCV(
            base_model,
            param_distributions=self.get_param_distributions(),
            n_iter=n_iter,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_state,
            return_train_score=True,
        )

        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params = search.best_params_

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV MAE: {-search.best_score_:.2f}")

        # Check for overfitting
        train_score = search.cv_results_["mean_train_score"][search.best_index_]
        val_score = search.best_score_
        logger.info(f"Train MAE: {-train_score:.2f}, Val MAE: {-val_score:.2f}")

        if abs(train_score - val_score) > 0.3 * abs(val_score):
            logger.warning("Potential overfitting detected. Train-val gap is large.")

    def _train_default(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train with default/sensible parameters."""
        logger.info("Training with default parameters...")

        params = self.get_default_params()
        self.best_params = params

        self.model = RandomForestRegressor(
            **params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True,
        )
        self.model.fit(X_train, y_train)

        if hasattr(self.model, "oob_score_"):
            logger.info(f"OOB Score: {self.model.oob_score_:.4f}")

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray:
        """
        Make predictions with optional standard deviation.

        Args:
            X: Features
            return_std: If True, return (predictions, std_dev)

        Returns:
            Predictions array, or tuple (predictions, std_dev)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        if return_std:
            # Get predictions from all trees
            predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            return mean_pred, std_pred
        else:
            return self.model.predict(X)

    def get_prediction_intervals(
        self,
        X: np.ndarray,
        confidence: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using tree ensemble quantiles.

        Args:
            X: Features
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get predictions from all trees
        tree_predictions = np.array(
            [tree.predict(X) for tree in self.model.estimators_]
        )

        predictions = tree_predictions.mean(axis=0)

        alpha = (1 - confidence) / 2
        lower = np.percentile(tree_predictions, alpha * 100, axis=0)
        upper = np.percentile(tree_predictions, (1 - alpha) * 100, axis=0)

        return predictions, lower, upper

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance DataFrame.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            return None

        importances = self.model.feature_importances_

        if self.feature_names and len(self.feature_names) == len(importances):
            return pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)
        else:
            return pd.DataFrame(
                {
                    "feature": [f"feature_{i}" for i in range(len(importances))],
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all model metrics.

        Returns:
            Dictionary with training and CV metrics
        """
        if not self.is_trained:
            return {"error": "Model not trained"}

        return {
            **self.train_metrics,
            **self.cv_metrics,
            "best_params": self.best_params,
            "is_trained": self.is_trained,
            "oob_score": getattr(self.model, "oob_score_", None),
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        include_business_metrics: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data with comprehensive metrics.

        Args:
            X_test: Test features
            y_test: Test targets
            include_business_metrics: Include MAPE and directional accuracy

        Returns:
            Dictionary with evaluation metrics including:
            - test_r2: R-squared score
            - test_mae: Mean Absolute Error
            - test_rmse: Root Mean Squared Error
            - test_mape: Mean Absolute Percentage Error (if enabled)
            - test_directional_accuracy: Directional accuracy % (if enabled)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        predictions = self.predict(X_test)

        metrics = {
            "test_r2": r2_score(y_test, predictions),
            "test_mae": mean_absolute_error(y_test, predictions),
            "test_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        }

        if include_business_metrics:
            metrics["test_mape"] = mean_absolute_percentage_error(y_test, predictions)
            metrics["test_directional_accuracy"] = directional_accuracy(
                y_test, predictions
            )

        return metrics

    def evaluate_per_item(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        item_names: pd.Series,
    ) -> pd.DataFrame:
        """
        Evaluate model performance per item.

        Args:
            X_test: Test features (DataFrame)
            y_test: Test targets (Series)
            item_names: Item names corresponding to test samples

        Returns:
            DataFrame with per-item metrics (MAE, RMSE, MAPE, R²)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        predictions = self.predict(X_test)

        # Create DataFrame for analysis
        results_df = pd.DataFrame(
            {
                "item_name": item_names.values,
                "y_true": y_test.values,
                "y_pred": predictions,
                "error": y_test.values - predictions,
                "abs_error": np.abs(y_test.values - predictions),
            }
        )

        # Calculate per-item metrics
        per_item_metrics = []

        for item in results_df["item_name"].unique():
            item_data = results_df[results_df["item_name"] == item]

            if len(item_data) < 2:
                continue

            y_true_item = item_data["y_true"].values
            y_pred_item = item_data["y_pred"].values

            mae = mean_absolute_error(y_true_item, y_pred_item)
            rmse = np.sqrt(mean_squared_error(y_true_item, y_pred_item))
            r2 = r2_score(y_true_item, y_pred_item)
            mape = mean_absolute_percentage_error(y_true_item, y_pred_item)

            per_item_metrics.append(
                {
                    "item_name": item,
                    "n_samples": len(item_data),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "mape": mape,
                    "mean_actual": y_true_item.mean(),
                    "mean_predicted": y_pred_item.mean(),
                }
            )

        return pd.DataFrame(per_item_metrics).sort_values("mae")

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "best_params": self.best_params,
                "feature_names": self.feature_names,
                "train_metrics": self.train_metrics,
                "cv_metrics": self.cv_metrics,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DemandForecaster":
        """
        Load model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded DemandForecaster instance
        """
        import joblib

        data = joblib.load(path)

        instance = cls(tune_hyperparams=False)
        instance.model = data["model"]
        instance.best_params = data["best_params"]
        instance.feature_names = data["feature_names"]
        instance.train_metrics = data["train_metrics"]
        instance.cv_metrics = data["cv_metrics"]
        instance.is_trained = True

        logger.info(f"Model loaded from {path}")
        return instance
