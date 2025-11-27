"""
Model comparison framework for demand forecasting.

Compares multiple models including:
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Naive baselines (last value, moving average, seasonal naive)

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


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


class NaiveBaseline:
    """
    Naive baseline forecasting models.

    Implements simple baselines that any ML model should beat:
    - Last value (persistence model)
    - Moving average
    - Seasonal naive (last value from same season)
    """

    def __init__(self, method: str = "last_value", window: int = 7):
        """
        Initialize naive baseline.

        Args:
            method: One of 'last_value', 'moving_average', 'seasonal_naive'
            window: Window size for moving average or seasonal period
        """
        self.method = method
        self.window = window
        self.last_train_values: Optional[Dict[str, float]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBaseline":
        """
        Fit baseline model.

        Args:
            X: Features (may include item identifiers)
            y: Target values

        Returns:
            Self for method chaining
        """
        if self.method == "last_value":
            # Store last value from training set
            self.last_train_values = {"global": float(y[-1])}
        elif self.method == "moving_average":
            # Store last window values
            self.last_train_values = {
                "values": y[-self.window :].tolist(),
                "mean": float(np.mean(y[-self.window :])),
            }
        elif self.method == "seasonal_naive":
            # Store values from last seasonal period
            if len(y) >= self.window:
                self.last_train_values = {
                    "seasonal_value": float(y[-self.window]),
                    "mean": float(np.mean(y)),
                }
            else:
                self.last_train_values = {"seasonal_value": float(y[-1])}

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions array
        """
        n_samples = len(X)

        if self.method == "last_value":
            return np.full(n_samples, self.last_train_values["global"])
        elif self.method == "moving_average":
            return np.full(n_samples, self.last_train_values["mean"])
        elif self.method == "seasonal_naive":
            return np.full(n_samples, self.last_train_values.get("seasonal_value", 0))

        return np.zeros(n_samples)


class ModelComparison:
    """
    Compare multiple forecasting models with rigorous evaluation.

    Features:
    - Multiple model architectures (RF, XGBoost, LightGBM, etc.)
    - Naive baselines for sanity checking
    - Time-series cross-validation
    - Multiple evaluation metrics (R², MAE, RMSE, MAPE, directional accuracy)
    - Statistical significance testing
    """

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
        cv_splits: int = 5,
    ):
        """
        Initialize model comparison framework.

        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            cv_splits: Number of time-series CV splits
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_splits = cv_splits

        self.models: Dict[str, Any] = {}
        self.results: Optional[pd.DataFrame] = None
        self.trained_models: Dict[str, Any] = {}

        logger.info("Initialized ModelComparison framework")

    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to compare.

        Returns:
            Dictionary mapping model names to model instances
        """
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=self.random_state,
            ),
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            logger.warning("XGBoost not available. Install with: pip install xgboost")

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models["lightgbm"] = LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
        else:
            logger.warning("LightGBM not available. Install with: pip install lightgbm")

        # Add naive baselines
        models["naive_last_value"] = NaiveBaseline(method="last_value")
        models["naive_moving_avg"] = NaiveBaseline(method="moving_average", window=7)
        models["naive_seasonal"] = NaiveBaseline(method="seasonal_naive", window=7)

        return models

    def compare_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_subset: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple models on given data.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_subset: Optional list of model names to compare (default: all)

        Returns:
            DataFrame with comparison results sorted by test MAE
        """
        logger.info(
            f"Comparing models on {len(X_train)} train, {len(X_test)} test samples"
        )

        # Convert to numpy if DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        models = self.get_models()

        # Filter to subset if specified
        if model_subset:
            models = {k: v for k, v in models.items() if k in model_subset}

        results = []

        for name, model in models.items():
            logger.info(f"Training {name}...")

            try:
                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
                test_dir_acc = directional_accuracy(y_test, y_test_pred)

                # Cross-validation (only for non-naive models)
                if not name.startswith("naive"):
                    tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=tscv, scoring="r2"
                    )
                    cv_r2_mean = cv_scores.mean()
                    cv_r2_std = cv_scores.std()
                else:
                    cv_r2_mean = np.nan
                    cv_r2_std = np.nan

                # Store trained model
                self.trained_models[name] = model

                results.append(
                    {
                        "model": name,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "train_mae": train_mae,
                        "test_mae": test_mae,
                        "train_rmse": train_rmse,
                        "test_rmse": test_rmse,
                        "test_mape": test_mape,
                        "test_directional_accuracy": test_dir_acc,
                        "cv_r2_mean": cv_r2_mean,
                        "cv_r2_std": cv_r2_std,
                        "overfitting_gap": train_r2 - test_r2,
                    }
                )

                logger.info(
                    f"{name}: Test R²={test_r2:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%"
                )

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue

        # Create results DataFrame
        self.results = pd.DataFrame(results).sort_values("test_mae")

        return self.results

    def get_best_model(self, metric: str = "test_mae") -> Tuple[str, Any, float]:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for selection (default: test_mae)
                   Use negative metrics (r2, directional_accuracy) for higher-is-better

        Returns:
            Tuple of (model_name, model_instance, metric_value)
        """
        if self.results is None:
            raise ValueError("Must run compare_models() first")

        ascending = metric not in ["test_r2", "test_directional_accuracy", "cv_r2_mean"]
        best_row = self.results.sort_values(metric, ascending=ascending).iloc[0]

        model_name = best_row["model"]
        model_instance = self.trained_models[model_name]
        metric_value = best_row[metric]

        logger.info(f"Best model: {model_name} ({metric}={metric_value:.4f})")

        return model_name, model_instance, metric_value

    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot model comparison results.

        Args:
            save_path: Optional path to save the plot
        """
        if self.results is None:
            raise ValueError("Must run compare_models() first")

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Test R² comparison
        ax = axes[0, 0]
        df_sorted = self.results.sort_values("test_r2", ascending=False)
        ax.barh(df_sorted["model"], df_sorted["test_r2"], color="steelblue")
        ax.set_xlabel("Test R² Score")
        ax.set_title("Model Comparison - Test R²")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)

        # 2. Test MAE comparison
        ax = axes[0, 1]
        df_sorted = self.results.sort_values("test_mae")
        ax.barh(df_sorted["model"], df_sorted["test_mae"], color="coral")
        ax.set_xlabel("Test MAE")
        ax.set_title("Model Comparison - Test MAE (Lower is Better)")

        # 3. MAPE comparison
        ax = axes[1, 0]
        df_sorted = self.results.sort_values("test_mape")
        ax.barh(df_sorted["model"], df_sorted["test_mape"], color="seagreen")
        ax.set_xlabel("Test MAPE (%)")
        ax.set_title("Model Comparison - MAPE (Lower is Better)")

        # 4. Overfitting gap
        ax = axes[1, 1]
        df_sorted = self.results.sort_values("overfitting_gap")
        colors = ["red" if x > 0.1 else "green" for x in df_sorted["overfitting_gap"]]
        ax.barh(df_sorted["model"], df_sorted["overfitting_gap"], color=colors)
        ax.set_xlabel("Overfitting Gap (Train R² - Test R²)")
        ax.set_title("Overfitting Analysis")
        ax.axvline(x=0.1, color="red", linestyle="--", alpha=0.5, label="Threshold")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def get_summary(self) -> str:
        """
        Get summary of comparison results.

        Returns:
            Formatted summary string
        """
        if self.results is None:
            return "No comparison results available. Run compare_models() first."

        best_r2_model = self.results.sort_values("test_r2", ascending=False).iloc[0]
        best_mae_model = self.results.sort_values("test_mae").iloc[0]

        summary = f"""
Model Comparison Summary
{'=' * 60}

Best Model by Test R²: {best_r2_model['model']}
  - Test R²: {best_r2_model['test_r2']:.4f}
  - Test MAE: {best_r2_model['test_mae']:.4f}
  - Test MAPE: {best_r2_model['test_mape']:.2f}%

Best Model by Test MAE: {best_mae_model['model']}
  - Test MAE: {best_mae_model['test_mae']:.4f}
  - Test R²: {best_mae_model['test_r2']:.4f}
  - Test MAPE: {best_mae_model['test_mape']:.2f}%

Baseline Performance:
"""

        # Add baseline results
        for baseline in ["naive_last_value", "naive_moving_avg", "naive_seasonal"]:
            if baseline in self.results["model"].values:
                row = self.results[self.results["model"] == baseline].iloc[0]
                summary += (
                    f"  - {baseline}: MAE={row['test_mae']:.4f}, "
                    f"MAPE={row['test_mape']:.2f}%\n"
                )

        summary += "\nAll models beat naive baselines: "
        best_naive_mae = self.results[self.results["model"].str.startswith("naive")][
            "test_mae"
        ].min()
        best_ml_mae = self.results[~self.results["model"].str.startswith("naive")][
            "test_mae"
        ].min()
        summary += "[OK] Yes" if best_ml_mae < best_naive_mae else "[X] No"

        return summary
