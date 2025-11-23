"""
Visualization and charting for menu optimization results.

Creates comprehensive visualizations for model performance,
portfolio metrics, and category analysis.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Configure matplotlib
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        plt.style.use("ggplot")

sns.set_palette("husl")


class ModelVisualizer:
    """Creates visualizations for menu optimization results."""

    def __init__(self, output_dir: str = "static/charts"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_model_performance(
        self, metrics: Dict, save_path: Optional[str] = None
    ) -> str:
        """
        Plot model performance metrics.

        Args:
            metrics: Dictionary with r2_score, mae, rmse, cv_r2_mean, cv_r2_std
            save_path: Custom save path (optional)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Performance Metrics", fontsize=16, fontweight="bold")

        # R-squared scores
        axes[0, 0].bar(
            ["R² Score", "CV R² Mean"],
            [metrics["r2_score"], metrics["cv_r2_mean"]],
            color=["#2ecc71", "#3498db"],
        )
        axes[0, 0].axhline(y=0.8, color="r", linestyle="--", label="80% Threshold")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("R² Score (Accuracy)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Error metrics
        axes[0, 1].bar(
            ["MAE", "RMSE"],
            [metrics["mae"], metrics["rmse"]],
            color=["#e74c3c", "#c0392b"],
        )
        axes[0, 1].set_ylabel("Error")
        axes[0, 1].set_title("Error Metrics")
        axes[0, 1].grid(True, alpha=0.3)

        # Cross-validation with error bars
        axes[1, 0].barh(
            ["CV R²"],
            [metrics["cv_r2_mean"]],
            xerr=metrics["cv_r2_std"],
            color="#9b59b6",
            capsize=10,
        )
        axes[1, 0].axvline(x=0.8, color="r", linestyle="--", label="80% Threshold")
        axes[1, 0].set_xlabel("Score")
        axes[1, 0].set_title("Cross-Validation R² (with std dev)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Summary text
        status = "PASS (>80%)" if metrics["r2_score"] > 0.8 else "FAIL (<80%)"
        summary_text = f"""
        Model Performance Summary

        R² Score: {metrics['r2_score']:.4f}
        CV R² Mean: {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}
        MAE: {metrics['mae']:.4f}
        RMSE: {metrics['rmse']:.4f}

        Status: {status}
        """
        axes[1, 1].text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[1, 1].axis("off")

        plt.tight_layout()
        output_path = save_path or f"{self.output_dir}/model_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")
        return output_path

    def plot_feature_importance(
        self, feature_importance_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> str:
        """
        Plot feature importance.

        Args:
            feature_importance_df: DataFrame with feature and importance columns
            save_path: Custom save path (optional)

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        df_sorted = feature_importance_df.sort_values("importance", ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))

        ax.barh(df_sorted["feature"], df_sorted["importance"], color=colors)
        ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Feature Importance (Random Forest)", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(
                row["importance"] + 0.01,
                i,
                f"{row['importance']:.3f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        output_path = save_path or f"{self.output_dir}/feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")
        return output_path

    def plot_portfolio_metrics(
        self, portfolio_metrics: Dict, save_path: Optional[str] = None
    ) -> str:
        """
        Plot portfolio analysis metrics.

        Args:
            portfolio_metrics: Dictionary with portfolio metrics
            save_path: Custom save path (optional)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Portfolio Analysis", fontsize=16, fontweight="bold")

        # Sharpe Ratio
        sharpe = portfolio_metrics["sharpe_ratio"]
        color = "#2ecc71" if sharpe > 1.5 else "#f39c12" if sharpe > 0.8 else "#e74c3c"
        axes[0, 0].barh(["Sharpe Ratio"], [sharpe], color=color)
        axes[0, 0].axvline(x=1.5, color="g", linestyle="--", label="Keep Threshold")
        axes[0, 0].axvline(
            x=0.8, color="orange", linestyle="--", label="Monitor Threshold"
        )
        axes[0, 0].set_xlabel("Sharpe Ratio")
        axes[0, 0].set_title("Portfolio Sharpe Ratio")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Returns and Volatility
        axes[0, 1].bar(
            ["Mean Return", "Volatility"],
            [portfolio_metrics["mean_return"], portfolio_metrics["volatility"]],
            color=["#3498db", "#9b59b6"],
        )
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].set_title("Return vs Volatility")
        axes[0, 1].grid(True, alpha=0.3)

        # Recommendations pie chart
        recs = portfolio_metrics["recommendations"]
        if recs:
            rec_counts = pd.Series(list(recs.values())).value_counts()
            colors_pie = {"keep": "#2ecc71", "monitor": "#f39c12", "remove": "#e74c3c"}
            pie_colors = [colors_pie.get(r, "#95a5a6") for r in rec_counts.index]
            axes[1, 0].pie(
                rec_counts.values,
                labels=rec_counts.index,
                autopct="%1.1f%%",
                colors=pie_colors,
            )
        axes[1, 0].set_title("Item Recommendations Distribution")

        # Summary
        recs_list = list(recs.values()) if recs else []
        summary_text = f"""
        Portfolio Summary

        Total Items: {portfolio_metrics['num_items']}
        Mean Return: {portfolio_metrics['mean_return']:.4f}
        Volatility: {portfolio_metrics['volatility']:.4f}
        Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}

        Recommendations:
        - Keep: {recs_list.count('keep')}
        - Monitor: {recs_list.count('monitor')}
        - Remove: {recs_list.count('remove')}
        """
        axes[1, 1].text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[1, 1].axis("off")

        plt.tight_layout()
        output_path = save_path or f"{self.output_dir}/portfolio_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")
        return output_path

    def plot_predictions_vs_actual(
        self,
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot predictions vs actual values.

        Args:
            y_actual: Array of actual values
            y_predicted: Array of predicted values
            save_path: Custom save path (optional)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        axes[0].scatter(y_actual, y_predicted, alpha=0.6, s=50, color="#3498db")
        min_val = min(min(y_actual), min(y_predicted))
        max_val = max(max(y_actual), max(y_predicted))
        axes[0].plot(
            [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect"
        )
        axes[0].set_xlabel("Actual Profit Margin", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Predicted Profit Margin", fontsize=12, fontweight="bold")
        axes[0].set_title("Predictions vs Actual", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_actual - y_predicted
        axes[1].scatter(y_predicted, residuals, alpha=0.6, s=50, color="#e74c3c")
        axes[1].axhline(y=0, color="black", linestyle="--", lw=2)
        axes[1].set_xlabel("Predicted Profit Margin", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Residuals", fontsize=12, fontweight="bold")
        axes[1].set_title("Residuals Plot", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = save_path or f"{self.output_dir}/predictions_vs_actual.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")
        return output_path

    def plot_category_analysis(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot analysis by category.

        Args:
            data: DataFrame with category/season columns
            predictions: Array of predicted margins
            save_path: Custom save path (optional)

        Returns:
            Path to saved figure
        """
        df = data.copy()
        df["predicted_margin"] = predictions

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Category Analysis", fontsize=16, fontweight="bold")

        if "category" in df.columns:
            # Average margins by category
            category_margins = (
                df.groupby("category")["predicted_margin"].mean().sort_values()
            )
            axes[0, 0].barh(
                category_margins.index, category_margins.values, color="#3498db"
            )
            axes[0, 0].set_xlabel("Average Predicted Margin")
            axes[0, 0].set_title("Average Profit Margin by Category")
            axes[0, 0].grid(True, alpha=0.3)

            # Box plot
            categories = df["category"].unique()
            data_by_cat = [
                df[df["category"] == c]["predicted_margin"].values for c in categories
            ]
            axes[0, 1].boxplot(data_by_cat, labels=categories)
            axes[0, 1].set_ylabel("Predicted Margin")
            axes[0, 1].set_title("Margin Distribution by Category")
            axes[0, 1].grid(True, alpha=0.3)

        if "season" in df.columns:
            # Seasonal trend
            season_margins = df.groupby("season")["predicted_margin"].mean()
            season_order = ["Winter", "Spring", "Summer", "Fall"]
            season_margins = season_margins.reindex(
                [s for s in season_order if s in season_margins.index]
            )
            axes[1, 0].plot(
                season_margins.index,
                season_margins.values,
                marker="o",
                linewidth=2,
                markersize=8,
                color="#e74c3c",
            )
            axes[1, 0].set_ylabel("Average Predicted Margin")
            axes[1, 0].set_title("Average Profit Margin by Season")
            axes[1, 0].grid(True, alpha=0.3)

        if "category" in df.columns:
            # Price vs COGS scatter
            for cat in df["category"].unique():
                cat_data = df[df["category"] == cat]
                axes[1, 1].scatter(
                    cat_data["cogs"],
                    cat_data["current_price"],
                    label=cat,
                    alpha=0.6,
                    s=50,
                )
            axes[1, 1].set_xlabel("COGS")
            axes[1, 1].set_ylabel("Current Price")
            axes[1, 1].set_title("Price vs COGS by Category")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = save_path or f"{self.output_dir}/category_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")
        return output_path

    def create_all_charts(
        self,
        metrics: Dict,
        portfolio_metrics: Dict,
        feature_importance: pd.DataFrame,
        data: pd.DataFrame,
        predictions: np.ndarray,
    ) -> Dict[str, str]:
        """
        Create all visualization charts.

        Args:
            metrics: Training metrics dictionary
            portfolio_metrics: Portfolio analysis results
            feature_importance: Feature importance DataFrame
            data: Input DataFrame
            predictions: Model predictions

        Returns:
            Dictionary mapping chart names to file paths
        """
        logger.info("Generating all visualization charts...")

        charts = {}
        charts["model_performance"] = self.plot_model_performance(metrics)
        charts["feature_importance"] = self.plot_feature_importance(feature_importance)
        charts["portfolio_metrics"] = self.plot_portfolio_metrics(portfolio_metrics)
        charts["category_analysis"] = self.plot_category_analysis(data, predictions)

        logger.info(f"Generated {len(charts)} charts in {self.output_dir}")
        return charts
