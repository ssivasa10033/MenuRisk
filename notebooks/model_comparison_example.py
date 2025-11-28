"""
Model Comparison Example - MenuRisk

This example demonstrates the comprehensive model comparison framework
for demand forecasting, comparing multiple ML models against naive baselines.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.model_comparison import ModelComparison
from src.data.feature_engineer import create_train_test_split
from src.data.loader import DataLoader
from src.finance.risk_metrics import PortfolioAnalyzer

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %% Load Data
print("=" * 60)
print("MENURISK MODEL COMPARISON FRAMEWORK")
print("=" * 60)

# Load your menu data
# For this example, we'll use the data loader
loader = DataLoader()
df = loader.load_data("data/sample_menu_data.csv")  # Update path as needed

print(f"\nLoaded {len(df)} rows of data")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Number of items: {df['item_name'].nunique()}")

# %% Feature Engineering with Time-Series Split
print("\n" + "=" * 60)
print("FEATURE ENGINEERING & TRAIN-TEST SPLIT")
print("=" * 60)

# Create proper time-series train-test split
X_train, X_test, y_train, y_test, feature_engineer = create_train_test_split(
    df, test_size_days=30, target_col="quantity_sold"
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Features: {X_train.shape[1]}")

# %% Model Comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Initialize comparison framework
comparison = ModelComparison(random_state=42, n_jobs=-1, cv_splits=5)

# Compare all models
print("\nTraining and evaluating models...")
results = comparison.compare_models(X_train, X_test, y_train, y_test)

# Display results
print("\n" + "-" * 60)
print("COMPARISON RESULTS")
print("-" * 60)
print(
    results[
        [
            "model",
            "test_r2",
            "test_mae",
            "test_rmse",
            "test_mape",
            "test_directional_accuracy",
        ]
    ].to_string(index=False)
)

# %% Get Best Model
print("\n" + "=" * 60)
print("BEST MODEL SELECTION")
print("=" * 60)

# Get best model by MAE
best_name, best_model, best_mae = comparison.get_best_model(metric="test_mae")

print(f"\nBest Model: {best_name}")
print(f"Test MAE: {best_mae:.4f}")

# Get full metrics for best model
best_metrics = results[results["model"] == best_name].iloc[0]
print(f"\nFull Metrics:")
print(f"  R²: {best_metrics['test_r2']:.4f}")
print(f"  MAE: {best_metrics['test_mae']:.4f}")
print(f"  RMSE: {best_metrics['test_rmse']:.4f}")
print(f"  MAPE: {best_metrics['test_mape']:.2f}%")
print(f"  Directional Accuracy: {best_metrics['test_directional_accuracy']:.2f}%")
print(f"  Overfitting Gap: {best_metrics['overfitting_gap']:.4f}")

# %% Baseline Comparison
print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

# Get naive baseline results
naive_results = results[results["model"].str.startswith("naive")]
ml_results = results[~results["model"].str.startswith("naive")]

print("\nNaive Baselines:")
print(naive_results[["model", "test_mae", "test_mape"]].to_string(index=False))

print("\nML Models:")
print(ml_results[["model", "test_mae", "test_mape"]].to_string(index=False))

# Calculate improvement over best baseline
best_baseline_mae = naive_results["test_mae"].min()
best_ml_mae = ml_results["test_mae"].min()
improvement = ((best_baseline_mae - best_ml_mae) / best_baseline_mae) * 100

print(f"\n[OK] Best ML model improves over best baseline by {improvement:.1f}%")

# %% Summary Report
print("\n" + "=" * 60)
print(comparison.get_summary())
print("=" * 60)

# %% Visualization
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Plot comparison
comparison.plot_comparison(save_path="results/model_comparison.png")
print("\n[OK] Saved comparison plot to results/model_comparison.png")

# %% Per-Item Performance Analysis
print("\n" + "=" * 60)
print("PER-ITEM PERFORMANCE ANALYSIS")
print("=" * 60)

# Use the DemandForecaster with the best model
from src.models.demand_forecaster import DemandForecaster

# Train DemandForecaster for per-item analysis
forecaster = DemandForecaster(tune_hyperparams=False, random_state=42)
forecaster.train(X_train, y_train)

# Get per-item metrics (requires item names in test data)
if "item_name" in df.columns:
    # Get item names for test set
    test_indices = y_test.index
    item_names = df.loc[test_indices, "item_name"]

    per_item = forecaster.evaluate_per_item(
        pd.DataFrame(X_test), pd.Series(y_test), pd.Series(item_names)
    )

    print("\nTop 5 Best Predicted Items:")
    print(per_item.head(5)[["item_name", "mae", "mape", "r2"]].to_string(index=False))

    print("\nTop 5 Worst Predicted Items:")
    print(per_item.tail(5)[["item_name", "mae", "mape", "r2"]].to_string(index=False))

# %% Financial Risk Analysis
print("\n" + "=" * 60)
print("FINANCIAL RISK ANALYSIS")
print("=" * 60)

# Calculate portfolio metrics
analyzer = PortfolioAnalyzer()
portfolio_metrics = analyzer.calculate_portfolio_metrics(df)

print(f"\nPortfolio Metrics:")
print(f"  Mean Return: {portfolio_metrics['mean_return']:.4f}")
print(f"  Volatility: {portfolio_metrics['volatility']:.4f}")
print(f"  Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
print(f"  Number of Items: {portfolio_metrics['num_items']}")

# Test normality of returns
if len(df) > 0:
    # Calculate returns
    df_copy = df.copy()
    df_copy["profit_margin"] = np.where(
        df_copy["cogs"] > 0,
        (df_copy["current_price"] - df_copy["cogs"]) / df_copy["cogs"],
        0,
    )
    returns = df_copy["profit_margin"].values
    returns = returns[~np.isnan(returns)]
    returns = returns[returns != np.inf]

    if len(returns) > 3:
        normality = analyzer.test_normality(returns)

        print(f"\nNormality Testing:")
        print(f"  Is Normal: {normality['is_normal']}")
        print(f"  Shapiro p-value: {normality['shapiro_p_value']:.4f}")
        print(f"  Skewness: {normality['skewness']:.4f}")
        print(f"  Kurtosis: {normality['kurtosis']:.4f}")
        print(f"\n  Recommendation: {normality['recommendation']}")

# %% Feature Importance Analysis
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

importance = forecaster.get_feature_importance()

if importance is not None:
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = importance.head(15)
    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=300, bbox_inches="tight")
    print("\n[OK] Saved feature importance plot to results/feature_importance.png")

# %% Prediction Intervals
print("\n" + "=" * 60)
print("PREDICTION INTERVALS")
print("=" * 60)

# Get predictions with confidence intervals
predictions, lower, upper = forecaster.get_prediction_intervals(
    X_test[:50], confidence=0.90
)

print("\nSample predictions with 90% confidence intervals:")
sample_df = pd.DataFrame(
    {
        "actual": y_test[:50].values,
        "predicted": predictions,
        "lower_90": lower,
        "upper_90": upper,
        "within_interval": (y_test[:50].values >= lower)
        & (y_test[:50].values <= upper),
    }
)

print(sample_df.head(10).to_string(index=False))

coverage = sample_df["within_interval"].mean() * 100
print(f"\nInterval Coverage: {coverage:.1f}% (target: 90%)")

# %% Recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

print("\nBased on the model comparison:")
print(f"1. [OK] Best performing model: {best_name}")
print(f"2. [OK] Achieves {improvement:.1f}% improvement over naive baseline")
print(f"3. [OK] Test MAPE: {best_metrics['test_mape']:.2f}% (business-friendly metric)")
print(f"4. [OK] Directional Accuracy: {best_metrics['test_directional_accuracy']:.2f}%")

if best_metrics["overfitting_gap"] < 0.1:
    print("5. [OK] Low overfitting risk (train-test gap < 0.1)")
else:
    gap = best_metrics['overfitting_gap']
    print(f"5. [WARNING] Potential overfitting (train-test gap: {gap:.4f})")

print("\nNext Steps:")
print("- Use this model for production forecasting")
print("- Monitor model performance over time")
print("- Retrain periodically with new data")
print("- Consider ensemble methods for even better performance")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
