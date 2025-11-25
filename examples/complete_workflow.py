"""
Complete MenuRisk Workflow Example.

Demonstrates the full pipeline from data generation to price optimization.

This example shows how to:
1. Generate or load historical sales data
2. Train a demand forecasting model
3. Calculate portfolio risk metrics
4. Optimize menu prices
5. Generate actionable recommendations

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

# Import MenuRisk components
from tests.generate_sample_data import generate_synthetic_menu_data
from src.data.feature_engineer import TimeSeriesFeatureEngineer
from src.models.demand_forecaster import DemandForecaster
from src.models.price_optimizer import PriceOptimizer
from src.finance.portfolio_analytics import PortfolioAnalyzer
from config import RISK_FREE_RATE, OPTIMIZATION_CONSTRAINTS, ML_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run complete MenuRisk workflow."""

    print_section("MenuRisk: Complete Workflow Example")

    # =========================================================================
    # STEP 1: Generate or Load Data
    # =========================================================================
    print_section("[1/6] Loading Data")

    logger.info("Generating synthetic menu data...")
    df = generate_synthetic_menu_data(
        n_items=15,
        n_days=180,  # 6 months of history
        include_seasonality=True,
        include_price_changes=True,
        seed=42
    )

    print(f"✅ Loaded {len(df):,} records for {df['item_name'].nunique()} items")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Avg daily sales: {df.groupby('date')['quantity_sold'].sum().mean():.1f} units")

    # Save sample for inspection
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/workflow_sample_data.csv', index=False)
    print(f"   Saved to: data/workflow_sample_data.csv")

    # =========================================================================
    # STEP 2: Train Demand Forecasting Model
    # =========================================================================
    print_section("[2/6] Training Demand Forecasting Model")

    # Split data for training
    cutoff_date = df['date'].max() - pd.Timedelta(days=30)
    train_df = df[df['date'] <= cutoff_date].copy()
    test_df = df[df['date'] > cutoff_date].copy()

    print(f"Train period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Test period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")

    # Feature engineering
    logger.info("Engineering features...")
    feature_engineer = TimeSeriesFeatureEngineer()
    feature_engineer.fit(train_df, cutoff_date.strftime('%Y-%m-%d'))

    train_features = feature_engineer.transform(train_df).dropna()
    test_features = feature_engineer.transform(df[df['date'] > cutoff_date]).dropna()

    # Prepare training data
    exclude_cols = ['date', 'item_name', 'quantity_sold', 'profit_margin']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    numeric_cols = train_features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_train = train_features[numeric_cols].fillna(0)
    y_train = train_features['quantity_sold']

    X_test = test_features[numeric_cols].fillna(0)
    y_test = test_features['quantity_sold']

    # Train model
    logger.info("Training Random Forest demand forecaster...")
    forecaster = DemandForecaster(
        tune_hyperparams=False,  # Set to True for production
        random_state=ML_CONFIG['random_state']
    )

    metrics = forecaster.train(
        X_train=X_train,
        y_train=y_train,
        feature_names=numeric_cols,
        cv_splits=ML_CONFIG['cv_folds']
    )

    print(f"\n✅ Training Complete!")
    print(f"   R² Score: {metrics['r2_score']:.4f}")
    print(f"   MAE: {metrics['mae']:.2f} units")
    print(f"   RMSE: {metrics['rmse']:.2f} units")

    # Evaluate on test set
    test_metrics = forecaster.evaluate(X_test, y_test)
    print(f"\n📊 Test Set Performance:")
    print(f"   R² Score: {test_metrics['test_r2']:.4f}")
    print(f"   MAE: {test_metrics['test_mae']:.2f} units")
    print(f"   MAPE: {test_metrics['test_mape']:.2f}%")

    # Top 5 important features
    feature_importance = metrics['feature_importance']
    print(f"\n🔍 Top 5 Most Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")

    # =========================================================================
    # STEP 3: Calculate Portfolio Risk Metrics
    # =========================================================================
    print_section("[3/6] Calculating Portfolio Risk Metrics")

    logger.info("Analyzing menu portfolio using time-series returns...")
    portfolio_analyzer = PortfolioAnalyzer(risk_free_rate=RISK_FREE_RATE)

    risk_metrics_df = portfolio_analyzer.analyze_menu(df)

    print(f"✅ Portfolio Analysis Complete for {len(risk_metrics_df)} items\n")

    # Display top performers
    print("📈 Top 5 Items by Sharpe Ratio:")
    top_5 = risk_metrics_df.nlargest(5, 'sharpe_ratio')
    for idx, row in top_5.iterrows():
        print(f"   {row['item_name']:25} | Sharpe: {row['sharpe_ratio']:6.2f} | {row['recommendation']}")

    # Display worst performers
    print("\n📉 Bottom 5 Items by Sharpe Ratio:")
    bottom_5 = risk_metrics_df.nsmallest(5, 'sharpe_ratio')
    for idx, row in bottom_5.iterrows():
        print(f"   {row['item_name']:25} | Sharpe: {row['sharpe_ratio']:6.2f} | {row['recommendation']}")

    # =========================================================================
    # STEP 4: Optimize Menu Prices
    # =========================================================================
    print_section("[4/6] Optimizing Menu Prices")

    logger.info("Creating price optimizer...")
    optimizer = PriceOptimizer(
        demand_forecaster=forecaster,
        feature_engineer=feature_engineer
    )

    logger.info("Finding profit-maximizing prices...")
    optimized_menu = optimizer.optimize_menu(
        df_historical=df,
        min_margin=OPTIMIZATION_CONSTRAINTS['min_margin'],
        max_price_multiplier=OPTIMIZATION_CONSTRAINTS['max_price_multiplier']
    )

    print(f"✅ Price Optimization Complete\n")

    # Display top opportunities by profit improvement
    print("💰 Top 5 Price Optimization Opportunities:")
    top_profit = optimized_menu.nlargest(5, 'expected_profit')
    for idx, row in top_profit.iterrows():
        print(f"   {row['item_name']:25} | "
              f"Current: ${row['current_price']:6.2f} → "
              f"Optimal: ${row['optimal_price']:6.2f} | "
              f"Expected Profit: ${row['expected_profit']:7.2f}/day")

    # Display items needing price changes
    significant_changes = optimized_menu[abs(optimized_menu['price_change_pct']) > 5]
    print(f"\n⚠️  {len(significant_changes)} items need significant price changes (>5%):")
    for idx, row in significant_changes.head(5).iterrows():
        direction = "↑" if row['price_change'] > 0 else "↓"
        print(f"   {row['item_name']:25} | "
              f"Change: {direction} {abs(row['price_change_pct']):5.1f}% | "
              f"Elasticity: {row['price_elasticity']:5.2f}")

    # =========================================================================
    # STEP 5: Generate Final Recommendations
    # =========================================================================
    print_section("[5/6] Generating Final Recommendations")

    # Merge risk metrics with pricing optimization
    final_recommendations = optimized_menu.merge(
        risk_metrics_df[['item_name', 'sharpe_ratio', 'recommendation', 'volatility_daily']],
        on='item_name',
        how='left'
    )

    # Categorize items
    print("📋 Menu Recommendations Summary:\n")

    for category in ['KEEP', 'MONITOR', 'REMOVE']:
        items_in_category = final_recommendations[
            final_recommendations['recommendation'].str.contains(category, na=False)
        ]

        print(f"{category} ({len(items_in_category)} items):")

        if len(items_in_category) > 0:
            for idx, item in items_in_category.head(3).iterrows():
                print(f"  • {item['item_name']}")
                print(f"    Sharpe: {item['sharpe_ratio']:5.2f} | "
                      f"Optimal Price: ${item['optimal_price']:6.2f} | "
                      f"Expected Profit: ${item['expected_profit']:7.2f}/day")
        print()

    # =========================================================================
    # STEP 6: Save Results
    # =========================================================================
    print_section("[6/6] Saving Results")

    # Save all results
    final_recommendations.to_csv('data/optimization_results.csv', index=False)
    risk_metrics_df.to_csv('data/risk_metrics.csv', index=False)

    print("✅ Results saved:")
    print("   • data/optimization_results.csv")
    print("   • data/risk_metrics.csv")
    print("   • data/workflow_sample_data.csv")

    # Summary statistics
    print_section("Summary Statistics")

    total_current_profit = (
        df.groupby('item_name').last()['current_price'] *
        df.groupby('item_name')['quantity_sold'].mean()
    ).sum()

    total_expected_profit = final_recommendations['expected_profit'].sum()

    profit_improvement = total_expected_profit - total_current_profit
    improvement_pct = (profit_improvement / total_current_profit * 100) if total_current_profit > 0 else 0

    print(f"📊 Financial Impact:")
    print(f"   Current Daily Profit: ${total_current_profit:,.2f}")
    print(f"   Optimized Daily Profit: ${total_expected_profit:,.2f}")
    print(f"   Improvement: ${profit_improvement:,.2f}/day ({improvement_pct:+.1f}%)")
    print(f"   Annual Impact: ${profit_improvement * 365:,.2f}")

    print(f"\n📈 Portfolio Quality:")
    avg_sharpe = risk_metrics_df['sharpe_ratio'].mean()
    high_sharpe = (risk_metrics_df['sharpe_ratio'] >= 1.5).sum()
    low_sharpe = (risk_metrics_df['sharpe_ratio'] < 0.8).sum()

    print(f"   Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   High performers (Sharpe ≥ 1.5): {high_sharpe} items")
    print(f"   Low performers (Sharpe < 0.8): {low_sharpe} items")

    print_section("✅ Workflow Complete!")

    print("\nNext Steps:")
    print("1. Review optimization_results.csv for detailed recommendations")
    print("2. Implement price changes gradually (A/B testing recommended)")
    print("3. Monitor actual vs. predicted demand")
    print("4. Retrain model monthly with new data")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)
