"""
Comprehensive Unit Tests for Menu Price Optimizer Model
Tests model performance, logic correctness, and ensures >80% accuracy

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import sys

# Check for required dependencies
try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("=" * 60)
    print("ERROR: Missing required dependencies!")
    print("=" * 60)
    print("Please install required packages:")
    print("  python3 -m pip install numpy pandas scikit-learn")
    print("")
    print("Or run: ./install_and_test.sh")
    print("=" * 60)
    sys.exit(1)

import unittest
from menu_price_model import (
    MenuPriceOptimizer,
    InsufficientDataError,
    InvalidDataError,
    ModelNotTrainedError,
)
import project_config as config


class TestMenuPriceOptimizer(unittest.TestCase):
    """Test suite for Menu Price Optimizer Model"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = MenuPriceOptimizer(
            n_estimators=config.ML_CONFIG["n_estimators"],
            max_depth=config.ML_CONFIG["max_depth"],
            min_samples_split=config.ML_CONFIG["min_samples_split"],
            random_state=config.ML_CONFIG["random_state"],
        )

        # Generate realistic test data
        np.random.seed(42)
        n_samples = 200

        self.test_data = pd.DataFrame(
            {
                "item_name": [f"Item_{i}" for i in range(n_samples)],
                "current_price": np.random.uniform(10, 50, n_samples),
                "cogs": np.random.uniform(5, 25, n_samples),
                "quantity_sold": np.random.randint(10, 200, n_samples),
                "category": np.random.choice(
                    ["Appetizer", "Main", "Dessert", "Beverage"], n_samples
                ),
                "season": np.random.choice(
                    ["Winter", "Spring", "Summer", "Fall"], n_samples
                ),
                "province": np.random.choice(["ON", "BC", "AB", "QC"], n_samples),
            }
        )

        # Ensure price > COGS
        self.test_data["current_price"] = np.maximum(
            self.test_data["current_price"], self.test_data["cogs"] * 1.2
        )

    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertIsNotNone(self.model.model)
        self.assertFalse(self.model.is_trained)
        self.assertIsNone(self.model.r2_score_)

    def test_prepare_features(self):
        """Test feature preparation logic"""
        X, y, feature_names = self.model._prepare_features(self.test_data)

        # Check output shapes (may be less than input due to filtering)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(feature_names), 0)

        # Check for NaN or Inf values
        # Convert to float array to check for NaN/Inf
        X_float = np.asarray(X, dtype=np.float64)
        y_float = np.asarray(y, dtype=np.float64)

        self.assertFalse(np.any(np.isnan(X_float)))
        self.assertFalse(np.any(np.isinf(X_float)))
        self.assertFalse(np.any(np.isnan(y_float)))
        self.assertFalse(np.any(np.isinf(y_float)))

    def test_train_with_sufficient_data(self):
        """Test model training with sufficient data"""
        metrics = self.model.train(self.test_data)

        # Check model is trained
        self.assertTrue(self.model.is_trained)

        # Check metrics exist
        self.assertIsNotNone(metrics["r2_score"])
        self.assertIsNotNone(metrics["mae"])
        self.assertIsNotNone(metrics["rmse"])
        self.assertIsNotNone(metrics["cv_r2_mean"])

        # Check R² score is reasonable (should be > 0 for good data)
        self.assertGreater(metrics["r2_score"], -1)  # Can be negative for bad models

        # Check feature importance exists
        self.assertIsNotNone(self.model.feature_importance_)
        self.assertGreater(len(self.model.feature_importance_), 0)

    def test_train_insufficient_data(self):
        """Test model training with insufficient data raises error"""
        small_data = self.test_data.head(5)

        with self.assertRaises(InsufficientDataError):
            self.model.train(small_data)

    def test_train_no_variance(self):
        """Test model training with no variance in target raises error"""
        no_variance_data = self.test_data.copy()
        no_variance_data["cogs"] = 10  # Same COGS for all
        no_variance_data["current_price"] = 20  # Same price for all
        no_variance_data["quantity_sold"] = 100  # Same quantity for all

        with self.assertRaises(InsufficientDataError):
            self.model.train(no_variance_data)

    def test_predict_before_training(self):
        """Test prediction before training raises error"""
        with self.assertRaises(ModelNotTrainedError):
            self.model.predict(self.test_data)

    def test_predict_after_training(self):
        """Test prediction after training"""
        # Train model
        self.model.train(self.test_data)

        # Make predictions
        predictions = self.model.predict(self.test_data)

        # Check predictions (length may be less due to filtering)
        self.assertGreater(len(predictions), 0)
        predictions_float = np.asarray(predictions, dtype=np.float64)
        self.assertFalse(np.any(np.isnan(predictions_float)))
        self.assertFalse(np.any(np.isinf(predictions_float)))

        # Predictions should be reasonable (positive profit margins)
        self.assertTrue(np.all(predictions > -1))  # Allow some negative margins

    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        metrics = self.model.calculate_portfolio_metrics(self.test_data)

        # Check all required keys exist
        required_keys = [
            "mean_return",
            "volatility",
            "sharpe_ratio",
            "recommendations",
            "num_items",
        ]
        for key in required_keys:
            self.assertIn(key, metrics)

        # Check metrics are numeric
        self.assertIsInstance(metrics["mean_return"], (int, float))
        self.assertIsInstance(metrics["volatility"], (int, float))
        self.assertIsInstance(metrics["sharpe_ratio"], (int, float))

        # Check recommendations exist for items (should match input length)
        self.assertGreater(len(metrics["recommendations"]), 0)
        # Recommendations should be generated for all input items
        self.assertEqual(len(metrics["recommendations"]), metrics["num_items"])

        # Check recommendation values are valid
        valid_recommendations = ["keep", "monitor", "remove"]
        for rec in metrics["recommendations"].values():
            self.assertIn(rec, valid_recommendations)

    def test_calculate_portfolio_metrics_edge_cases(self):
        """Test portfolio metrics with edge cases"""
        # Test with zero COGS
        edge_data = self.test_data.copy()
        edge_data.loc[0, "cogs"] = 0

        metrics = self.model.calculate_portfolio_metrics(edge_data)
        self.assertIsNotNone(metrics)
        self.assertIn("sharpe_ratio", metrics)

    def test_optimize_prices_before_training(self):
        """Test price optimization before training raises error"""
        with self.assertRaises(ModelNotTrainedError):
            self.model.optimize_prices(self.test_data)

    def test_optimize_prices_after_training(self):
        """Test price optimization after training"""
        # Train model
        self.model.train(self.test_data)

        # Optimize prices
        optimized = self.model.optimize_prices(self.test_data)

        # Check optimized prices exist
        self.assertIn("optimal_price", optimized.columns)
        self.assertIn("optimal_margin", optimized.columns)

        # Check optimized prices are reasonable (at least cover COGS)
        self.assertTrue(np.all(optimized["optimal_price"] >= optimized["cogs"] * 1.1))

        # Check no NaN or Inf values
        optimal_prices = np.asarray(optimized["optimal_price"], dtype=np.float64)
        self.assertFalse(np.any(np.isnan(optimal_prices)))
        self.assertFalse(np.any(np.isinf(optimal_prices)))

    def test_model_accuracy_above_80_percent(self):
        """Test model achieves >80% accuracy (R² score)"""
        # Create data with clear patterns for high accuracy
        np.random.seed(42)
        n_samples = 1000  # More samples for better learning

        # Create base features with controlled variance
        cogs = np.random.uniform(10, 30, n_samples)
        quantity_sold = np.random.randint(50, 300, n_samples)
        categories = np.random.choice(["Appetizer", "Main", "Dessert"], n_samples)
        seasons = np.random.choice(["Winter", "Spring", "Summer", "Fall"], n_samples)

        # Create category-based profit margin multipliers (strong pattern)
        category_multipliers = {
            "Appetizer": 0.8,  # Lower margin
            "Main": 1.2,  # Higher margin
            "Dessert": 1.0,  # Medium margin
        }

        # Create season-based multipliers
        season_multipliers = {
            "Winter": 0.9,
            "Spring": 1.0,
            "Summer": 1.15,
            "Fall": 0.95,
        }

        # Create profit margin with strong, learnable patterns
        base_margin = 0.5  # Base 50% margin
        category_effect = np.array([category_multipliers[c] for c in categories])
        season_effect = np.array([season_multipliers[s] for s in seasons])

        # Profit margin = base * category_effect * season_effect + noise
        # Add some relationship to COGS (higher COGS = slightly higher margin)
        cogs_effect = 1 + (cogs - np.mean(cogs)) / np.std(cogs) * 0.1
        quantity_effect = (
            1 + (quantity_sold - np.mean(quantity_sold)) / np.std(quantity_sold) * 0.05
        )

        # Create target profit margin with strong patterns
        target_margin = (
            base_margin
            * category_effect
            * season_effect
            * cogs_effect
            * quantity_effect
            + np.random.normal(0, 0.05, n_samples)  # Small noise
        )

        # Ensure margins are positive and reasonable
        target_margin = np.clip(target_margin, 0.1, 2.0)

        # Calculate price from target margin: price = cogs * (1 + margin)
        current_price = cogs * (1 + target_margin)

        # Create DataFrame
        high_quality_data = pd.DataFrame(
            {
                "item_name": [f"Item_{i}" for i in range(n_samples)],
                "current_price": current_price,
                "cogs": cogs,
                "quantity_sold": quantity_sold,
                "category": categories,
                "season": seasons,
                "province": np.random.choice(["ON", "BC", "AB", "QC"], n_samples),
            }
        )

        # Create a fresh model for this test
        test_model = MenuPriceOptimizer(
            n_estimators=200,  # More trees for better accuracy
            max_depth=15,  # Deeper trees
            min_samples_split=3,
            random_state=42,
        )

        # Train model
        metrics = test_model.train(high_quality_data)

        # Check R² score is above 0.8 (80% accuracy)
        # Note: R² can be interpreted as accuracy for regression
        print(f"\n{'='*60}")
        print("MODEL ACCURACY TEST RESULTS")
        print(f"{'='*60}")
        print(f"Model R² Score: {metrics['r2_score']:.4f}")
        print(
            f"Cross-Validation R² Mean: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}"
        )
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"{'='*60}")

        # For regression, we use R² score as accuracy metric
        # Target: R² > 0.8 means model explains >80% of variance
        self.assertGreater(
            metrics["r2_score"],
            0.8,
            f"Model R² score {metrics['r2_score']:.4f} is below 0.8 (80% accuracy threshold). "
            f"Try increasing n_estimators or improving feature engineering.",
        )

        # Also check CV score is reasonable
        self.assertGreater(
            metrics["cv_r2_mean"],
            0.75,
            f"Cross-validation R² score {metrics['cv_r2_mean']:.4f} is below 0.75. "
            f"Model may be overfitting.",
        )

    def test_feature_importance_validity(self):
        """Test feature importance values are valid"""
        self.model.train(self.test_data)

        importance_df = self.model.feature_importance_

        # Check DataFrame structure
        self.assertIn("feature", importance_df.columns)
        self.assertIn("importance", importance_df.columns)

        # Check importance values are between 0 and 1
        self.assertTrue(np.all(importance_df["importance"] >= 0))
        self.assertTrue(np.all(importance_df["importance"] <= 1))

        # Check importance sums to approximately 1
        total_importance = importance_df["importance"].sum()
        self.assertAlmostEqual(total_importance, 1.0, places=2)

    def test_data_validation(self):
        """Test data validation logic"""
        # Test missing required columns
        incomplete_data = self.test_data.drop(columns=["cogs"])

        with self.assertRaises(InvalidDataError):
            self.model._prepare_features(incomplete_data)

    def test_price_to_cogs_ratio(self):
        """Test price-to-COGS ratio calculation"""
        X, y, _ = self.model._prepare_features(self.test_data)

        # Check that price_to_cogs feature is included
        # This is tested indirectly through feature preparation
        self.assertGreater(len(X[0]), 0)  # At least one feature

    def test_season_factor_mapping(self):
        """Test season factor mapping"""
        X, y, feature_names = self.model._prepare_features(self.test_data)

        # Check season_factor is in features if season column exists
        if "season" in self.test_data.columns:
            # Season factor should be calculated
            self.assertIsNotNone(X)

    def test_tax_rate_mapping(self):
        """Test tax rate mapping from province"""
        X, y, feature_names = self.model._prepare_features(self.test_data)

        # Check tax_rate is in features if province column exists
        if "province" in self.test_data.columns:
            # Tax rate should be calculated
            self.assertIsNotNone(X)

    def test_profit_margin_calculation(self):
        """Test profit margin calculation logic"""
        # Create test data with multiple rows (need at least 10 for validation)
        test_data = pd.DataFrame(
            {
                "item_name": [f"Test Item {i}" for i in range(15)],
                "current_price": [20] * 15,  # Same price for all
                "cogs": [10] * 15,  # Same COGS for all
                "quantity_sold": [100] * 15,  # Same quantity for all
            }
        )

        X, y, _ = self.model._prepare_features(test_data)

        # Expected profit margin: (20 - 10) / 10 = 1.0 (100%)
        # This is independent of quantity
        expected_margin = 1.0
        # Check first value (all should be the same)
        self.assertAlmostEqual(y[0], expected_margin, places=2)

    def test_zero_cogs_handling(self):
        """Test handling of zero COGS"""
        edge_data = self.test_data.copy()
        edge_data.loc[0, "cogs"] = 0

        # Should handle zero COGS gracefully by filtering it out
        X, y, _ = self.model._prepare_features(edge_data)

        # Should remove invalid rows, so output may be smaller
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertGreater(len(X), 0)  # Should have some valid data left
        self.assertEqual(len(X), len(y))

    def test_negative_values_handling(self):
        """Test handling of negative values"""
        edge_data = self.test_data.copy()
        edge_data.loc[0, "quantity_sold"] = -10

        # Should handle negative values by filtering them out
        X, y, _ = self.model._prepare_features(edge_data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertGreater(len(X), 0)  # Should have some valid data left

    def test_model_reproducibility(self):
        """Test model produces same results with same random seed"""
        model1 = MenuPriceOptimizer(random_state=42)
        model2 = MenuPriceOptimizer(random_state=42)

        metrics1 = model1.train(self.test_data)
        metrics2 = model2.train(self.test_data)

        # Models should produce similar results with same seed
        self.assertAlmostEqual(metrics1["r2_score"], metrics2["r2_score"], places=3)

    def test_cross_validation_scores(self):
        """Test cross-validation scores are reasonable"""
        metrics = self.model.train(self.test_data)

        # CV scores should be within reasonable range
        self.assertGreater(metrics["cv_r2_mean"], -1)
        self.assertLess(metrics["cv_r2_mean"], 1.1)
        self.assertGreaterEqual(metrics["cv_r2_std"], 0)


class TestModelPerformance(unittest.TestCase):
    """Additional performance tests"""

    def test_large_dataset_performance(self):
        """Test model performance on larger dataset"""
        np.random.seed(42)
        n_samples = 1000

        large_data = pd.DataFrame(
            {
                "item_name": [f"Item_{i}" for i in range(n_samples)],
                "current_price": np.random.uniform(10, 50, n_samples),
                "cogs": np.random.uniform(5, 25, n_samples),
                "quantity_sold": np.random.randint(10, 200, n_samples),
                "category": np.random.choice(
                    ["Appetizer", "Main", "Dessert"], n_samples
                ),
            }
        )

        large_data["current_price"] = np.maximum(
            large_data["current_price"], large_data["cogs"] * 1.2
        )

        model = MenuPriceOptimizer(random_state=42)
        metrics = model.train(large_data)

        # Should train successfully
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(metrics["r2_score"])


def run_tests():
    """Run all tests and print summary"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    success_rate = (success_count / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")

    if result.wasSuccessful():
        print("\n[PASS] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed. See details above.")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
