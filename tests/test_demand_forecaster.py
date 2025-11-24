"""
Tests for the DemandForecaster model.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import numpy as np
import pandas as pd
import pytest

from src.models.demand_forecaster import DemandForecaster


class TestDemandForecaster:
    """Tests for DemandForecaster class."""

    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 200

        X = np.random.randn(n_samples, 5)
        # Create target with some correlation to features
        y = 50 + 10 * X[:, 0] + 5 * X[:, 1] + np.random.randn(n_samples) * 5

        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

        return X, y, feature_names

    @pytest.fixture
    def trained_model(self, sample_training_data):
        """Get a trained model."""
        X, y, feature_names = sample_training_data

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X, y, feature_names=feature_names)

        return model

    def test_initialization(self):
        """Test model initialization."""
        model = DemandForecaster()

        assert model.model is None
        assert not model.is_trained
        assert model.tune_hyperparams

    def test_initialization_no_tuning(self):
        """Test initialization without hyperparameter tuning."""
        model = DemandForecaster(tune_hyperparams=False)

        assert not model.tune_hyperparams

    def test_train_basic(self, sample_training_data):
        """Test basic training."""
        X, y, feature_names = sample_training_data

        model = DemandForecaster(tune_hyperparams=False)
        metrics = model.train(X, y, feature_names=feature_names)

        assert model.is_trained
        assert model.model is not None
        assert 'r2_score' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics

    def test_train_with_dataframe(self, sample_training_data):
        """Test training with DataFrame input."""
        X, y, feature_names = sample_training_data

        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        model = DemandForecaster(tune_hyperparams=False)
        metrics = model.train(X_df, y_series)

        assert model.is_trained
        assert model.feature_names == feature_names

    def test_predict_before_training(self, sample_training_data):
        """Test that predicting before training raises error."""
        X, _, _ = sample_training_data

        model = DemandForecaster()

        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predict_after_training(self, trained_model, sample_training_data):
        """Test prediction after training."""
        X, _, _ = sample_training_data

        predictions = trained_model.predict(X[:10])

        assert len(predictions) == 10
        assert all(np.isfinite(predictions))

    def test_predict_with_std(self, trained_model, sample_training_data):
        """Test prediction with standard deviation."""
        X, _, _ = sample_training_data

        predictions, std = trained_model.predict(X[:10], return_std=True)

        assert len(predictions) == 10
        assert len(std) == 10
        assert all(std >= 0)  # Std should be non-negative

    def test_prediction_intervals(self, trained_model, sample_training_data):
        """Test prediction interval calculation."""
        X, _, _ = sample_training_data

        preds, lower, upper = trained_model.get_prediction_intervals(X[:10], confidence=0.90)

        # Lower should be less than or equal to prediction
        assert all(lower <= preds)
        # Upper should be greater than or equal to prediction
        assert all(upper >= preds)
        # 90% interval should be narrower than 80%
        _, lower_80, upper_80 = trained_model.get_prediction_intervals(X[:10], confidence=0.80)
        assert np.mean(upper - lower) > np.mean(upper_80 - lower_80)

    def test_feature_importance(self, trained_model):
        """Test feature importance extraction."""
        importance_df = trained_model.get_feature_importance()

        assert importance_df is not None
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        # Importances should sum to approximately 1
        assert abs(importance_df['importance'].sum() - 1.0) < 0.01

    def test_get_metrics(self, trained_model):
        """Test getting model metrics."""
        metrics = trained_model.get_metrics()

        assert 'r2_score' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'is_trained' in metrics
        assert metrics['is_trained']

    def test_get_metrics_untrained(self):
        """Test metrics for untrained model."""
        model = DemandForecaster()
        metrics = model.get_metrics()

        assert 'error' in metrics

    def test_evaluate(self, trained_model, sample_training_data):
        """Test model evaluation on test data."""
        X, y, _ = sample_training_data

        # Use last 20 samples as test
        X_test = X[-20:]
        y_test = y[-20:]

        eval_metrics = trained_model.evaluate(X_test, y_test)

        assert 'test_r2' in eval_metrics
        assert 'test_mae' in eval_metrics
        assert 'test_rmse' in eval_metrics

    def test_model_reproducibility(self, sample_training_data):
        """Test that model training is reproducible with same seed."""
        X, y, feature_names = sample_training_data

        model1 = DemandForecaster(tune_hyperparams=False, random_state=42)
        model1.train(X, y, feature_names=feature_names)
        preds1 = model1.predict(X[:10])

        model2 = DemandForecaster(tune_hyperparams=False, random_state=42)
        model2.train(X, y, feature_names=feature_names)
        preds2 = model2.predict(X[:10])

        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_default_params(self):
        """Test default parameter values."""
        model = DemandForecaster(tune_hyperparams=False)
        params = model.get_default_params()

        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert params['n_estimators'] >= 100

    def test_param_distributions(self):
        """Test hyperparameter search space."""
        model = DemandForecaster()
        params = model.get_param_distributions()

        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'min_samples_split' in params
        assert len(params['n_estimators']) > 1


class TestDemandForecasterWithTuning:
    """Tests for hyperparameter tuning."""

    @pytest.fixture
    def larger_training_data(self):
        """Generate larger training data for tuning tests."""
        np.random.seed(42)
        n_samples = 500

        X = np.random.randn(n_samples, 5)
        y = 50 + 10 * X[:, 0] + 5 * X[:, 1] + np.random.randn(n_samples) * 5

        return X, y

    @pytest.mark.slow
    def test_train_with_tuning(self, larger_training_data):
        """Test training with hyperparameter tuning (slower)."""
        X, y = larger_training_data

        model = DemandForecaster(tune_hyperparams=True)
        metrics = model.train(X, y, cv_splits=3, n_iter=5)  # Reduced for speed

        assert model.is_trained
        assert model.best_params is not None
        assert 'n_estimators' in model.best_params


class TestModelPersistence:
    """Tests for model save/load functionality."""

    @pytest.fixture
    def trained_model_with_data(self, tmp_path):
        """Get trained model and test data."""
        np.random.seed(42)
        n_samples = 100

        X = np.random.randn(n_samples, 5)
        y = 50 + 10 * X[:, 0] + np.random.randn(n_samples) * 5

        model = DemandForecaster(tune_hyperparams=False)
        model.train(X, y, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])

        return model, X, tmp_path

    def test_save_and_load(self, trained_model_with_data):
        """Test model save and load."""
        model, X, tmp_path = trained_model_with_data
        model_path = tmp_path / "model.joblib"

        # Get predictions before save
        original_preds = model.predict(X[:10])

        # Save
        model.save(str(model_path))
        assert model_path.exists()

        # Load
        loaded_model = DemandForecaster.load(str(model_path))

        assert loaded_model.is_trained
        assert loaded_model.feature_names == model.feature_names

        # Predictions should match
        loaded_preds = loaded_model.predict(X[:10])
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
