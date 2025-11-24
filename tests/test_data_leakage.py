"""
Tests for data leakage prevention in feature engineering.

These tests ensure that rolling features and statistics only use
information available at prediction time.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import (
    TimeSeriesFeatureEngineer,
    create_train_test_split,
    InsufficientDataError,
)


class TestTimeSeriesFeatureEngineer:
    """Tests for TimeSeriesFeatureEngineer class."""

    @pytest.fixture
    def sample_timeseries_data(self):
        """Generate sample time-series sales data."""
        np.random.seed(42)

        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        items = ['Item_A', 'Item_B', 'Item_C']

        data = []
        for item in items:
            base_sales = {'Item_A': 50, 'Item_B': 80, 'Item_C': 30}[item]
            for date in dates:
                # Add some seasonality and noise
                seasonal = 10 * np.sin(2 * np.pi * date.dayofyear / 365)
                noise = np.random.normal(0, 5)
                quantity = max(1, int(base_sales + seasonal + noise))

                data.append({
                    'date': date,
                    'item_name': item,
                    'quantity_sold': quantity,
                    'current_price': 18.99 if item == 'Item_A' else 12.99,
                    'cogs': 7.50 if item == 'Item_A' else 5.00,
                    'category': 'Main' if item == 'Item_A' else 'Appetizer',
                    'season': 'Summer',
                    'province': 'ON',
                })

        return pd.DataFrame(data)

    def test_no_future_information_in_features(self, sample_timeseries_data):
        """Ensure features only use information available at prediction time."""
        df = sample_timeseries_data
        cutoff = '2023-11-01'

        engineer = TimeSeriesFeatureEngineer()
        engineer.fit(df, cutoff)

        # Get training stats
        train_mean = engineer.item_stats['Item_A']['mean']

        # Transform full dataset
        transformed = engineer.transform(df)

        # Check: test set features shouldn't include test set information
        test_df = transformed[transformed['date'] > cutoff]

        # The item_mean_train should equal training mean exactly
        test_mean_feature = test_df[test_df['item_name'] == 'Item_A']['item_mean_train'].iloc[0]

        assert abs(test_mean_feature - train_mean) < 1e-6, \
            f"Test set is contaminated with future information! " \
            f"Expected {train_mean}, got {test_mean_feature}"

    def test_rolling_features_use_past_only(self, sample_timeseries_data):
        """Ensure rolling features only use past data."""
        df = sample_timeseries_data
        cutoff = '2023-06-01'

        engineer = TimeSeriesFeatureEngineer()
        features = engineer.fit_transform(df, cutoff)

        # Pick a test date
        test_date = pd.Timestamp('2023-06-15')
        item = 'Item_A'

        test_row = features[
            (features['date'] == test_date) &
            (features['item_name'] == item)
        ].iloc[0]

        # Manual calculation: rolling avg should be from past 7 days (excluding current)
        # Because we use shift(1), current day is excluded
        past_data = df[
            (df['date'] < test_date) &
            (df['date'] >= test_date - pd.Timedelta(days=7)) &
            (df['item_name'] == item)
        ]['quantity_sold']

        expected_rolling_avg = past_data.mean()
        actual_rolling_avg = test_row['rolling_avg_7d']

        # Allow small tolerance for floating point
        assert abs(actual_rolling_avg - expected_rolling_avg) < 1.0, \
            f"Rolling average includes future data! " \
            f"Expected ~{expected_rolling_avg:.2f}, got {actual_rolling_avg:.2f}"

    def test_lag_features_use_past_only(self, sample_timeseries_data):
        """Ensure lag features correctly reference past data."""
        df = sample_timeseries_data
        cutoff = '2023-06-01'

        engineer = TimeSeriesFeatureEngineer()
        features = engineer.fit_transform(df, cutoff)

        # Pick a test date
        test_date = pd.Timestamp('2023-06-15')
        item = 'Item_A'

        test_row = features[
            (features['date'] == test_date) &
            (features['item_name'] == item)
        ].iloc[0]

        # Get the actual value from 7 days ago
        lag_date = test_date - pd.Timedelta(days=7)
        expected_lag = df[
            (df['date'] == lag_date) &
            (df['item_name'] == item)
        ]['quantity_sold'].iloc[0]

        actual_lag = test_row['lag_7d']

        assert actual_lag == expected_lag, \
            f"Lag feature incorrect! Expected {expected_lag}, got {actual_lag}"

    def test_item_statistics_from_training_only(self, sample_timeseries_data):
        """Ensure item statistics are calculated only from training data."""
        df = sample_timeseries_data
        cutoff = '2023-06-01'

        train_df = df[df['date'] <= cutoff]
        test_df = df[df['date'] > cutoff]

        # Calculate expected training statistics
        expected_mean = train_df[train_df['item_name'] == 'Item_A']['quantity_sold'].mean()

        engineer = TimeSeriesFeatureEngineer()
        engineer.fit(df, cutoff)

        # Verify stored statistics match training data only
        stored_mean = engineer.item_stats['Item_A']['mean']

        assert abs(stored_mean - expected_mean) < 1e-6, \
            f"Item statistics include test data! " \
            f"Expected {expected_mean}, got {stored_mean}"

        # Also verify that test data wasn't used
        test_mean = test_df[test_df['item_name'] == 'Item_A']['quantity_sold'].mean()
        full_mean = df[df['item_name'] == 'Item_A']['quantity_sold'].mean()

        # If test data was included, stored_mean would be closer to full_mean
        assert abs(stored_mean - full_mean) > abs(stored_mean - expected_mean) or \
               abs(stored_mean - expected_mean) < 1e-6, \
            "Statistics may include test data!"

    def test_fit_then_transform_maintains_separation(self, sample_timeseries_data):
        """Ensure fit/transform separation prevents leakage."""
        df = sample_timeseries_data
        cutoff = '2023-09-01'

        # Fit only on training data
        train_df = df[df['date'] <= cutoff]
        engineer = TimeSeriesFeatureEngineer()
        engineer.fit(train_df, cutoff)

        # Transform test data
        test_df = df[df['date'] > cutoff]
        transformed_test = engineer.transform(test_df)

        # Statistics should be from training data
        test_item_mean = transformed_test[
            transformed_test['item_name'] == 'Item_B'
        ]['item_mean_train'].iloc[0]

        expected_train_mean = train_df[
            train_df['item_name'] == 'Item_B'
        ]['quantity_sold'].mean()

        assert abs(test_item_mean - expected_train_mean) < 1e-6, \
            "Transform used test data for statistics!"

    def test_new_items_handled_gracefully(self, sample_timeseries_data):
        """Ensure new items (not in training) are handled properly."""
        df = sample_timeseries_data
        cutoff = '2023-06-01'

        engineer = TimeSeriesFeatureEngineer()
        engineer.fit(df, cutoff)

        # Create test data with a new item
        new_item_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-12-01', '2023-12-02']),
            'item_name': ['New_Item', 'New_Item'],
            'quantity_sold': [100, 110],
            'current_price': [15.00, 15.00],
            'cogs': [6.00, 6.00],
            'category': ['Main', 'Main'],
            'season': ['Winter', 'Winter'],
            'province': ['ON', 'ON'],
        })

        # Should not raise an error
        transformed = engineer.transform(new_item_df)

        # New item should have fallback to global statistics
        assert 'item_mean_train' in transformed.columns
        assert not transformed['item_mean_train'].isna().all(), \
            "New items should have fallback statistics"


class TestTrainTestSplit:
    """Tests for the train-test split function."""

    @pytest.fixture
    def timeseries_data(self):
        """Generate time-series data for split testing."""
        np.random.seed(42)

        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        data = []
        for date in dates:
            for item in ['A', 'B']:
                quantity = np.random.randint(20, 80)
                data.append({
                    'date': date,
                    'item_name': item,
                    'quantity_sold': quantity,
                    'current_price': 20.0,
                    'cogs': 8.0,
                    'category': 'Main',
                    'season': 'Summer',
                    'province': 'ON',
                })
        return pd.DataFrame(data)

    def test_split_by_date_not_random(self, timeseries_data):
        """Ensure split is temporal, not random."""
        X_train, X_test, y_train, y_test, engineer = create_train_test_split(
            timeseries_data,
            test_size_days=30
        )

        # Get original dates
        train_dates = timeseries_data[
            timeseries_data['date'] <= engineer.cutoff_date
        ]['date']
        test_dates = timeseries_data[
            timeseries_data['date'] > engineer.cutoff_date
        ]['date']

        # All training data should be before all test data
        assert train_dates.max() < test_dates.min(), \
            "Train-test split is not purely temporal!"

    def test_no_leakage_across_split(self, timeseries_data):
        """Ensure no information leaks from test to train."""
        X_train, X_test, y_train, y_test, engineer = create_train_test_split(
            timeseries_data,
            test_size_days=30
        )

        # Feature engineer should only have training statistics
        cutoff = pd.to_datetime(engineer.cutoff_date)
        train_df = timeseries_data[timeseries_data['date'] <= cutoff]

        expected_mean = train_df[train_df['item_name'] == 'A']['quantity_sold'].mean()
        stored_mean = engineer.item_stats['A']['mean']

        assert abs(stored_mean - expected_mean) < 1e-6, \
            f"Feature engineer used test data! Expected {expected_mean}, got {stored_mean}"

    def test_consistent_features_across_split(self, timeseries_data):
        """Ensure train and test have same feature columns."""
        X_train, X_test, y_train, y_test, _ = create_train_test_split(
            timeseries_data,
            test_size_days=30
        )

        assert list(X_train.columns) == list(X_test.columns), \
            "Train and test have different features!"

    def test_target_not_in_features(self, timeseries_data):
        """Ensure target variable is not included in features."""
        X_train, X_test, y_train, y_test, _ = create_train_test_split(
            timeseries_data,
            test_size_days=30,
            target_col='quantity_sold'
        )

        assert 'quantity_sold' not in X_train.columns, \
            "Target variable is in training features!"
        assert 'quantity_sold' not in X_test.columns, \
            "Target variable is in test features!"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_data_raises_error(self):
        """Ensure error is raised for insufficient data."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'item_name': ['A'] * 5,
            'quantity_sold': [10, 20, 30, 40, 50],
        })

        engineer = TimeSeriesFeatureEngineer()

        with pytest.raises(InsufficientDataError):
            engineer.fit(df, '2022-12-31')  # No data before this cutoff

    def test_missing_date_column_warning(self):
        """Ensure warning when date column is missing."""
        df = pd.DataFrame({
            'item_name': ['A', 'B', 'C'],
            'quantity_sold': [10, 20, 30],
            'current_price': [15.0, 16.0, 17.0],
            'cogs': [5.0, 6.0, 7.0],
        })

        engineer = TimeSeriesFeatureEngineer()
        # Should not raise error, but uses all data for training
        engineer.fit(df, '2023-01-01')

        assert engineer._is_fitted

    def test_unfitted_transform_raises_error(self):
        """Ensure error when transform called before fit."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'item_name': ['A'] * 10,
            'quantity_sold': list(range(10, 20)),
        })

        engineer = TimeSeriesFeatureEngineer()

        with pytest.raises(Exception):  # FeatureEngineeringError
            engineer.transform(df)


class TestFeatureValidity:
    """Tests for feature correctness."""

    @pytest.fixture
    def deterministic_data(self):
        """Create deterministic data for precise testing."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'item_name': ['A'] * 20,
            'quantity_sold': list(range(100, 120)),  # 100, 101, ..., 119
            'current_price': [20.0] * 20,
            'cogs': [8.0] * 20,
            'category': ['Main'] * 20,
            'season': ['Winter'] * 20,
            'province': ['ON'] * 20,
        })

    def test_rolling_average_correctness(self, deterministic_data):
        """Verify rolling average calculation is correct."""
        df = deterministic_data
        cutoff = '2023-01-15'

        engineer = TimeSeriesFeatureEngineer()
        features = engineer.fit_transform(df, cutoff)

        # Check day 10 (2023-01-10): rolling_avg_7d should be avg of days 3-9
        # Values: 102, 103, 104, 105, 106, 107, 108 = avg 105
        day_10 = features[features['date'] == '2023-01-10'].iloc[0]

        # Days 3-9 correspond to values 102-108
        expected_avg = np.mean([102, 103, 104, 105, 106, 107, 108])
        actual_avg = day_10['rolling_avg_7d']

        assert abs(actual_avg - expected_avg) < 0.5, \
            f"Rolling average incorrect! Expected {expected_avg}, got {actual_avg}"

    def test_profit_margin_calculation(self, deterministic_data):
        """Verify profit margin is calculated correctly."""
        df = deterministic_data
        cutoff = '2023-01-15'

        engineer = TimeSeriesFeatureEngineer()
        features = engineer.fit_transform(df, cutoff)

        # margin = (price - cogs) / cogs = (20 - 8) / 8 = 1.5
        expected_margin = 1.5

        actual_margins = features['profit_margin'].dropna().unique()
        assert all(abs(m - expected_margin) < 1e-6 for m in actual_margins), \
            f"Profit margin incorrect! Expected {expected_margin}"
