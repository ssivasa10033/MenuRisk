"""
Tests for DataLoader and DataPreprocessor modules.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from src.data.loader import DataLoader, InvalidDataError, DataLoaderError
from src.data.preprocessor import DataPreprocessor, InsufficientDataError


class TestDataLoader:
    """Test suite for DataLoader."""

    def test_load_dataframe(self, sample_data, data_loader):
        """Test loading data from DataFrame."""
        result = data_loader.load_dataframe(sample_data)

        assert result is not None
        assert len(result) == len(sample_data)
        assert list(result.columns) == list(sample_data.columns)

    def test_load_csv(self, sample_data):
        """Test loading data from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            loader = DataLoader()
            result = loader.load_csv(filepath)

            assert result is not None
            assert len(result) == len(sample_data)
        finally:
            os.unlink(filepath)

    def test_load_missing_file(self, data_loader):
        """Test loading non-existent file raises error."""
        with pytest.raises(DataLoaderError):
            data_loader.load_csv('/nonexistent/path/data.csv')

    def test_validate_missing_columns(self, data_loader):
        """Test validation catches missing required columns."""
        incomplete_data = pd.DataFrame({
            'item_name': ['A', 'B'],
            'current_price': [10, 20]
            # Missing: cogs, quantity_sold
        })

        with pytest.raises(InvalidDataError):
            data_loader.load_dataframe(incomplete_data)

    def test_get_summary(self, sample_data, data_loader):
        """Test data summary generation."""
        data_loader.load_dataframe(sample_data)
        summary = data_loader.get_summary()

        assert 'rows' in summary
        assert 'columns' in summary
        assert 'required_present' in summary
        assert summary['rows'] == len(sample_data)

    def test_generate_sample_data(self):
        """Test sample data generation."""
        data = DataLoader.generate_sample_data(n_samples=100)

        assert len(data) == 100
        assert 'item_name' in data.columns
        assert 'current_price' in data.columns
        assert 'cogs' in data.columns
        assert 'quantity_sold' in data.columns
        assert np.all(data['current_price'] > data['cogs'])


class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        preprocessor = DataPreprocessor()
        X, y, feature_names = preprocessor.prepare_features(sample_data)

        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(feature_names) > 0
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_prepare_features_insufficient_data(self):
        """Test feature preparation with insufficient data."""
        preprocessor = DataPreprocessor(min_samples=10)
        small_data = pd.DataFrame({
            'item_name': ['A', 'B', 'C'],
            'current_price': [10, 20, 30],
            'cogs': [5, 10, 15],
            'quantity_sold': [100, 200, 300]
        })

        with pytest.raises(InsufficientDataError):
            preprocessor.prepare_features(small_data)

    def test_transform_after_fit(self, sample_data):
        """Test transform uses fitted scaler."""
        preprocessor = DataPreprocessor()

        # First fit
        X1, y1, names1 = preprocessor.prepare_features(sample_data, fit_scaler=True)

        # Then transform (without fitting)
        X2, names2 = preprocessor.transform(sample_data)

        assert X2 is not None
        assert names1 == names2

    def test_handles_zero_cogs(self, sample_data):
        """Test handling of zero COGS values."""
        preprocessor = DataPreprocessor()
        data_with_zero = sample_data.copy()
        data_with_zero.loc[0, 'cogs'] = 0

        X, y, _ = preprocessor.prepare_features(data_with_zero)

        # Should filter out invalid rows but still work
        assert X is not None
        assert len(X) > 0

    def test_handles_negative_values(self, sample_data):
        """Test handling of negative values."""
        preprocessor = DataPreprocessor()
        data_with_negative = sample_data.copy()
        data_with_negative.loc[0, 'quantity_sold'] = -10

        X, y, _ = preprocessor.prepare_features(data_with_negative)

        # Should filter out invalid rows
        assert len(X) < len(sample_data)

    def test_category_encoding(self, sample_data):
        """Test category one-hot encoding."""
        preprocessor = DataPreprocessor()
        X, y, feature_names = preprocessor.prepare_features(sample_data)

        # Should have category features
        category_features = [f for f in feature_names if f.startswith('category_')]
        assert len(category_features) > 0

    def test_season_factor(self, sample_data):
        """Test season factor feature."""
        preprocessor = DataPreprocessor()
        X, y, feature_names = preprocessor.prepare_features(sample_data)

        assert 'season_factor' in feature_names

    def test_tax_rate_feature(self, sample_data):
        """Test tax rate feature from province."""
        preprocessor = DataPreprocessor()
        X, y, feature_names = preprocessor.prepare_features(sample_data)

        assert 'tax_rate' in feature_names
