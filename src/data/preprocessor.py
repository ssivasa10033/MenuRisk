"""
Data preprocessing utilities for menu optimization.

Handles feature engineering, scaling, and data transformations.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.canadian import get_tax_rate, get_seasonal_factor

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


class InsufficientDataError(PreprocessingError):
    """Raised when there's insufficient data for processing."""
    pass


class DataPreprocessor:
    """
    Preprocesses menu data for ML training.

    Handles feature engineering, scaling, and data validation
    for the menu optimization pipeline.
    """

    def __init__(self, min_samples: int = 10):
        """
        Initialize the preprocessor.

        Args:
            min_samples: Minimum number of samples required (default: 10)
        """
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self._is_fitted = False

    def prepare_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features from menu data.

        Args:
            df: DataFrame with menu item data
            fit_scaler: Whether to fit the scaler (True for training)

        Returns:
            Tuple of (X: features, y: target, feature_names)

        Raises:
            PreprocessingError: If data validation fails
            InsufficientDataError: If insufficient valid data remains
        """
        logger.debug("Preparing features from data")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Calculate derived metrics
        df = self._calculate_metrics(df)

        # Filter invalid data
        df = self._filter_invalid_data(df)

        # Check minimum samples
        if len(df) < self.min_samples:
            raise InsufficientDataError(
                f"Insufficient data. Need at least {self.min_samples} samples, "
                f"got {len(df)}. Check data quality."
            )

        # Build feature matrix
        features, feature_names = self._build_features(df)

        # Extract target variable
        y = df['profit_margin'].values.astype(np.float64)

        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(features)
            self._is_fitted = True
        else:
            if not self._is_fitted:
                raise PreprocessingError(
                    "Scaler not fitted. Call prepare_features with fit_scaler=True first."
                )
            X = self.scaler.transform(features)

        self.feature_names = feature_names

        # Final validation
        self._validate_arrays(X, y)

        return X, y, feature_names

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics from raw data."""
        # Revenue and profit
        df['revenue'] = df['current_price'] * df['quantity_sold']
        df['profit'] = df['revenue'] - (df['cogs'] * df['quantity_sold'])

        # Profit margin: (price - cogs) / cogs
        df['profit_margin'] = np.where(
            df['cogs'] > 0,
            (df['current_price'] - df['cogs']) / df['cogs'],
            np.nan
        )

        # Price-to-COGS ratio
        df['price_to_cogs'] = np.where(
            df['cogs'] > 0,
            df['current_price'] / df['cogs'],
            np.nan
        )

        # Total COGS
        df['total_cogs'] = df['cogs'] * df['quantity_sold']

        # Profit per unit
        df['profit_per_unit'] = np.where(
            df['quantity_sold'] > 0,
            df['profit'] / df['quantity_sold'],
            0
        )

        return df

    def _filter_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid rows."""
        initial_count = len(df)

        # Remove invalid profit margins
        df = df.dropna(subset=['profit_margin'])
        df = df[df['profit_margin'] != np.inf]
        df = df[df['profit_margin'] > -1]  # Allow some negative but not < -100%

        # Remove invalid prices/costs
        df = df[df['current_price'] > 0]
        df = df[df['cogs'] > 0]
        df = df[df['quantity_sold'] >= 0]

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            pct = filtered_count / initial_count * 100
            logger.info(f"Filtered {filtered_count} invalid rows ({pct:.1f}%)")

        return df

    def _build_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Build feature matrix from preprocessed data."""
        features = []

        # Basic features
        basic_features = [
            'current_price', 'cogs', 'quantity_sold',
            'price_to_cogs', 'revenue', 'profit',
            'total_cogs', 'profit_per_unit'
        ]
        features.extend(basic_features)

        # Category one-hot encoding
        if 'category' in df.columns:
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            df = pd.concat([df, category_dummies], axis=1)
            features.extend(category_dummies.columns.tolist())
            logger.debug(f"Added {len(category_dummies.columns)} category features")

        # Season factor
        if 'season' in df.columns:
            df['season_factor'] = df['season'].apply(get_seasonal_factor)
            features.append('season_factor')
            logger.debug("Added season factor feature")

        # Province tax rate
        if 'province' in df.columns:
            df['tax_rate'] = df['province'].apply(get_tax_rate)
            features.append('tax_rate')
            logger.debug("Added tax rate feature")

        # Filter to available features
        available_features = [f for f in features if f in df.columns]

        if not available_features:
            raise PreprocessingError("No valid features could be created")

        logger.info(f"Created {len(available_features)} features")

        X = df[available_features].values.astype(np.float64)

        return X, available_features

    def _validate_arrays(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate feature and target arrays."""
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise PreprocessingError("Feature matrix contains NaN or Inf values")

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise PreprocessingError("Target variable contains NaN or Inf values")

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Transform new data using fitted scaler.

        Args:
            df: DataFrame with menu item data

        Returns:
            Tuple of (X: scaled features, feature_names)
        """
        X, _, feature_names = self.prepare_features(df, fit_scaler=False)
        return X, feature_names
