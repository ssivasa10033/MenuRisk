"""
Time-series feature engineering with data leakage prevention.

Ensures that rolling features only use information available at prediction time.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.canadian import get_tax_rate, get_seasonal_factor

logger = logging.getLogger(__name__)


class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""
    pass


class InsufficientDataError(FeatureEngineeringError):
    """Raised when there's insufficient data for processing."""
    pass


class TimeSeriesFeatureEngineer:
    """
    Feature engineer that prevents data leakage by only using
    information available at prediction time.

    This class implements proper time-series feature engineering:
    - Lag features are calculated using only past data
    - Rolling statistics use only historical information
    - Training statistics are captured and reused for test data
    """

    def __init__(self, lookback_periods: Optional[Dict[str, int]] = None):
        """
        Initialize the time-series feature engineer.

        Args:
            lookback_periods: Dict of feature names to lookback windows
                {'lag_7d': 7, 'rolling_avg_7d': 7, 'rolling_avg_30d': 30}
        """
        self.lookback_periods = lookback_periods or {
            'lag_7d': 7,
            'lag_14d': 14,
            'rolling_avg_7d': 7,
            'rolling_avg_30d': 30,
            'rolling_std_7d': 7,
        }
        self.item_stats: Dict[str, Dict[str, float]] = {}
        self.global_stats: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.cutoff_date: Optional[str] = None

    def fit(self, df: pd.DataFrame, cutoff_date: str) -> 'TimeSeriesFeatureEngineer':
        """
        Fit on training data only (before cutoff_date).

        IMPORTANT: This method captures statistics only from training data
        to prevent data leakage from the test set.

        Args:
            df: DataFrame with columns [date, item_name, quantity_sold, current_price, cogs]
            cutoff_date: Last date to include in training (YYYY-MM-DD)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting feature engineer with cutoff date: {cutoff_date}")

        # Ensure date column is datetime
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            cutoff = pd.to_datetime(cutoff_date)
            train_df = df[df['date'] <= cutoff].copy()
        else:
            # If no date column, use all data for training
            train_df = df.copy()
            logger.warning("No 'date' column found. Using all data for training statistics.")

        self.cutoff_date = cutoff_date

        if len(train_df) == 0:
            raise InsufficientDataError(
                f"No training data before cutoff date {cutoff_date}"
            )

        # Calculate item-level statistics ONLY on training data
        if 'item_name' in train_df.columns and 'quantity_sold' in train_df.columns:
            self.item_stats = (
                train_df.groupby('item_name')['quantity_sold']
                .agg(['mean', 'std', 'median', 'min', 'max'])
                .to_dict('index')
            )
            logger.info(f"Calculated statistics for {len(self.item_stats)} items")

        # Calculate global statistics
        if 'quantity_sold' in train_df.columns:
            self.global_stats = {
                'quantity_mean': train_df['quantity_sold'].mean(),
                'quantity_std': train_df['quantity_sold'].std(),
                'quantity_median': train_df['quantity_sold'].median(),
            }

        # Calculate profit margin statistics
        if 'current_price' in train_df.columns and 'cogs' in train_df.columns:
            train_df['profit_margin'] = np.where(
                train_df['cogs'] > 0,
                (train_df['current_price'] - train_df['cogs']) / train_df['cogs'],
                np.nan
            )
            valid_margins = train_df['profit_margin'].dropna()
            self.global_stats['margin_mean'] = valid_margins.mean()
            self.global_stats['margin_std'] = valid_margins.std()

        self._is_fitted = True
        logger.info("Feature engineer fitted successfully")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using only past information.

        This method creates features that only use information available
        at each prediction time point.

        Args:
            df: DataFrame sorted by date

        Returns:
            DataFrame with engineered features
        """
        if not self._is_fitted:
            raise FeatureEngineeringError(
                "Feature engineer not fitted. Call fit() first."
            )

        logger.debug(f"Transforming {len(df)} rows")
        df = df.copy()

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['item_name', 'date'] if 'item_name' in df.columns else ['date'])

        # Calculate basic derived metrics
        df = self._calculate_basic_metrics(df)

        # Add lag features (using only past data)
        if 'date' in df.columns and 'item_name' in df.columns:
            df = self._add_lag_features(df)
            df = self._add_rolling_features(df)

        # Add item-level statistics from training data
        df = self._add_item_statistics(df)

        # Add categorical features
        df = self._add_categorical_features(df)

        logger.debug(f"Transformation complete. {len(df.columns)} columns")

        return df

    def _calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic derived metrics."""
        # Revenue and profit
        if 'current_price' in df.columns and 'quantity_sold' in df.columns:
            df['revenue'] = df['current_price'] * df['quantity_sold']

        if all(col in df.columns for col in ['revenue', 'cogs', 'quantity_sold']):
            df['profit'] = df['revenue'] - (df['cogs'] * df['quantity_sold'])
            df['total_cogs'] = df['cogs'] * df['quantity_sold']

        # Profit margin
        if 'current_price' in df.columns and 'cogs' in df.columns:
            df['profit_margin'] = np.where(
                df['cogs'] > 0,
                (df['current_price'] - df['cogs']) / df['cogs'],
                np.nan
            )
            df['price_to_cogs'] = np.where(
                df['cogs'] > 0,
                df['current_price'] / df['cogs'],
                np.nan
            )

        # Profit per unit
        if 'profit' in df.columns and 'quantity_sold' in df.columns:
            df['profit_per_unit'] = np.where(
                df['quantity_sold'] > 0,
                df['profit'] / df['quantity_sold'],
                0
            )

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features using only past data.

        CRITICAL: shift() ensures we only use past values, preventing leakage.
        """
        if 'quantity_sold' not in df.columns or 'item_name' not in df.columns:
            return df

        for lag_name, lag_days in self.lookback_periods.items():
            if 'lag' in lag_name:
                df[lag_name] = (
                    df.groupby('item_name')['quantity_sold']
                    .shift(lag_days)
                )
                logger.debug(f"Added {lag_name} feature with {lag_days} day lag")

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling features using only past data.

        CRITICAL: rolling() with shift(1) ensures we don't include current value.
        """
        if 'quantity_sold' not in df.columns or 'item_name' not in df.columns:
            return df

        for roll_name, window in self.lookback_periods.items():
            if 'rolling_avg' in roll_name:
                # Shift by 1 to exclude current observation
                df[roll_name] = (
                    df.groupby('item_name')['quantity_sold']
                    .transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                    )
                )
            elif 'rolling_std' in roll_name:
                df[roll_name] = (
                    df.groupby('item_name')['quantity_sold']
                    .transform(
                        lambda x: x.shift(1).rolling(window, min_periods=2).std()
                    )
                )

        return df

    def _add_item_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add item-level statistics calculated from training data only."""
        if 'item_name' not in df.columns or not self.item_stats:
            return df

        # Map training statistics to items
        df['item_mean_train'] = df['item_name'].map(
            {k: v.get('mean', self.global_stats.get('quantity_mean', 0))
             for k, v in self.item_stats.items()}
        )
        df['item_std_train'] = df['item_name'].map(
            {k: v.get('std', self.global_stats.get('quantity_std', 0))
             for k, v in self.item_stats.items()}
        )
        df['item_median_train'] = df['item_name'].map(
            {k: v.get('median', self.global_stats.get('quantity_median', 0))
             for k, v in self.item_stats.items()}
        )

        # Fill NaN for new items not in training data
        for col in ['item_mean_train', 'item_std_train', 'item_median_train']:
            df[col] = df[col].fillna(self.global_stats.get('quantity_mean', 0))

        return df

    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical features (one-hot encoding, seasonal factors, etc.)."""
        # Category one-hot encoding
        if 'category' in df.columns:
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            df = pd.concat([df, category_dummies], axis=1)
            logger.debug(f"Added {len(category_dummies.columns)} category features")

        # Season factor
        if 'season' in df.columns:
            df['season_factor'] = df['season'].apply(get_seasonal_factor)
            logger.debug("Added season factor feature")

        # Province tax rate
        if 'province' in df.columns:
            df['tax_rate'] = df['province'].apply(get_tax_rate)
            logger.debug("Added tax rate feature")

        return df

    def fit_transform(self, df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, cutoff_date).transform(df)

    def get_feature_columns(
        self,
        exclude_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of feature column names.

        Args:
            exclude_cols: Columns to exclude (default: date, item_name, target)

        Returns:
            List of feature column names
        """
        default_exclude = ['date', 'item_name', 'quantity_sold', 'profit_margin']
        exclude = exclude_cols or default_exclude

        if self.feature_names:
            return [f for f in self.feature_names if f not in exclude]
        return []


def create_train_test_split(
    df: pd.DataFrame,
    test_size_days: int = 30,
    target_col: str = 'quantity_sold'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, TimeSeriesFeatureEngineer]:
    """
    Create time-series train-test split with no data leakage.

    This function implements proper time-series splitting:
    1. Splits by date (not random)
    2. Fits feature engineer only on training data
    3. Transforms both sets using training statistics

    Args:
        df: Full dataset with 'date' column
        test_size_days: Number of days for test set
        target_col: Name of target column

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_engineer)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Split by date
    cutoff_date = df['date'].max() - pd.Timedelta(days=test_size_days)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    train_mask = df['date'] <= cutoff_date
    test_mask = df['date'] > cutoff_date

    logger.info(f"Train period: up to {cutoff_str}")
    logger.info(f"Test period: after {cutoff_str}")
    logger.info(f"Train size: {train_mask.sum()}, Test size: {test_mask.sum()}")

    # Fit feature engineer on training data ONLY
    feature_engineer = TimeSeriesFeatureEngineer()
    feature_engineer.fit(df[train_mask], cutoff_str)

    # Transform both sets (features only use past info)
    train_df = feature_engineer.transform(df[train_mask])

    # For test set, we need to transform the full data to get rolling features
    # but only keep the test portion
    full_transformed = feature_engineer.transform(df)
    test_df = full_transformed[test_mask]

    # Drop rows with NaN in lag features (first few days)
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Get feature columns
    exclude_cols = ['date', 'item_name', target_col, 'profit_margin']
    feature_cols = [col for col in train_df.columns
                   if col not in exclude_cols and not col.startswith('_')]

    # Filter to numeric columns only
    numeric_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_train = train_df[numeric_cols]
    X_test = test_df[numeric_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    logger.info(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_engineer
