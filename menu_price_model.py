"""
Menu Price Optimizer Model
Uses Random Forest Regression to predict optimal menu item prices
and calculate portfolio metrics (Sharpe ratio, returns, volatility)

Author: Seon Sivasathan
Institution: Computer Science @ Western University
LinkedIn: https://www.linkedin.com/in/seon-sivasathan

"""

import logging
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom Exceptions
class MenuOptimizerError(Exception):
    """Base exception for Menu Optimizer"""
    pass


class InsufficientDataError(MenuOptimizerError):
    """Raised when insufficient data for training"""
    pass


class InvalidDataError(MenuOptimizerError):
    """Raised when data validation fails"""
    pass


class ModelNotTrainedError(MenuOptimizerError):
    """Raised when trying to predict without training"""
    pass


class MenuPriceOptimizer:
    """
    Menu Price Optimization Model using Random Forest Regression
    
    This model applies Modern Portfolio Theory concepts to menu optimization,
    treating menu items as assets with risk-return profiles.
    
    Attributes:
        model: RandomForestRegressor instance
        scaler: StandardScaler for feature normalization
        is_trained: Boolean indicating if model has been trained
        feature_importance_: DataFrame of feature importances
        r2_score_: Model R² score on test set
        mae_: Mean Absolute Error on test set
        rmse_: Root Mean Squared Error on test set
    """
    
    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: int = 10, 
        min_samples_split: int = 5, 
        random_state: int = 42
    ) -> None:
        """
        Initialize the model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
        """
        logger.info("Initializing MenuPriceOptimizer")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.r2_score_: Optional[float] = None
        self.mae_: Optional[float] = None
        self.rmse_: Optional[float] = None
        self._feature_names: Optional[List[str]] = None
        logger.info("Model initialized successfully")
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data before processing
        
        Args:
            df: DataFrame with menu item data
            
        Raises:
            InvalidDataError: If data validation fails
        """
        logger.debug("Validating input data")
        errors = []
        
        # Check for required columns
        required_cols = ['item_name', 'current_price', 'cogs', 'quantity_sold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if errors:
            raise InvalidDataError(f"Data validation failed:\n" + "\n".join(errors))
        
        # Check for negative values
        if (df['current_price'] < 0).any():
            negative_count = (df['current_price'] < 0).sum()
            logger.warning(f"Found {negative_count} negative prices, will be filtered out")
        
        if (df['cogs'] < 0).any():
            negative_count = (df['cogs'] < 0).sum()
            logger.warning(f"Found {negative_count} negative COGS values, will be filtered out")
        
        if (df['quantity_sold'] < 0).any():
            negative_count = (df['quantity_sold'] < 0).sum()
            logger.warning(f"Found {negative_count} negative quantities, will be filtered out")
        
        # Check for invalid relationships (price < COGS = selling at a loss)
        loss_items = (df['current_price'] < df['cogs']).sum()
        if loss_items > 0:
            logger.warning(f"Found {loss_items} items selling at a loss (price < COGS)")
        
        # Check for extreme outliers (only if we have enough data)
        if len(df) > 10:
            try:
                price_quantile_99 = df['current_price'].quantile(0.99)
                if not np.isnan(price_quantile_99) and price_quantile_99 > 0:
                    price_outliers = df['current_price'] > price_quantile_99 * 3
                    if price_outliers.any():
                        outlier_count = price_outliers.sum()
                        logger.warning(f"Detected {outlier_count} extreme price outliers (>3x 99th percentile)")
            except Exception as e:
                logger.debug(f"Could not calculate outliers: {e}")
        
        # Check data types
        numeric_cols = ['current_price', 'cogs', 'quantity_sold']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric, found {df[col].dtype}")
        
        if errors:
            raise InvalidDataError(f"Data validation failed:\n" + "\n".join(errors))
        
        logger.debug("Input data validation passed")
    
    def _prepare_features(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features from menu data with robust validation
        
        Args:
            df: DataFrame with menu item data
            
        Returns:
            Tuple of (X: Feature matrix, y: Target variable, feature_names: List of feature names)
            
        Raises:
            InvalidDataError: If data validation fails
            InsufficientDataError: If insufficient valid data remains
        """
        logger.debug("Preparing features from data")
        
        # Validate input data
        self._validate_input_data(df)
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Calculate revenue and profit
        df['revenue'] = df['current_price'] * df['quantity_sold']
        df['profit'] = df['revenue'] - (df['cogs'] * df['quantity_sold'])
        
        # Calculate profit margin: (price - cogs) / cogs
        # This is independent of quantity and represents the markup percentage
        df['profit_margin'] = np.where(
            df['cogs'] > 0,
            (df['current_price'] - df['cogs']) / df['cogs'],
            np.nan
        )
        
        # Filter out invalid data
        initial_count = len(df)
        df = df.dropna(subset=['profit_margin'])
        df = df[df['profit_margin'] != np.inf]
        df = df[df['profit_margin'] > -1]  # Allow some negative margins but not < -100%
        df = df[df['current_price'] > 0]
        df = df[df['cogs'] > 0]
        df = df[df['quantity_sold'] >= 0]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} invalid rows ({filtered_count/initial_count*100:.1f}%)")
        
        # Check if we have any valid data left
        if len(df) == 0:
            raise InsufficientDataError(
                "No valid data after removing invalid profit margins. "
                "Check that: price > 0, COGS > 0, quantity >= 0"
            )
        
        if len(df) < 10:
            raise InsufficientDataError(
                f"Insufficient data for training. Need at least 10 samples, got {len(df)}. "
                "Please provide more data or check data quality."
            )
        
        # Feature engineering
        features = []
        
        # Basic features
        features.extend(['current_price', 'cogs', 'quantity_sold'])
        
        # Price-to-COGS ratio (markup multiplier)
        df['price_to_cogs'] = df['current_price'] / df['cogs']
        features.append('price_to_cogs')
        
        # Revenue features
        features.append('revenue')
        
        # Profit features
        features.append('profit')
        
        # Total COGS
        df['total_cogs'] = df['cogs'] * df['quantity_sold']
        features.append('total_cogs')
        
        # Profit per unit
        df['profit_per_unit'] = np.where(
            df['quantity_sold'] > 0,
            df['profit'] / df['quantity_sold'],
            0
        )
        features.append('profit_per_unit')
        
        # Categorical features - One-hot encode category
        if 'category' in df.columns:
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            df = pd.concat([df, category_dummies], axis=1)
            features.extend(category_dummies.columns.tolist())
            logger.debug(f"Added {len(category_dummies.columns)} category features")
        
        # Season feature - Map to seasonal factor
        if 'season' in df.columns:
            season_map = config.SEASONAL_FACTORS
            df['season_factor'] = df['season'].map(season_map).fillna(1.0)
            features.append('season_factor')
            logger.debug("Added season factor feature")
        
        # Province feature - Map to tax rate
        if 'province' in df.columns:
            df['tax_rate'] = df['province'].map(config.TAX_RATES).fillna(0.13)
            features.append('tax_rate')
            logger.debug("Added tax rate feature")
        
        # Ensure all features exist
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) == 0:
            raise InvalidDataError("No valid features could be created from the data")
        
        logger.info(f"Created {len(available_features)} features for modeling")
        
        X = df[available_features].values
        y = df['profit_margin'].values
        
        # Convert to float64 for NaN/Inf checking (handles mixed types from one-hot encoding)
        X_float = np.asarray(X, dtype=np.float64)
        y_float = np.asarray(y, dtype=np.float64)
        
        # Final validation - check for NaN or Inf in features
        if np.any(np.isnan(X_float)) or np.any(np.isinf(X_float)):
            logger.error("NaN or Inf values detected in features after preparation")
            raise InvalidDataError("Feature matrix contains NaN or Inf values")
        
        if np.any(np.isnan(y_float)) or np.any(np.isinf(y_float)):
            logger.error("NaN or Inf values detected in target variable")
            raise InvalidDataError("Target variable contains NaN or Inf values")
        
        # Use float64 arrays for training
        X = X_float
        y = y_float
        
        return X, y, available_features
    
    def train(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'profit_margin'
    ) -> Dict[str, float]:
        """
        Train the model on menu data
        
        Args:
            df: DataFrame with menu item data
            target_col: Target column name (default: profit_margin)
            
        Returns:
            dict: Training metrics including r2_score, mae, rmse, cv_r2_mean, cv_r2_std
            
        Raises:
            InsufficientDataError: If insufficient data for training
            InvalidDataError: If data validation fails
        """
        logger.info(f"Starting model training with {len(df)} samples")
        
        try:
            # Prepare features
            X, y, feature_names = self._prepare_features(df)
            self._feature_names = feature_names
            
            # Check for sufficient variance
            if len(np.unique(y)) < 2:
                raise InsufficientDataError(
                    "Target variable has insufficient variance. "
                    "All profit margins are identical, cannot train model."
                )
            
            # Train-test split
            test_size = config.ML_CONFIG.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=config.ML_CONFIG.get('random_state', 42)
            )
            
            logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            logger.info("Training Random Forest model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            self.r2_score_ = r2_score(y_test, y_test_pred)
            self.mae_ = mean_absolute_error(y_test, y_test_pred)
            self.rmse_ = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Feature importance
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.is_trained = True
            
            # Cross-validation score
            logger.info("Running cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='r2'
            )
            
            metrics = {
                'r2_score': self.r2_score_,
                'mae': self.mae_,
                'rmse': self.rmse_,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'feature_importance': self.feature_importance_
            }
            
            logger.info(f"Training complete. R² Score: {self.r2_score_:.4f}, MAE: {self.mae_:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal profit margins for menu items
        
        Args:
            df: DataFrame with menu item data
            
        Returns:
            np.ndarray: Predicted profit margins
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained yet
            InvalidDataError: If data validation fails
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before making predictions. Call train() first."
            )
        
        logger.debug(f"Making predictions for {len(df)} items")
        
        try:
            X, _, _ = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            logger.debug("Predictions completed successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def calculate_portfolio_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate portfolio metrics (Sharpe ratio, returns, volatility)
        
        Applies Modern Portfolio Theory concepts to menu analysis:
        - Mean return: Expected profit margin
        - Volatility: Standard deviation of profit margins
        - Sharpe ratio: Risk-adjusted return metric
        
        Args:
            df: DataFrame with menu item data
            
        Returns:
            dict: Portfolio metrics including mean_return, volatility, sharpe_ratio, recommendations
        """
        logger.debug("Calculating portfolio metrics")
        
        try:
            # Calculate profit margins
            df = df.copy()
            df['revenue'] = df['current_price'] * df['quantity_sold']
            df['profit'] = df['revenue'] - (df['cogs'] * df['quantity_sold'])
            df['profit_margin'] = np.where(
                df['cogs'] > 0,
                (df['current_price'] - df['cogs']) / df['cogs'],
                0
            )
            
            # Get valid returns
            returns = df['profit_margin'].values
            returns = returns[~np.isnan(returns)]
            returns = returns[returns != np.inf]
            returns = returns[returns > -1]  # Filter extreme negative returns
            
            if len(returns) == 0:
                logger.warning("No valid returns found, returning zero metrics")
                return {
                    'mean_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'recommendations': {},
                    'num_items': 0
                }
            
            # Portfolio metrics
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Sharpe ratio (using risk-free rate from config)
            risk_free_rate = config.RISK_FREE_RATE
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            logger.info(f"Portfolio metrics - Return: {mean_return:.4f}, "
                       f"Volatility: {volatility:.4f}, Sharpe: {sharpe_ratio:.4f}")
            
            # Generate recommendations per item
            recommendations = {}
            for idx, row in df.iterrows():
                item_name = row.get('item_name', f'item_{idx}')
                item_return = row['profit_margin']
                
                # Handle invalid returns
                if np.isnan(item_return) or item_return == np.inf or item_return < -1:
                    recommendations[item_name] = 'remove'
                    continue
                
                # Calculate item Sharpe ratio using portfolio volatility
                if volatility > 0:
                    item_sharpe = (item_return - risk_free_rate) / volatility
                else:
                    item_sharpe = item_return - risk_free_rate
                
                # Apply recommendation thresholds from config
                if item_sharpe >= config.RECOMMENDATION_THRESHOLDS['keep_sharpe']:
                    recommendations[item_name] = 'keep'
                elif item_sharpe >= config.RECOMMENDATION_THRESHOLDS['monitor_sharpe']:
                    recommendations[item_name] = 'monitor'
                else:
                    recommendations[item_name] = 'remove'
            
            return {
                'mean_return': mean_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'recommendations': recommendations,
                'num_items': len(df)
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {str(e)}")
            raise
    
    def optimize_prices(
        self, 
        df: pd.DataFrame, 
        target_sharpe: float = 1.5,
        min_margin: float = 0.1
    ) -> pd.DataFrame:
        """
        Optimize menu item prices to achieve target Sharpe ratio
        
        Args:
            df: DataFrame with menu item data
            target_sharpe: Target Sharpe ratio (default: 1.5)
            min_margin: Minimum acceptable profit margin (default: 0.1 = 10%)
            
        Returns:
            DataFrame: Original data with optimized prices
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before optimizing prices. Call train() first."
            )
        
        logger.info(f"Optimizing prices with target Sharpe ratio: {target_sharpe}")
        
        try:
            df = df.copy()
            
            # Predict optimal profit margins
            optimal_margins = self.predict(df)
            
            # Calculate optimal prices
            df['optimal_margin'] = optimal_margins
            df['optimal_price'] = df['cogs'] * (1 + df['optimal_margin'])
            
            # Ensure prices meet minimum margin requirement
            min_price = df['cogs'] * (1 + min_margin)
            df['optimal_price'] = np.maximum(df['optimal_price'], min_price)
            
            # Ensure no negative or zero prices
            df['optimal_price'] = np.where(
                (df['optimal_price'] > 0) & (df['cogs'] > 0),
                df['optimal_price'],
                df['cogs'] * (1 + min_margin * 2)  # Default to 20% margin if invalid
            )
            
            # Calculate price change
            df['price_change'] = df['optimal_price'] - df['current_price']
            # Avoid division by zero
            df['price_change_pct'] = np.where(
                df['current_price'] > 0,
                (df['price_change'] / df['current_price']) * 100,
                0
            )
            
            logger.info(f"Price optimization complete. Average price change: "
                       f"{df['price_change_pct'].mean():.2f}%")
            
            return df
            
        except Exception as e:
            logger.error(f"Price optimization failed: {str(e)}")
            raise