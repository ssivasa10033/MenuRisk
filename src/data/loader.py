"""
Data loading utilities for menu data.

Handles CSV loading, validation, and basic data cleaning.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Base exception for data loading errors."""

    pass


class InvalidDataError(DataLoaderError):
    """Raised when data validation fails."""

    pass


class DataLoader:
    """
    Handles loading and initial validation of menu data.

    Supports CSV files with menu item information including
    prices, costs, quantities, and optional categorical data.
    """

    REQUIRED_COLUMNS = ["item_name", "current_price", "cogs", "quantity_sold"]
    OPTIONAL_COLUMNS = ["category", "season", "province", "date", "event_type"]

    def __init__(self):
        """Initialize the DataLoader."""
        self.data: Optional[pd.DataFrame] = None
        self._validation_errors: List[str] = []

    def load_csv(self, filepath: str, validate: bool = True) -> pd.DataFrame:
        """
        Load menu data from a CSV file.

        Args:
            filepath: Path to the CSV file
            validate: Whether to validate required columns (default: True)

        Returns:
            DataFrame with loaded data

        Raises:
            DataLoaderError: If file cannot be loaded
            InvalidDataError: If validation fails
        """
        logger.info(f"Loading data from {filepath}")

        try:
            path = Path(filepath)
            if not path.exists():
                raise DataLoaderError(f"File not found: {filepath}")

            self.data = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.data)} rows")

            if validate:
                self._validate_columns()

            return self.data

        except pd.errors.EmptyDataError:
            raise DataLoaderError(f"File is empty: {filepath}")
        except pd.errors.ParserError as e:
            raise DataLoaderError(f"CSV parsing error: {e}")

    def load_dataframe(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Load data from an existing DataFrame.

        Args:
            df: Input DataFrame
            validate: Whether to validate required columns (default: True)

        Returns:
            Validated DataFrame

        Raises:
            InvalidDataError: If validation fails
        """
        self.data = df.copy()

        if validate:
            self._validate_columns()

        return self.data

    def _validate_columns(self) -> None:
        """
        Validate that required columns exist.

        Raises:
            InvalidDataError: If required columns are missing
        """
        if self.data is None:
            raise InvalidDataError("No data loaded")

        missing_cols = [
            col for col in self.REQUIRED_COLUMNS if col not in self.data.columns
        ]

        if missing_cols:
            raise InvalidDataError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )

        # Check for numeric columns
        numeric_cols = ["current_price", "cogs", "quantity_sold"]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                    logger.warning(f"Converted {col} to numeric type")
                except Exception:
                    raise InvalidDataError(f"Column '{col}' must be numeric")

        logger.debug("Column validation passed")

    def get_summary(self) -> dict:
        """
        Get a summary of the loaded data.

        Returns:
            Dictionary with data summary statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}

        return {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "required_present": all(
                col in self.data.columns for col in self.REQUIRED_COLUMNS
            ),
            "optional_present": [
                col for col in self.OPTIONAL_COLUMNS if col in self.data.columns
            ],
            "missing_values": self.data.isnull().sum().to_dict(),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
        }

    @staticmethod
    def generate_sample_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
        """
        Generate sample menu data for testing.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with synthetic menu data
        """
        np.random.seed(seed)

        cogs = np.random.uniform(5, 25, n_samples)
        # Ensure price > COGS with markup
        markup = np.random.uniform(1.2, 2.5, n_samples)
        current_price = cogs * markup

        data = pd.DataFrame(
            {
                "item_name": [f"Item_{i}" for i in range(n_samples)],
                "current_price": current_price,
                "cogs": cogs,
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

        return data
