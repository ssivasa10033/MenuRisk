"""
Data validation using Pydantic for menu optimization.

Provides schema validation for CSV uploads and API requests.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
from datetime import date
from typing import List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

logger = logging.getLogger(__name__)

# Valid Canadian provinces/territories
VALID_PROVINCES = [
    "AB",
    "BC",
    "MB",
    "NB",
    "NL",
    "NS",
    "NT",
    "NU",
    "ON",
    "PE",
    "QC",
    "SK",
    "YT",
]

# Valid categories
VALID_CATEGORIES = ["Appetizer", "Main", "Dessert", "Beverage"]

# Valid seasons
VALID_SEASONS = ["Winter", "Spring", "Summer", "Fall"]


class MenuItemSchema(BaseModel):
    """Schema for a single menu item transaction."""

    model_config = ConfigDict(extra="ignore")

    item_name: str = Field(..., min_length=1, max_length=100)
    current_price: float = Field(..., gt=0, description="Must be positive")
    cogs: float = Field(..., gt=0, description="Cost of goods sold")
    quantity_sold: int = Field(..., ge=0, description="Cannot be negative")
    category: Optional[str] = None
    season: Optional[str] = None
    province: Optional[str] = None
    date: Optional[Union[date, str]] = None
    event_type: Optional[str] = None

    @field_validator("cogs")
    @classmethod
    def cogs_less_than_price(cls, v: float, info) -> float:
        """COGS must be less than selling price."""
        if "current_price" in info.data and v >= info.data["current_price"]:
            raise ValueError(
                f"COGS ({v}) must be less than price ({info.data['current_price']})"
            )
        return v

    @field_validator("quantity_sold")
    @classmethod
    def reasonable_quantity(cls, v: int) -> int:
        """Flag unreasonably high quantities."""
        if v > 10000:
            raise ValueError(f"Quantity ({v}) seems unreasonably high. Please verify.")
        return v

    @field_validator("province")
    @classmethod
    def valid_canadian_province(cls, v: Optional[str]) -> Optional[str]:
        """Validate Canadian province codes."""
        if v is not None and v not in VALID_PROVINCES:
            raise ValueError(
                f"Invalid province code: {v}. "
                f"Valid codes: {', '.join(VALID_PROVINCES)}"
            )
        return v

    @field_validator("category")
    @classmethod
    def valid_category(cls, v: Optional[str]) -> Optional[str]:
        """Validate category."""
        if v is not None and v not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {v}. "
                f"Valid categories: {', '.join(VALID_CATEGORIES)}"
            )
        return v

    @field_validator("season")
    @classmethod
    def valid_season(cls, v: Optional[str]) -> Optional[str]:
        """Validate season."""
        if v is not None and v not in VALID_SEASONS:
            raise ValueError(
                f"Invalid season: {v}. " f"Valid seasons: {', '.join(VALID_SEASONS)}"
            )
        return v

    @model_validator(mode="after")
    def check_profit_margin(self) -> "MenuItemSchema":
        """Validate profit margin is reasonable."""
        if self.current_price and self.cogs:
            margin = (self.current_price - self.cogs) / self.cogs
            if margin < 0:
                raise ValueError(
                    f"Negative profit margin ({margin:.2%}). "
                    "Price must be greater than COGS."
                )
            if margin > 10:  # 1000% markup
                logger.warning(
                    f"Very high margin ({margin:.2%}) for {self.item_name}. "
                    "Please verify pricing."
                )
        return self


class PredictionRequestSchema(BaseModel):
    """Schema for prediction API requests."""

    model_config = ConfigDict(extra="ignore")

    items: List[MenuItemSchema] = Field(
        ..., min_length=1, description="List of menu items to analyze"
    )
    include_recommendations: bool = True
    confidence_level: float = Field(
        default=0.90,
        ge=0.5,
        le=0.99,
        description="Confidence level for prediction intervals",
    )


class ValidationResult(BaseModel):
    """Result of data validation."""

    model_config = ConfigDict(extra="ignore")

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    valid_row_count: int
    invalid_row_count: int


class DataValidator:
    """
    Validate uploaded CSV data.

    This class provides comprehensive validation for menu data:
    - Schema validation using Pydantic
    - Statistical validation for data quality
    - Business rule validation
    """

    def __init__(
        self,
        min_observations: int = 30,
        max_file_size_mb: float = 16.0,
        require_date: bool = False,
    ):
        """
        Initialize the validator.

        Args:
            min_observations: Minimum recommended observations per item
            max_file_size_mb: Maximum file size in MB
            require_date: Whether date column is required
        """
        self.min_observations = min_observations
        self.max_file_size_mb = max_file_size_mb
        self.require_date = require_date

    def validate_csv(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate entire DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation status and errors/warnings
        """
        errors: List[str] = []
        warnings: List[str] = []
        invalid_rows: List[int] = []

        # Check required columns
        required_cols = ["item_name", "current_price", "cogs", "quantity_sold"]
        if self.require_date:
            required_cols.append("date")

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                valid_row_count=0,
                invalid_row_count=len(df),
            )

        # Validate each row
        for idx, row in df.iterrows():
            try:
                # Convert row to dict, handling NaN values
                row_dict = {}
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        row_dict[col] = None
                    elif col == "quantity_sold":
                        row_dict[col] = int(val)
                    elif col in ["current_price", "cogs"]:
                        row_dict[col] = float(val)
                    elif col == "date":
                        if isinstance(val, str):
                            row_dict[col] = val
                        elif hasattr(val, "date"):
                            row_dict[col] = val.date()
                        else:
                            row_dict[col] = str(val)
                    else:
                        row_dict[col] = val

                MenuItemSchema(**row_dict)

            except Exception as e:
                # +2 for 1-indexing + header row
                errors.append(f"Row {idx + 2}: {str(e)}")
                invalid_rows.append(idx)

        # Statistical validation
        if len(df) < 10:
            warnings.append(
                f"Only {len(df)} rows provided. "
                f"Recommend at least 30 observations per item for reliable analysis."
            )

        # Check observations per item
        if "item_name" in df.columns:
            item_counts = df["item_name"].value_counts()
            low_count_items = item_counts[item_counts < self.min_observations]
            if len(low_count_items) > 0:
                warnings.append(
                    f"{len(low_count_items)} items have fewer than "
                    f"{self.min_observations} observations. "
                    f"Items: {', '.join(low_count_items.index[:5])}..."
                )

        # Check for duplicate entries
        if "date" in df.columns:
            duplicates = df.duplicated(subset=["item_name", "date"], keep=False)
            if duplicates.any():
                dup_count = duplicates.sum()
                warnings.append(
                    f"{dup_count} duplicate item-date combinations found. "
                    "Consider aggregating duplicate entries."
                )

        # Check for negative values
        for col in ["current_price", "cogs"]:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    errors.append(f"{neg_count} rows have negative {col} values.")

        # Check for unreasonable margins
        if "current_price" in df.columns and "cogs" in df.columns:
            df_clean = df[(df["cogs"] > 0) & (df["current_price"] > 0)]
            if len(df_clean) > 0:
                margins = (df_clean["current_price"] - df_clean["cogs"]) / df_clean[
                    "cogs"
                ]
                neg_margin_count = (margins < 0).sum()
                if neg_margin_count > 0:
                    errors.append(
                        f"{neg_margin_count} items have negative profit margins "
                        "(price < COGS)."
                    )

        # Determine validity (errors are blockers, warnings are not)
        is_valid = (
            len([e for e in errors if not e.startswith("Row")]) == 0
            or len(invalid_rows) < len(df) * 0.5
        )  # Allow up to 50% invalid rows

        valid_count = len(df) - len(invalid_rows)

        return ValidationResult(
            is_valid=is_valid,
            errors=errors[:50],  # Limit errors to first 50
            warnings=warnings,
            valid_row_count=valid_count,
            invalid_row_count=len(invalid_rows),
        )

    def validate_file_size(self, file_size_bytes: int) -> bool:
        """
        Check if file size is within limits.

        Args:
            file_size_bytes: File size in bytes

        Returns:
            True if valid, False if too large
        """
        max_bytes = self.max_file_size_mb * 1024 * 1024
        return file_size_bytes <= max_bytes

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame by removing invalid rows.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame with only valid rows
        """
        df = df.copy()

        # Remove rows with missing required columns
        required_cols = ["item_name", "current_price", "cogs", "quantity_sold"]
        for col in required_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])

        # Remove rows with invalid values
        if "current_price" in df.columns:
            df = df[df["current_price"] > 0]
        if "cogs" in df.columns:
            df = df[df["cogs"] > 0]
        if "quantity_sold" in df.columns:
            df = df[df["quantity_sold"] >= 0]

        # Ensure COGS < price
        if "current_price" in df.columns and "cogs" in df.columns:
            df = df[df["cogs"] < df["current_price"]]

        logger.info(f"Cleaned DataFrame: {len(df)} valid rows")

        return df


def validate_upload(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Convenience function for validating uploads.

    Args:
        df: DataFrame from uploaded CSV

    Returns:
        Tuple of (is_valid, error_messages, cleaned_dataframe)
    """
    validator = DataValidator()
    result = validator.validate_csv(df)

    all_messages = result.errors + result.warnings

    if result.is_valid:
        cleaned_df = validator.clean_dataframe(df)
        return True, all_messages, cleaned_df
    else:
        return False, result.errors, df
