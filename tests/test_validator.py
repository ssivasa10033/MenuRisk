"""
Tests for data validation module.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.validator import (
    DataValidator,
    MenuItemSchema,
    validate_upload,
    VALID_PROVINCES,
)


class TestMenuItemSchema:
    """Tests for MenuItemSchema Pydantic model."""

    def test_valid_item(self):
        """Test valid menu item creation."""
        item = MenuItemSchema(
            item_name="Butter Chicken",
            current_price=18.99,
            cogs=7.50,
            quantity_sold=45,
            category="Main",
            season="Summer",
            province="ON",
        )
        assert item.item_name == "Butter Chicken"
        assert item.current_price == 18.99
        assert item.cogs == 7.50

    def test_minimal_item(self):
        """Test item with only required fields."""
        item = MenuItemSchema(
            item_name="Test Item",
            current_price=10.00,
            cogs=5.00,
            quantity_sold=10,
        )
        assert item.category is None
        assert item.province is None

    def test_invalid_negative_price(self):
        """Test rejection of negative price."""
        with pytest.raises(ValidationError):
            MenuItemSchema(
                item_name="Test",
                current_price=-10.00,
                cogs=5.00,
                quantity_sold=10,
            )

    def test_invalid_negative_cogs(self):
        """Test rejection of negative COGS."""
        with pytest.raises(ValidationError):
            MenuItemSchema(
                item_name="Test",
                current_price=10.00,
                cogs=-5.00,
                quantity_sold=10,
            )

    def test_invalid_negative_quantity(self):
        """Test rejection of negative quantity."""
        with pytest.raises(ValidationError):
            MenuItemSchema(
                item_name="Test",
                current_price=10.00,
                cogs=5.00,
                quantity_sold=-10,
            )

    def test_invalid_province_code(self):
        """Test rejection of invalid province code."""
        with pytest.raises(ValidationError):
            MenuItemSchema(
                item_name="Test",
                current_price=10.00,
                cogs=5.00,
                quantity_sold=10,
                province="XX",
            )

    def test_valid_province_codes(self):
        """Test all valid Canadian province codes."""
        for province in VALID_PROVINCES:
            item = MenuItemSchema(
                item_name="Test",
                current_price=10.00,
                cogs=5.00,
                quantity_sold=10,
                province=province,
            )
            assert item.province == province

    def test_empty_item_name(self):
        """Test rejection of empty item name."""
        with pytest.raises(ValidationError):
            MenuItemSchema(
                item_name="",
                current_price=10.00,
                cogs=5.00,
                quantity_sold=10,
            )

    def test_very_high_quantity_warning(self):
        """Test that very high quantities are rejected."""
        with pytest.raises(ValidationError):
            MenuItemSchema(
                item_name="Test",
                current_price=10.00,
                cogs=5.00,
                quantity_sold=50000,
            )


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid test DataFrame."""
        return pd.DataFrame(
            {
                "item_name": ["Item A", "Item B", "Item C"] * 20,
                "current_price": [15.99, 12.50, 8.99] * 20,
                "cogs": [6.00, 5.00, 3.50] * 20,
                "quantity_sold": [40, 55, 80] * 20,
                "category": ["Main", "Appetizer", "Beverage"] * 20,
                "season": ["Summer", "Summer", "Summer"] * 20,
                "province": ["ON", "BC", "AB"] * 20,
            }
        )

    @pytest.fixture
    def invalid_dataframe(self):
        """Create an invalid test DataFrame."""
        return pd.DataFrame(
            {
                "item_name": ["Item A", "", "Item C"],
                "current_price": [15.99, -10.00, 8.99],
                "cogs": [6.00, 5.00, 20.00],  # COGS > price for Item C
                "quantity_sold": [40, 55, -5],
            }
        )

    def test_validate_valid_data(self, valid_dataframe):
        """Test validation of valid data."""
        validator = DataValidator()
        result = validator.validate_csv(valid_dataframe)

        assert result.is_valid
        assert result.valid_row_count == len(valid_dataframe)
        assert result.invalid_row_count == 0

    def test_validate_missing_columns(self):
        """Test validation catches missing required columns."""
        df = pd.DataFrame(
            {
                "item_name": ["A", "B"],
                "current_price": [10.0, 20.0],
                # Missing 'cogs' and 'quantity_sold'
            }
        )

        validator = DataValidator()
        result = validator.validate_csv(df)

        assert not result.is_valid
        assert any("Missing" in e for e in result.errors)

    def test_validate_with_invalid_rows(self, invalid_dataframe):
        """Test validation identifies invalid rows."""
        validator = DataValidator()
        result = validator.validate_csv(invalid_dataframe)

        # Should have errors for invalid rows
        assert len(result.errors) > 0

    def test_clean_dataframe(self, invalid_dataframe):
        """Test DataFrame cleaning removes invalid rows."""
        validator = DataValidator()

        # Add valid rows
        df = pd.concat(
            [
                invalid_dataframe,
                pd.DataFrame(
                    {
                        "item_name": ["Valid Item"],
                        "current_price": [15.00],
                        "cogs": [6.00],
                        "quantity_sold": [50],
                    }
                ),
            ],
            ignore_index=True,
        )

        cleaned = validator.clean_dataframe(df)

        # Should have fewer rows after cleaning
        assert len(cleaned) < len(df)
        # All remaining rows should have valid values
        assert (cleaned["current_price"] > 0).all()
        assert (cleaned["cogs"] > 0).all()
        assert (cleaned["cogs"] < cleaned["current_price"]).all()

    def test_low_observations_warning(self):
        """Test warning for low observation counts."""
        df = pd.DataFrame(
            {
                "item_name": ["A", "B", "C"],
                "current_price": [10.0, 15.0, 20.0],
                "cogs": [4.0, 6.0, 8.0],
                "quantity_sold": [5, 10, 15],
            }
        )

        validator = DataValidator(min_observations=30)
        result = validator.validate_csv(df)

        # Should have warning about low observations
        assert any("observations" in w.lower() for w in result.warnings)


class TestValidateUpload:
    """Tests for validate_upload convenience function."""

    def test_valid_upload(self):
        """Test valid upload returns success."""
        df = pd.DataFrame(
            {
                "item_name": ["Item " + str(i) for i in range(50)],
                "current_price": [15.99] * 50,
                "cogs": [6.00] * 50,
                "quantity_sold": [40] * 50,
            }
        )

        is_valid, messages, cleaned = validate_upload(df)

        assert is_valid
        assert len(cleaned) == len(df)

    def test_invalid_upload(self):
        """Test invalid upload returns errors."""
        df = pd.DataFrame(
            {
                "item_name": ["A"],
                # Missing required columns
            }
        )

        is_valid, messages, cleaned = validate_upload(df)

        assert not is_valid
        assert len(messages) > 0
