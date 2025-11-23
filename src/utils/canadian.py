"""
Canadian-specific utilities for tax rates and seasonality.

Provides functions for Canadian provincial tax calculations
and seasonal adjustment factors.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

from typing import Optional, List

# Canadian Tax Rates by Province (2025)
TAX_RATES = {
    "ON": 0.13,  # HST (Harmonized Sales Tax)
    "BC": 0.12,  # GST (5%) + PST (7%)
    "AB": 0.05,  # GST only
    "QC": 0.14975,  # GST (5%) + QST (9.975%)
    "SK": 0.11,  # GST (5%) + PST (6%)
    "MB": 0.12,  # GST (5%) + PST (7%)
    "NB": 0.15,  # HST
    "NS": 0.15,  # HST
    "PE": 0.15,  # HST
    "NL": 0.15,  # HST
    "YT": 0.05,  # GST only
    "NT": 0.05,  # GST only
    "NU": 0.05,  # GST only
}

# Seasonal Factors for Canadian Climate
SEASONAL_FACTORS = {
    "Winter": 0.75,  # Dec-Feb: Reduced outdoor events
    "Spring": 1.0,  # Mar-May: Normal
    "Summer": 1.35,  # Jun-Aug: Peak wedding/event season
    "Fall": 0.95,  # Sep-Nov: Slightly below normal
}

# Canadian Holidays (2025)
CANADIAN_HOLIDAYS = [
    "01-01",  # New Year's Day
    "02-17",  # Family Day (varies by province)
    "03-29",  # Good Friday 2025
    "04-01",  # Easter Monday 2025
    "05-19",  # Victoria Day 2025
    "07-01",  # Canada Day
    "08-04",  # Civic Holiday
    "09-01",  # Labour Day 2025
    "10-13",  # Thanksgiving 2025
    "11-11",  # Remembrance Day
    "12-25",  # Christmas
    "12-26",  # Boxing Day
]


def get_tax_rate(province: str, default: float = 0.13) -> float:
    """
    Get the tax rate for a Canadian province.

    Args:
        province: Two-letter province code (e.g., 'ON', 'BC')
        default: Default rate if province not found (default: 0.13)

    Returns:
        Tax rate as a decimal (e.g., 0.13 for 13%)
    """
    return TAX_RATES.get(province.upper(), default)


def get_seasonal_factor(season: str, default: float = 1.0) -> float:
    """
    Get the seasonal adjustment factor.

    Args:
        season: Season name ('Winter', 'Spring', 'Summer', 'Fall')
        default: Default factor if season not found (default: 1.0)

    Returns:
        Seasonal adjustment factor
    """
    return SEASONAL_FACTORS.get(season.capitalize(), default)


def get_province_list() -> List[str]:
    """
    Get list of all Canadian province/territory codes.

    Returns:
        List of two-letter province codes
    """
    return list(TAX_RATES.keys())


def is_holiday(date_str: str) -> bool:
    """
    Check if a date (MM-DD format) is a Canadian holiday.

    Args:
        date_str: Date string in MM-DD format

    Returns:
        True if the date is a holiday
    """
    return date_str in CANADIAN_HOLIDAYS


def calculate_net_revenue(
    gross_revenue: float, province: str, season: Optional[str] = None
) -> float:
    """
    Calculate net revenue after tax and seasonal adjustment.

    Args:
        gross_revenue: Revenue before tax
        province: Province code for tax calculation
        season: Optional season for adjustment

    Returns:
        Net revenue after adjustments
    """
    tax_rate = get_tax_rate(province)
    net = gross_revenue / (1 + tax_rate)

    if season:
        seasonal_factor = get_seasonal_factor(season)
        net *= seasonal_factor

    return net
