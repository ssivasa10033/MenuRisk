"""
Configuration file for Canadian-specific settings
Updated with REAL 2025 Canadian Economic Data

Data Sources:
- Bank of Canada Rate: 2.25% (Oct 29, 2025)
- CPI Inflation: 2.4% (Sept 2025, Statistics Canada)
- Core Inflation: ~2.5%

Author: Seon Sivasathan
LinkedIn: https://www.linkedin.com/in/seon-sivasathan
Institution: Computer Science @ Western University
"""

# ==============================================================================
# REAL 2025 CANADIAN ECONOMIC DATA
# ==============================================================================

# Bank of Canada overnight rate (as of October 29, 2025)
# Source: Bank of Canada announcement Oct 29, 2025
RISK_FREE_RATE = 0.0225  # 2.25%

# Current inflation data (as of September 2025)
# Source: Statistics Canada CPI report
CURRENT_CPI_INFLATION = 0.024  # 2.4%
CURRENT_CORE_INFLATION = 0.025  # 2.5%

# Economic context notes (2025)
ECONOMIC_CONTEXT_2025 = {
    'trade_war': 'US-Canada trade tensions with tariff threats',
    'labor_market': 'Unemployment at 7.1%, weakness in trade-sensitive sectors',
    'rate_outlook': 'Policy rate in neutral range, likely to pause cuts',
    'gdp_growth': 'Economy showing signs of slowing due to trade uncertainty'
}

# ==============================================================================
# CANADIAN TAX RATES BY PROVINCE (2025)
# ==============================================================================

TAX_RATES = {
    'ON': 0.13,      # HST (Harmonized Sales Tax)
    'BC': 0.12,      # GST (5%) + PST (7%)
    'AB': 0.05,      # GST only
    'QC': 0.14975,   # GST (5%) + QST (9.975%)
    'SK': 0.11,      # GST (5%) + PST (6%)
    'MB': 0.12,      # GST (5%) + PST (7%)
    'NB': 0.15,      # HST
    'NS': 0.15,      # HST
    'PE': 0.15,      # HST
    'NL': 0.15,      # HST
    'YT': 0.05,      # GST only
    'NT': 0.05,      # GST only
    'NU': 0.05,      # GST only
}

# ==============================================================================
# SEASONALITY FACTORS FOR CANADIAN CLIMATE
# ==============================================================================

SEASONAL_FACTORS = {
    'Winter': 0.75,   # Dec-Feb: Reduced outdoor events
    'Spring': 1.0,    # Mar-May: Normal
    'Summer': 1.35,   # Jun-Aug: Peak wedding/event season
    'Fall': 0.95      # Sep-Nov: Slightly below normal
}

# ==============================================================================
# CANADIAN HOLIDAYS (2025)
# ==============================================================================

CANADIAN_HOLIDAYS = [
    '01-01',  # New Year's Day
    '02-17',  # Family Day (varies by province)
    '03-29',  # Good Friday 2025
    '04-01',  # Easter Monday 2025
    '05-19',  # Victoria Day 2025 (Monday before May 25)
    '07-01',  # Canada Day
    '08-04',  # Civic Holiday (first Monday of August)
    '09-01',  # Labour Day 2025 (first Monday of September)
    '10-13',  # Thanksgiving 2025 (second Monday of October)
    '11-11',  # Remembrance Day
    '12-25',  # Christmas
    '12-26',  # Boxing Day
]

# ==============================================================================
# MACHINE LEARNING MODEL PARAMETERS
# ==============================================================================

ML_CONFIG = {
    'n_estimators': 300,      # Increased from 100 for better accuracy
    'max_depth': 15,          # Increased from 10 for capturing complex patterns
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5            # Time-series cross-validation folds
}

# Monte Carlo simulation parameters
MONTE_CARLO_ITERATIONS = 10000

# ==============================================================================
# PRICE OPTIMIZATION CONSTRAINTS
# ==============================================================================

OPTIMIZATION_CONSTRAINTS = {
    'min_margin': 0.10,              # Minimum 10% profit margin
    'max_price_multiplier': 5.0,     # Max price = 5x COGS
    'min_price_multiplier': 1.1,     # Min price = 1.1x COGS (covers overhead)
    'elasticity_test_pct': 0.01      # Test price changes of ±1% for elasticity
}

# ==============================================================================
# RECOMMENDATION THRESHOLDS
# ==============================================================================

RECOMMENDATION_THRESHOLDS = {
    'keep_sharpe': 1.5,      # Sharpe ratio > 1.5 = Keep
    'monitor_sharpe': 0.8,   # 0.8 < Sharpe < 1.5 = Monitor
    'remove_sharpe': 0.8     # Sharpe ratio < 0.8 = Remove
}

# ==============================================================================
# PROJECT METADATA
# ==============================================================================

PROJECT_INFO = {
    'name': 'Menu Portfolio Optimizer',
    'version': '1.0.0',
    'author': 'Seon Sivasathan',
    'institution': 'Computer Science @ Western University',
    'linkedin': 'https://www.linkedin.com/in/seon-sivasathan',
    'data_year': 2025,
    'last_updated': '2025-11-16'
}
