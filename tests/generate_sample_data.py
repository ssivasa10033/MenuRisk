"""
Generate synthetic menu data for testing.

Creates realistic time-series sales data with:
- Price variations
- Seasonal patterns
- Weekly cycles
- Random noise
- Price elasticity effects

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_menu_data(
    n_items: int = 20,
    n_days: int = 365,
    start_date: str = "2024-01-01",
    include_seasonality: bool = True,
    include_price_changes: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic menu sales data with realistic patterns.

    The data generation model:
    1. Each item has a base demand level
    2. Demand responds to price changes via elasticity
    3. Seasonal patterns affect demand
    4. Day-of-week effects (weekends busier)
    5. Random noise is added

    Args:
        n_items: Number of menu items to generate
        n_days: Number of days of history
        start_date: Starting date (YYYY-MM-DD format)
        include_seasonality: Whether to add seasonal demand patterns
        include_price_changes: Whether to vary prices over time
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
        - date: Date of sale
        - item_name: Menu item identifier
        - price: Price at time of sale (current_price)
        - cogs: Cost of goods sold
        - quantity_sold: Units sold (demand)
        - category: Item category
        - season: Season (Winter/Spring/Summer/Fall)
        - province: Province (default ON)
    """
    np.random.seed(seed)

    logger.info(f"Generating synthetic data: {n_items} items over {n_days} days")

    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    # Define categories and their characteristics
    categories = ["Appetizer", "Main", "Dessert", "Beverage"]
    category_traits = {
        "Appetizer": {
            "cogs_range": (3, 8),
            "base_demand": (15, 40),
            "markup": (1.8, 2.5),
        },
        "Main": {"cogs_range": (8, 18), "base_demand": (25, 60), "markup": (2.0, 3.5)},
        "Dessert": {
            "cogs_range": (2, 6),
            "base_demand": (10, 35),
            "markup": (2.5, 4.0),
        },
        "Beverage": {
            "cogs_range": (1, 4),
            "base_demand": (20, 50),
            "markup": (3.0, 5.0),
        },
    }

    # Create menu items with properties
    items = []
    for i in range(n_items):
        category = np.random.choice(categories)
        traits = category_traits[category]

        cogs = np.random.uniform(*traits["cogs_range"])
        markup = np.random.uniform(*traits["markup"])
        base_price = cogs * markup
        base_demand = np.random.uniform(*traits["base_demand"])

        # Price elasticity: typically negative for normal goods
        # More negative = more price-sensitive
        price_elasticity = np.random.uniform(-2.5, -0.5)

        # Seasonality amplitude
        seasonality_amp = np.random.uniform(0.1, 0.3) if include_seasonality else 0

        items.append(
            {
                "item_name": f"{category}_{i+1}",
                "category": category,
                "base_cogs": cogs,
                "base_price": base_price,
                "base_demand": base_demand,
                "price_elasticity": price_elasticity,
                "seasonality_amplitude": seasonality_amp,
            }
        )

    logger.info(f"Created {len(items)} items across {len(categories)} categories")

    # Generate time series data
    records = []

    for date in dates:
        # Determine season
        month = date.month
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4, 5]:
            season = "Spring"
        elif month in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"

        # Seasonal factor (Summer is busier for restaurants)
        day_of_year = date.dayofyear
        season_factor = 1.0
        if include_seasonality:
            # Sinusoidal pattern with peak in summer
            season_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Day of week factor (weekends busier)
        # Friday=4, Saturday=5, Sunday=6
        dow = date.dayofweek
        if dow in [4, 5, 6]:  # Fri-Sun
            dow_factor = 1.3
        elif dow in [0, 3]:  # Mon, Thu
            dow_factor = 1.1
        else:  # Tue, Wed
            dow_factor = 1.0

        for item in items:
            # Determine price for this day
            if (
                include_price_changes and np.random.random() < 0.05
            ):  # 5% chance of price change
                # Price changes are typically small adjustments
                price = item["base_price"] * np.random.uniform(0.9, 1.1)
            else:
                price = item["base_price"]

            # Calculate expected demand using price elasticity
            # Q = Q_base * (P/P_base)^elasticity
            price_ratio = price / item["base_price"]
            price_effect = price_ratio ** item["price_elasticity"]

            # Combine all factors
            expected_demand = (
                item["base_demand"] * price_effect * season_factor * dow_factor
            )

            # Add item-specific seasonality noise
            seasonal_noise = 1 + item["seasonality_amplitude"] * np.sin(
                2 * np.pi * day_of_year / 365 + np.random.uniform(0, 2 * np.pi)
            )

            expected_demand *= seasonal_noise

            # Add random daily variation (±20%)
            noise_factor = np.random.normal(1.0, 0.15)
            expected_demand *= noise_factor

            # Convert to integer quantity (can't sell fractional items)
            quantity_sold = max(0, int(np.round(expected_demand)))

            # COGS can vary slightly (supplier price changes)
            cogs = item["base_cogs"] * np.random.uniform(0.95, 1.05)

            records.append(
                {
                    "date": date,
                    "item_name": item["item_name"],
                    "current_price": round(price, 2),
                    "cogs": round(cogs, 2),
                    "quantity_sold": quantity_sold,
                    "category": item["category"],
                    "season": season,
                    "province": "ON",  # Default to Ontario
                }
            )

    df = pd.DataFrame(records)

    # Calculate some summary statistics
    logger.info(f"Generated {len(df)} records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Unique items: {df['item_name'].nunique()}")
    logger.info(
        f"Avg daily sales per item: {df.groupby('date')['quantity_sold'].sum().mean():.1f}"
    )
    logger.info(f"Avg price: ${df['current_price'].mean():.2f}")
    logger.info(f"Avg COGS: ${df['cogs'].mean():.2f}")

    return df


def generate_test_data_with_events(
    n_items: int = 10, n_days: int = 180, n_events: int = 5, seed: int = 42
) -> pd.DataFrame:
    """
    Generate test data with special events (holidays, promotions, etc.).

    Args:
        n_items: Number of menu items
        n_days: Number of days
        n_events: Number of special events to include
        seed: Random seed

    Returns:
        DataFrame with additional event_type column
    """
    # Generate base data
    df = generate_synthetic_menu_data(n_items=n_items, n_days=n_days, seed=seed)

    # Add random events
    np.random.seed(seed)

    # Most days have no events
    df["event_type"] = "none"

    # Add some event days
    event_dates = np.random.choice(df["date"].unique(), size=n_events, replace=False)

    for event_date in event_dates:
        event_type = np.random.choice(["wedding", "corporate", "birthday", "holiday"])
        df.loc[df["date"] == event_date, "event_type"] = event_type

        # Events increase demand
        event_boost = np.random.uniform(1.3, 1.8)
        df.loc[df["date"] == event_date, "quantity_sold"] = (
            df.loc[df["date"] == event_date, "quantity_sold"] * event_boost
        ).astype(int)

    return df


def save_sample_data(output_path: str = "data/sample_menu_data_timeseries.csv"):
    """
    Generate and save sample data to CSV.

    Args:
        output_path: Path to save the CSV file
    """
    import os

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate data
    df = generate_synthetic_menu_data(
        n_items=20, n_days=365, include_seasonality=True, include_price_changes=True
    )

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sample data to {output_path}")

    # Display sample
    print("\nFirst 10 rows:")
    print(df.head(10))

    # Display statistics
    print("\nData statistics by item:")
    print(
        df.groupby("item_name")
        .agg(
            {
                "quantity_sold": ["mean", "std", "min", "max"],
                "current_price": ["mean", "min", "max"],
            }
        )
        .head()
    )

    return df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Generate and save sample data
    df = save_sample_data("data/sample_menu_data_timeseries.csv")

    print(f"\n[OK] Generated {len(df)} records for {df['item_name'].nunique()} items")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
