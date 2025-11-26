"""
Generate realistic Canadian catering sample data for MenuRisk demo.

This script creates 6 months of realistic daily sales data for a Canadian
catering business with seasonal variations, realistic pricing, and
Canadian-specific considerations.

Author: Seon Sivasathan
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Canadian Catering Menu Items (realistic 2025 pricing)
MENU_ITEMS = {
    # Appetizers
    'Samosas (6pc)': {'cogs': 1.50, 'base_price': 4.99, 'category': 'Appetizer', 'base_volume': 120},
    'Spring Rolls (8pc)': {'cogs': 1.80, 'base_price': 5.99, 'category': 'Appetizer', 'base_volume': 100},
    'Bruschetta Platter': {'cogs': 3.50, 'base_price': 9.99, 'category': 'Appetizer', 'base_volume': 60},
    'Chicken Wings (1lb)': {'cogs': 4.20, 'base_price': 12.99, 'category': 'Appetizer', 'base_volume': 85},
    'Caesar Salad': {'cogs': 2.80, 'base_price': 8.99, 'category': 'Appetizer', 'base_volume': 70},
    'Caprese Skewers': {'cogs': 3.20, 'base_price': 10.99, 'category': 'Appetizer', 'base_volume': 55},

    # Main Courses
    'Butter Chicken': {'cogs': 7.50, 'base_price': 18.99, 'category': 'Main', 'base_volume': 95},
    'Beef Lasagna': {'cogs': 6.80, 'base_price': 16.99, 'category': 'Main', 'base_volume': 80},
    'Grilled Salmon': {'cogs': 9.50, 'base_price': 24.99, 'category': 'Main', 'base_volume': 65},
    'Chicken Biryani': {'cogs': 6.50, 'base_price': 16.99, 'category': 'Main', 'base_volume': 75},
    'Vegetable Stir-Fry': {'cogs': 4.50, 'base_price': 13.99, 'category': 'Main', 'base_volume': 60},
    'AAA Ribeye Steak': {'cogs': 12.50, 'base_price': 32.99, 'category': 'Main', 'base_volume': 45},
    'Pad Thai': {'cogs': 5.20, 'base_price': 14.99, 'category': 'Main', 'base_volume': 70},
    'Maple Glazed Pork': {'cogs': 8.20, 'base_price': 21.99, 'category': 'Main', 'base_volume': 55},

    # Desserts
    'Butter Tart (3pc)': {'cogs': 2.10, 'base_price': 6.99, 'category': 'Dessert', 'base_volume': 90},
    'Nanaimo Bars (2pc)': {'cogs': 1.90, 'base_price': 5.99, 'category': 'Dessert', 'base_volume': 85},
    'Maple Cheesecake': {'cogs': 3.50, 'base_price': 9.99, 'category': 'Dessert', 'base_volume': 65},
    'Tiramisu': {'cogs': 3.20, 'base_price': 8.99, 'category': 'Dessert', 'base_volume': 70},
    'Chocolate Lava Cake': {'cogs': 2.80, 'base_price': 7.99, 'category': 'Dessert', 'base_volume': 75},

    # Beverages
    'Mango Lassi': {'cogs': 1.20, 'base_price': 3.99, 'category': 'Beverage', 'base_volume': 110},
    'Craft Beer (Local)': {'cogs': 2.50, 'base_price': 6.99, 'category': 'Beverage', 'base_volume': 95},
    'BC Okanagan Wine': {'cogs': 4.50, 'base_price': 12.99, 'category': 'Beverage', 'base_volume': 60},
    'Iced Coffee': {'cogs': 0.90, 'base_price': 3.49, 'category': 'Beverage', 'base_volume': 100},
    'Fresh Lemonade': {'cogs': 0.80, 'base_price': 2.99, 'category': 'Beverage', 'base_volume': 105},

    # Sides
    'Naan Bread (2pc)': {'cogs': 0.75, 'base_price': 2.99, 'category': 'Side', 'base_volume': 150},
    'Garlic Bread': {'cogs': 0.90, 'base_price': 3.49, 'category': 'Side', 'base_volume': 130},
    'Roasted Vegetables': {'cogs': 2.20, 'base_price': 6.99, 'category': 'Side', 'base_volume': 85},
    'Poutine': {'cogs': 3.50, 'base_price': 8.99, 'category': 'Side', 'base_volume': 95},
    'Rice Pilaf': {'cogs': 1.50, 'base_price': 4.99, 'category': 'Side', 'base_volume': 100},
}

# Canadian provinces (weighted by population for realistic distribution)
PROVINCES = ['ON', 'BC', 'AB', 'QC', 'MB', 'SK', 'NS', 'NB']
PROVINCE_WEIGHTS = [0.40, 0.20, 0.15, 0.15, 0.04, 0.03, 0.02, 0.01]

# Season mapping
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Seasonal multipliers (Canadian climate impact)
SEASONAL_MULTIPLIERS = {
    'Winter': 0.75,   # Lower catering demand Dec-Feb
    'Spring': 1.0,    # Normal
    'Summer': 1.35,   # Peak wedding/event season
    'Fall': 0.95      # Slightly below normal
}

# Day of week multipliers (more events on weekends)
DOW_MULTIPLIERS = {
    0: 0.7,   # Monday
    1: 0.75,  # Tuesday
    2: 0.8,   # Wednesday
    3: 0.85,  # Thursday
    4: 1.1,   # Friday
    5: 1.4,   # Saturday
    6: 1.2,   # Sunday
}

def generate_realistic_data(start_date='2025-01-01', months=6):
    """
    Generate realistic Canadian catering sales data.

    Args:
        start_date: Starting date for data generation
        months: Number of months of data to generate

    Returns:
        DataFrame with realistic sales data
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = start + timedelta(days=30 * months)

    data = []

    current_date = start
    while current_date < end:
        # Skip some random days (business doesn't operate every day)
        if np.random.random() < 0.15:  # 15% chance of no business
            current_date += timedelta(days=1)
            continue

        season = get_season(current_date)
        dow = current_date.weekday()

        # Randomly select province for this day's orders
        province = np.random.choice(PROVINCES, p=PROVINCE_WEIGHTS)

        # Generate orders for subset of menu items (not all items sold every day)
        items_to_sell = np.random.choice(
            list(MENU_ITEMS.keys()),
            size=np.random.randint(15, len(MENU_ITEMS)),
            replace=False
        )

        for item_name in items_to_sell:
            item = MENU_ITEMS[item_name]

            # Calculate realistic quantity with multiple factors
            base_qty = item['base_volume']
            seasonal_factor = SEASONAL_MULTIPLIERS[season]
            dow_factor = DOW_MULTIPLIERS[dow]
            random_factor = np.random.normal(1.0, 0.2)  # 20% std dev

            # Category-specific seasonal adjustments
            if item['category'] == 'Beverage' and season == 'Summer':
                seasonal_factor *= 1.2  # Extra boost for beverages in summer
            elif item['category'] == 'Dessert' and season == 'Winter':
                seasonal_factor *= 1.1  # Holiday desserts popular

            quantity = int(base_qty * seasonal_factor * dow_factor * random_factor)
            quantity = max(5, quantity)  # Minimum 5 units

            # Add small price variations (±5%) to simulate dynamic pricing
            price_variation = np.random.uniform(0.95, 1.05)
            current_price = round(item['base_price'] * price_variation, 2)

            # Add small COGS variations (±3%) for supply cost fluctuations
            cogs_variation = np.random.uniform(0.97, 1.03)
            cogs = round(item['cogs'] * cogs_variation, 2)

            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'item_name': item_name,
                'current_price': current_price,
                'cogs': cogs,
                'quantity_sold': quantity,
                'category': item['category'],
                'season': season,
                'province': province
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    print("Generating realistic Canadian catering sample data...")
    print("=" * 60)

    # Generate 6 months of data
    df = generate_realistic_data(start_date='2025-01-01', months=6)

    # Sort by date and item name
    df = df.sort_values(['date', 'item_name']).reset_index(drop=True)

    # Save to CSV
    output_path = 'data/canadian_catering_sample.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✓ Generated {len(df):,} rows of realistic sales data")
    print(f"✓ Saved to: {output_path}")
    print(f"\nData Summary:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique items: {df['item_name'].nunique()}")
    print(f"  Provinces: {sorted(df['province'].unique())}")
    print(f"  Seasons: {sorted(df['season'].unique())}")
    print(f"  Categories: {sorted(df['category'].unique())}")
    print(f"\nPrice range by category:")
    for category in sorted(df['category'].unique()):
        cat_data = df[df['category'] == category]
        print(f"  {category:12s}: ${cat_data['current_price'].min():.2f} - ${cat_data['current_price'].max():.2f}")

    print(f"\nTotal revenue: ${(df['current_price'] * df['quantity_sold']).sum():,.2f}")
    print(f"Total COGS: ${(df['cogs'] * df['quantity_sold']).sum():,.2f}")
    print(f"Gross profit: ${((df['current_price'] - df['cogs']) * df['quantity_sold']).sum():,.2f}")
    print(f"Avg margin: {(((df['current_price'] - df['cogs']) / df['current_price']).mean() * 100):.1f}%")

    print("\n" + "=" * 60)
    print("Sample data ready for MenuRisk demo!")
