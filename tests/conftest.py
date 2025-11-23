"""
Pytest configuration and fixtures.

Provides shared test data and fixtures for all tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.optimizer import MenuPriceOptimizer
from src.data.loader import DataLoader


@pytest.fixture
def sample_data():
    """Generate sample menu data for testing."""
    np.random.seed(42)
    n_samples = 200

    cogs = np.random.uniform(5, 25, n_samples)
    markup = np.random.uniform(1.2, 2.5, n_samples)
    current_price = cogs * markup

    return pd.DataFrame({
        'item_name': [f'Item_{i}' for i in range(n_samples)],
        'current_price': current_price,
        'cogs': cogs,
        'quantity_sold': np.random.randint(10, 200, n_samples),
        'category': np.random.choice(
            ['Appetizer', 'Main', 'Dessert', 'Beverage'], n_samples
        ),
        'season': np.random.choice(
            ['Winter', 'Spring', 'Summer', 'Fall'], n_samples
        ),
        'province': np.random.choice(['ON', 'BC', 'AB', 'QC'], n_samples)
    })


@pytest.fixture
def high_quality_data():
    """Generate data with strong patterns for accuracy tests."""
    np.random.seed(42)
    n_samples = 1000

    cogs = np.random.uniform(10, 30, n_samples)
    quantity_sold = np.random.randint(50, 300, n_samples)
    categories = np.random.choice(['Appetizer', 'Main', 'Dessert'], n_samples)
    seasons = np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples)

    # Strong patterns for ML to learn
    category_multipliers = {'Appetizer': 0.8, 'Main': 1.2, 'Dessert': 1.0}
    season_multipliers = {'Winter': 0.9, 'Spring': 1.0, 'Summer': 1.15, 'Fall': 0.95}

    category_effect = np.array([category_multipliers[c] for c in categories])
    season_effect = np.array([season_multipliers[s] for s in seasons])
    cogs_effect = 1 + (cogs - np.mean(cogs)) / np.std(cogs) * 0.1
    quantity_effect = 1 + (quantity_sold - np.mean(quantity_sold)) / np.std(quantity_sold) * 0.05

    base_margin = 0.5
    target_margin = (
        base_margin *
        category_effect *
        season_effect *
        cogs_effect *
        quantity_effect +
        np.random.normal(0, 0.05, n_samples)
    )
    target_margin = np.clip(target_margin, 0.1, 2.0)
    current_price = cogs * (1 + target_margin)

    return pd.DataFrame({
        'item_name': [f'Item_{i}' for i in range(n_samples)],
        'current_price': current_price,
        'cogs': cogs,
        'quantity_sold': quantity_sold,
        'category': categories,
        'season': seasons,
        'province': np.random.choice(['ON', 'BC', 'AB', 'QC'], n_samples)
    })


@pytest.fixture
def trained_model(sample_data):
    """Provide a pre-trained model."""
    model = MenuPriceOptimizer(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.train(sample_data)
    return model


@pytest.fixture
def data_loader():
    """Provide a DataLoader instance."""
    return DataLoader()
