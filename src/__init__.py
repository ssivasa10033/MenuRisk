"""
Menu Portfolio Optimizer Package

A quantitative finance application for menu optimization using
Modern Portfolio Theory and Machine Learning.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

from src.models.optimizer import MenuPriceOptimizer
from src.finance.risk_metrics import PortfolioAnalyzer
from src.data.loader import DataLoader
from src.visualization.charts import ModelVisualizer

__version__ = "1.0.0"
__author__ = "Seon Sivasathan"

__all__ = [
    "MenuPriceOptimizer",
    "PortfolioAnalyzer",
    "DataLoader",
    "ModelVisualizer",
]
