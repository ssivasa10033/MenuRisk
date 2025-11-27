"""
Shared metric functions for model evaluation.
"""

import numpy as np


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value as percentage (0-100)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle zero values by using epsilon
    epsilon = 1e-10
    mask = np.abs(y_true) > epsilon

    if not np.any(mask):
        return 0.0

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(mape)


def directional_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0
) -> float:
    """
    Calculate directional accuracy (% of times prediction captures trend direction).

    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Threshold for considering direction change

    Returns:
        Directional accuracy as percentage (0-100)
    """
    if len(y_true) < 2:
        return 0.0

    # Calculate period-over-period changes
    true_direction = np.diff(y_true) > threshold
    pred_direction = np.diff(y_pred) > threshold

    accuracy = np.mean(true_direction == pred_direction) * 100
    return float(accuracy)
