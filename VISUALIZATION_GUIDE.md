# Data Visualization Guide

## Overview
The visualization module creates comprehensive charts and graphs to analyze model performance, predictions, and portfolio metrics.

## Visualizations Created

### 1. Model Performance Metrics (`model_performance.png`)
- **R² Score**: Shows model accuracy (target: >0.8)
- **Cross-Validation R²**: Shows model generalization ability
- **Error Metrics**: MAE and RMSE
- **Performance Summary**: Text summary of all metrics

### 2. Feature Importance (`feature_importance.png`)
- Horizontal bar chart showing which features are most important
- Helps identify which factors (price, COGS, category, season, etc.) drive predictions
- Useful for understanding model behavior

### 3. Predictions vs Actual (`predictions_vs_actual.png`)
- **Scatter Plot**: Shows how well predictions match actual values
- **Residuals Plot**: Shows prediction errors distribution
- Helps identify if model has bias or systematic errors

### 4. Portfolio Metrics (`portfolio_metrics.png`)
- **Sharpe Ratio**: Risk-adjusted return metric
- **Return vs Volatility**: Risk-return tradeoff
- **Recommendations Distribution**: Pie chart of keep/monitor/remove items
- **Portfolio Summary**: Text summary of portfolio health

### 5. Category Analysis (`category_analysis.png`)
- **Average Margin by Category**: Which categories are most profitable
- **Margin Distribution**: Box plots showing variance by category
- **Seasonal Trends**: How margins vary by season
- **Price vs COGS**: Scatter plot showing pricing patterns

## Usage

### Run Visualizations Only
```bash
python3 visualize_results.py
```

### Run Tests and Visualizations Together
```bash
python3 run_tests_and_visualize.py
```

### Output Location
All charts are saved to: `static/charts/`

## Requirements
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

Install with:
```bash
python3 -m pip install -r requirements.txt
```

## Interpreting Results

### Good Model Performance
- R² Score > 0.8 (80% accuracy)
- Low MAE and RMSE
- Predictions close to diagonal line in scatter plot
- Residuals centered around zero

### Good Portfolio Health
- Sharpe Ratio > 1.5 (Keep items)
- Balanced recommendations (not all remove)
- Reasonable volatility
- Positive mean return

### Feature Insights
- High importance features should align with business logic
- Category and season effects should be visible
- Price-to-COGS ratio typically important

## Customization

You can modify `visualize_results.py` to:
- Change chart styles and colors
- Add additional visualizations
- Modify output directory
- Adjust figure sizes and DPI

## Author
Seon Sivasathan
Computer Science @ Western University

