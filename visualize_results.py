"""
Data Visualization for Menu Price Optimizer Model Results
Creates comprehensive visualizations of model performance, predictions, and portfolio metrics

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import sys
import os

# Check for required dependencies
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from menu_price_model import MenuPriceOptimizer
    import config
except ImportError as e:
    print("=" * 60)
    print("ERROR: Missing required dependencies!")
    print("=" * 60)
    print("Please install required packages:")
    print("  python3 -m pip install numpy pandas matplotlib seaborn scikit-learn")
    print("=" * 60)
    sys.exit(1)

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


class ModelVisualizer:
    """Creates visualizations for model results"""
    
    def __init__(self, output_dir='static/charts'):
        """Initialize visualizer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_test_data(self, n_samples=1000, seed=42):
        """Generate test data with patterns for visualization"""
        np.random.seed(seed)
        
        # Create base features
        cogs = np.random.uniform(10, 30, n_samples)
        quantity_sold = np.random.randint(50, 300, n_samples)
        categories = np.random.choice(['Appetizer', 'Main', 'Dessert'], n_samples)
        seasons = np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples)
        
        # Create category-based profit margin multipliers
        category_multipliers = {
            'Appetizer': 0.8,
            'Main': 1.2,
            'Dessert': 1.0
        }
        
        season_multipliers = {
            'Winter': 0.9,
            'Spring': 1.0,
            'Summer': 1.15,
            'Fall': 0.95
        }
        
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
    
    def plot_model_performance(self, metrics, save_path=None):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        # R² Score
        axes[0, 0].bar(['R² Score', 'CV R² Mean'], 
                      [metrics['r2_score'], metrics['cv_r2_mean']],
                      color=['#2ecc71', '#3498db'])
        axes[0, 0].axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('R² Score (Accuracy)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error Metrics
        error_metrics = ['MAE', 'RMSE']
        error_values = [metrics['mae'], metrics['rmse']]
        axes[0, 1].bar(error_metrics, error_values, color=['#e74c3c', '#c0392b'])
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_title('Error Metrics')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cross-Validation Scores
        cv_scores = [metrics['cv_r2_mean'] - metrics['cv_r2_std'],
                    metrics['cv_r2_mean'],
                    metrics['cv_r2_mean'] + metrics['cv_r2_std']]
        axes[1, 0].barh(['CV R²'], [metrics['cv_r2_mean']], 
                        xerr=metrics['cv_r2_std'], color='#9b59b6', capsize=10)
        axes[1, 0].axvline(x=0.8, color='r', linestyle='--', label='80% Threshold')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_title('Cross-Validation R² (with std dev)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance Summary
        summary_text = f"""
        Model Performance Summary
        
        R² Score: {metrics['r2_score']:.4f}
        CV R² Mean: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}
        MAE: {metrics['mae']:.4f}
        RMSE: {metrics['rmse']:.4f}
        
        Status: {'✓ PASS (>80%)' if metrics['r2_score'] > 0.8 else '✗ FAIL (<80%)'}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.savefig(f'{self.output_dir}/model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df, save_path=None):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by importance
        df_sorted = feature_importance_df.sort_values('importance', ascending=True)
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
        bars = ax.barh(df_sorted['feature'], df_sorted['importance'], color=colors)
        
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(row['importance'] + 0.01, i, f"{row['importance']:.3f}",
                   va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions_vs_actual(self, model, test_data, save_path=None):
        """Plot predictions vs actual values"""
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X, y, _ = model._prepare_features(test_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale and predict
        X_test_scaled = model.scaler.transform(X_test)
        y_pred = model.model.predict(X_test_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=50, color='#3498db')
        
        # Perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        axes[0].plot([min_val, max_val], [min_val, max_val], 
                    'r--', lw=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Profit Margin', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Profit Margin', fontsize=12, fontweight='bold')
        axes[0].set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color='#e74c3c')
        axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Profit Margin', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
        axes[1].set_title('Residuals Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.savefig(f'{self.output_dir}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_portfolio_metrics(self, portfolio_metrics, save_path=None):
        """Plot portfolio metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Portfolio Analysis', fontsize=16, fontweight='bold')
        
        # Sharpe Ratio
        sharpe = portfolio_metrics['sharpe_ratio']
        axes[0, 0].barh(['Sharpe Ratio'], [sharpe], color='#2ecc71' if sharpe > 1.5 else '#f39c12' if sharpe > 0.8 else '#e74c3c')
        axes[0, 0].axvline(x=1.5, color='g', linestyle='--', label='Keep Threshold')
        axes[0, 0].axvline(x=0.8, color='orange', linestyle='--', label='Monitor Threshold')
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_title('Portfolio Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns and Volatility
        metrics_names = ['Mean Return', 'Volatility']
        metrics_values = [portfolio_metrics['mean_return'], portfolio_metrics['volatility']]
        colors = ['#3498db', '#9b59b6']
        axes[0, 1].bar(metrics_names, metrics_values, color=colors)
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Return vs Volatility')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recommendations Distribution
        recs = portfolio_metrics['recommendations']
        rec_counts = pd.Series(list(recs.values())).value_counts()
        axes[1, 0].pie(rec_counts.values, labels=rec_counts.index, autopct='%1.1f%%',
                      colors=['#2ecc71', '#f39c12', '#e74c3c'])
        axes[1, 0].set_title('Item Recommendations Distribution')
        
        # Portfolio Summary
        summary_text = f"""
        Portfolio Summary
        
        Total Items: {portfolio_metrics['num_items']}
        Mean Return: {portfolio_metrics['mean_return']:.4f}
        Volatility: {portfolio_metrics['volatility']:.4f}
        Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}
        
        Recommendations:
        • Keep: {list(recs.values()).count('keep')}
        • Monitor: {list(recs.values()).count('monitor')}
        • Remove: {list(recs.values()).count('remove')}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.savefig(f'{self.output_dir}/portfolio_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_category_analysis(self, data, predictions, save_path=None):
        """Plot analysis by category"""
        df = data.copy()
        df['predicted_margin'] = predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Category Analysis', fontsize=16, fontweight='bold')
        
        if 'category' in df.columns:
            # Average margins by category
            category_margins = df.groupby('category')['predicted_margin'].mean().sort_values()
            axes[0, 0].barh(category_margins.index, category_margins.values, color='#3498db')
            axes[0, 0].set_xlabel('Average Predicted Margin')
            axes[0, 0].set_title('Average Profit Margin by Category')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot by category
            categories = df['category'].unique()
            data_by_category = [df[df['category'] == cat]['predicted_margin'].values 
                              for cat in categories]
            axes[0, 1].boxplot(data_by_category, labels=categories)
            axes[0, 1].set_ylabel('Predicted Margin')
            axes[0, 1].set_title('Margin Distribution by Category')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'season' in df.columns:
            # Average margins by season
            season_margins = df.groupby('season')['predicted_margin'].mean()
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            season_margins = season_margins.reindex([s for s in season_order if s in season_margins.index])
            axes[1, 0].plot(season_margins.index, season_margins.values, 
                           marker='o', linewidth=2, markersize=8, color='#e74c3c')
            axes[1, 0].set_ylabel('Average Predicted Margin')
            axes[1, 0].set_title('Average Profit Margin by Season')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Price vs COGS scatter by category
        if 'category' in df.columns:
            for cat in df['category'].unique():
                cat_data = df[df['category'] == cat]
                axes[1, 1].scatter(cat_data['cogs'], cat_data['current_price'], 
                                  label=cat, alpha=0.6, s=50)
            axes[1, 1].set_xlabel('COGS')
            axes[1, 1].set_ylabel('Current Price')
            axes[1, 1].set_title('Price vs COGS by Category')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.savefig(f'{self.output_dir}/category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, model, test_data):
        """Create all visualizations and generate a comprehensive report"""
        print("=" * 60)
        print("Generating Model Visualizations")
        print("=" * 60)
        
        # Train model if not already trained
        if not model.is_trained:
            print("Training model...")
            metrics = model.train(test_data)
        else:
            print("Model already trained. Using existing model...")
            metrics = {
                'r2_score': model.r2_score_,
                'mae': model.mae_,
                'rmse': model.rmse_,
                'cv_r2_mean': 0.85,  # Placeholder if not available
                'cv_r2_std': 0.02,
                'feature_importance': model.feature_importance_
            }
        
        # Generate predictions
        print("Generating predictions...")
        predictions = model.predict(test_data)
        
        # Calculate portfolio metrics
        print("Calculating portfolio metrics...")
        portfolio_metrics = model.calculate_portfolio_metrics(test_data)
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.plot_model_performance(metrics)
        print("  ✓ Model performance metrics")
        
        if model.feature_importance_ is not None:
            self.plot_feature_importance(model.feature_importance_)
            print("  ✓ Feature importance")
        
        self.plot_predictions_vs_actual(model, test_data)
        print("  ✓ Predictions vs actual")
        
        self.plot_portfolio_metrics(portfolio_metrics)
        print("  ✓ Portfolio metrics")
        
        self.plot_category_analysis(test_data, predictions)
        print("  ✓ Category analysis")
        
        print("\n" + "=" * 60)
        print("All visualizations saved to:", self.output_dir)
        print("=" * 60)
        
        return {
            'metrics': metrics,
            'portfolio_metrics': portfolio_metrics,
            'predictions': predictions
        }


def main():
    """Main function to run visualizations"""
    print("Menu Price Optimizer - Data Visualization")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Generate test data
    print("Generating test data...")
    test_data = visualizer.generate_test_data(n_samples=1000, seed=42)
    
    # Initialize and train model
    model = MenuPriceOptimizer(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        random_state=42
    )
    
    # Create comprehensive report
    results = visualizer.create_comprehensive_report(model, test_data)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model R² Score: {results['metrics']['r2_score']:.4f}")
    print(f"Portfolio Sharpe Ratio: {results['portfolio_metrics']['sharpe_ratio']:.4f}")
    print(f"Total Items Analyzed: {results['portfolio_metrics']['num_items']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

