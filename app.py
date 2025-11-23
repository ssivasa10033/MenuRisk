"""
Menu Portfolio Optimizer - Flask Application

A web application for menu optimization using Modern Portfolio Theory
and Machine Learning.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import os
import logging
from typing import Dict

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd

from src.models.optimizer import MenuPriceOptimizer
from src.data.loader import DataLoader
from src.visualization.charts import ModelVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

# Global model instance (in production, use proper state management)
model: MenuPriceOptimizer = None
last_results: Dict = {}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', results=last_results)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    global model, last_results

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and validate data
        loader = DataLoader()
        data = loader.load_csv(filepath)

        # Initialize and train model
        model = MenuPriceOptimizer(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        metrics = model.train(data)
        predictions = model.predict(data)
        portfolio_metrics = model.calculate_portfolio_metrics(data)

        # Generate visualizations
        visualizer = ModelVisualizer()
        charts = visualizer.create_all_charts(
            metrics=metrics,
            portfolio_metrics=portfolio_metrics,
            feature_importance=model.feature_importance_,
            data=data,
            predictions=predictions
        )

        # Store results
        last_results = {
            'metrics': {
                'r2_score': round(metrics['r2_score'], 4),
                'mae': round(metrics['mae'], 4),
                'rmse': round(metrics['rmse'], 4),
                'cv_r2_mean': round(metrics['cv_r2_mean'], 4),
                'cv_r2_std': round(metrics['cv_r2_std'], 4)
            },
            'portfolio': {
                'sharpe_ratio': round(portfolio_metrics['sharpe_ratio'], 4),
                'mean_return': round(portfolio_metrics['mean_return'], 4),
                'volatility': round(portfolio_metrics['volatility'], 4),
                'num_items': portfolio_metrics['num_items']
            },
            'recommendations': portfolio_metrics['recommendations'],
            'charts': charts,
            'data_summary': loader.get_summary()
        }

        logger.info(f"Analysis complete. R2: {metrics['r2_score']:.4f}")

        return jsonify({
            'success': True,
            'results': last_results
        })

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/optimize', methods=['POST'])
def optimize_prices():
    """Optimize menu prices."""
    global model, last_results

    if model is None or not model.is_trained:
        return jsonify({'error': 'Model not trained. Please upload data first.'}), 400

    try:
        target_sharpe = request.json.get('target_sharpe', 1.5)
        min_margin = request.json.get('min_margin', 0.1)

        # Load the last uploaded data
        loader = DataLoader()
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if not files:
            return jsonify({'error': 'No data uploaded'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], files[-1])
        data = loader.load_csv(filepath)

        # Optimize prices
        optimized = model.optimize_prices(
            data,
            target_sharpe=target_sharpe,
            min_margin=min_margin
        )

        # Prepare response
        results = optimized[[
            'item_name', 'current_price', 'optimal_price',
            'price_change', 'price_change_pct'
        ]].to_dict('records')

        return jsonify({
            'success': True,
            'optimized_items': results,
            'avg_price_change': round(optimized['price_change_pct'].mean(), 2)
        })

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics')
def get_metrics():
    """Get current model metrics."""
    if model is None:
        return jsonify({'error': 'No model trained'}), 404

    return jsonify(model.get_metrics())


@app.route('/api/recommendations')
def get_recommendations():
    """Get item recommendations."""
    if not last_results:
        return jsonify({'error': 'No analysis results available'}), 404

    return jsonify({
        'recommendations': last_results.get('recommendations', {}),
        'portfolio': last_results.get('portfolio', {})
    })


@app.route('/api/sample-data')
def get_sample_data():
    """Generate and return sample data."""
    try:
        sample_data = DataLoader.generate_sample_data(n_samples=50)
        return jsonify({
            'data': sample_data.to_dict('records'),
            'columns': list(sample_data.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_trained': model is not None and model.is_trained
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting Menu Portfolio Optimizer on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
