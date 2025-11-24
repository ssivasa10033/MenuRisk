"""
FastAPI application for MenuRisk API.

Provides endpoints for menu optimization and risk analysis.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Pydantic models for API
class MenuItemInput(BaseModel):
    """Input schema for menu item."""

    item_name: str
    current_price: float = Field(..., gt=0)
    cogs: float = Field(..., gt=0)
    quantity_sold: int = Field(..., ge=0)
    category: Optional[str] = None
    season: Optional[str] = None
    province: Optional[str] = None


class PredictionOutput(BaseModel):
    """Output schema for predictions."""

    item_name: str
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float
    recommendation: str
    sharpe_ratio: Optional[float]
    expected_return: Optional[float]
    risk: Optional[float]


class PortfolioMetricsOutput(BaseModel):
    """Portfolio-level metrics."""

    total_items: int
    keep_count: int
    monitor_count: int
    remove_count: int
    portfolio_sharpe: Optional[float]
    portfolio_return: Optional[float]
    portfolio_risk: Optional[float]


class UploadResponse(BaseModel):
    """Response for file upload."""

    message: str
    rows: int
    items: int
    warnings: List[str] = []


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str


class TrainingResponse(BaseModel):
    """Response for model training."""

    message: str
    r2_score: float
    mae: float
    rmse: float
    cv_r2_mean: Optional[float]
    cv_r2_std: Optional[float]


# Global state
class AppState:
    """Application state for storing model and data."""

    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.last_data = None
        self.last_results = None
        self.is_model_trained = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    # Startup
    logger.info("Starting MenuRisk API...")

    # Create directories
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.model_dir, exist_ok=True)
    os.makedirs(settings.charts_dir, exist_ok=True)

    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    yield

    # Shutdown
    logger.info("Shutting down MenuRisk API...")


# Initialize FastAPI
app = FastAPI(
    title="MenuRisk API",
    description="Menu portfolio optimization using ML and quantitative finance",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload CSV data for analysis.

    Returns validation results.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV")

    try:
        contents = await file.read()

        # Check file size
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
            )

        import io
        df = pd.read_csv(io.BytesIO(contents))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV parsing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    # Validate data
    from src.data.validator import DataValidator

    validator = DataValidator(min_observations=settings.min_observations)
    result = validator.validate_csv(df)

    if not result.is_valid:
        raise HTTPException(
            status_code=422,
            detail={"message": "Validation failed", "errors": result.errors},
        )

    # Clean and store data
    state.last_data = validator.clean_dataframe(df)

    return UploadResponse(
        message="Upload successful",
        rows=len(state.last_data),
        items=state.last_data['item_name'].nunique(),
        warnings=result.warnings,
    )


@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """
    Train the demand forecasting model on uploaded data.
    """
    if state.last_data is None:
        raise HTTPException(
            status_code=400,
            detail="No data available. Please upload data first.",
        )

    try:
        from src.data.feature_engineer import TimeSeriesFeatureEngineer
        from src.models.demand_forecaster import DemandForecaster

        df = state.last_data

        # Feature engineering
        if 'date' in df.columns:
            # Time-series split
            df['date'] = pd.to_datetime(df['date'])
            cutoff = df['date'].max() - pd.Timedelta(days=settings.test_size_days)
            cutoff_str = cutoff.strftime('%Y-%m-%d')

            state.feature_engineer = TimeSeriesFeatureEngineer()
            state.feature_engineer.fit(df, cutoff_str)
            features_df = state.feature_engineer.transform(df)
        else:
            # Non-time-series data
            from src.data.preprocessor import DataPreprocessor

            preprocessor = DataPreprocessor()
            X, y, feature_names = preprocessor.prepare_features(df)

            state.model = DemandForecaster(tune_hyperparams=settings.tune_hyperparams)
            metrics = state.model.train(X, y, feature_names=feature_names)
            state.is_model_trained = True

            return TrainingResponse(
                message="Model trained successfully",
                r2_score=metrics['r2_score'],
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                cv_r2_mean=metrics.get('cv_r2_mean'),
                cv_r2_std=metrics.get('cv_r2_std'),
            )

        # Prepare features for time-series
        exclude_cols = ['date', 'item_name', 'quantity_sold', 'profit_margin']
        feature_cols = [
            col for col in features_df.columns
            if col not in exclude_cols
        ]
        numeric_cols = features_df[feature_cols].select_dtypes(
            include=['number']
        ).columns.tolist()

        X = features_df[numeric_cols].dropna()
        y = features_df.loc[X.index, 'quantity_sold']

        # Train model
        state.model = DemandForecaster(tune_hyperparams=settings.tune_hyperparams)
        metrics = state.model.train(X.values, y.values, feature_names=numeric_cols)
        state.is_model_trained = True

        return TrainingResponse(
            message="Model trained successfully",
            r2_score=metrics['r2_score'],
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            cv_r2_mean=metrics.get('cv_r2_mean'),
            cv_r2_std=metrics.get('cv_r2_std'),
        )

    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/metrics", response_model=Dict)
async def get_model_metrics():
    """Get current model metrics."""
    if not state.is_model_trained or state.model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please train the model first.",
        )

    return state.model.get_metrics()


@app.get("/portfolio", response_model=Dict)
async def get_portfolio_metrics():
    """Get portfolio-level metrics."""
    if state.last_data is None:
        raise HTTPException(
            status_code=400,
            detail="No data available. Please upload data first.",
        )

    try:
        from src.finance.risk_metrics_v2 import PortfolioMetrics, RiskMetrics

        df = state.last_data.copy()

        # Ensure we have revenue and cogs
        if 'revenue' not in df.columns:
            df['revenue'] = df['current_price'] * df['quantity_sold']
        if 'total_cogs' not in df.columns:
            df['total_cogs'] = df['cogs'] * df['quantity_sold']

        risk_metrics = RiskMetrics(risk_free_rate=settings.risk_free_rate)
        portfolio = PortfolioMetrics(risk_metrics)

        summary = portfolio.get_portfolio_summary(
            df,
            item_col='item_name',
            date_col='date' if 'date' in df.columns else None,
            revenue_col='revenue',
            cogs_col='total_cogs',
        )

        return summary

    except Exception as e:
        logger.error(f"Portfolio metrics error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate portfolio metrics: {str(e)}",
        )


@app.get("/recommendations", response_model=Dict[str, str])
async def get_recommendations():
    """Get item recommendations (KEEP/MONITOR/REMOVE)."""
    if state.last_data is None:
        raise HTTPException(
            status_code=400,
            detail="No data available. Please upload data first.",
        )

    try:
        from src.finance.risk_metrics_v2 import RiskMetrics

        df = state.last_data.copy()

        # Ensure we have revenue and cogs
        if 'revenue' not in df.columns:
            df['revenue'] = df['current_price'] * df['quantity_sold']
        if 'total_cogs' not in df.columns:
            df['total_cogs'] = df['cogs'] * df['quantity_sold']

        risk_metrics = RiskMetrics(risk_free_rate=settings.risk_free_rate)

        # Calculate per-item metrics
        item_metrics = risk_metrics.calculate_all_metrics(
            df,
            item_col='item_name',
            revenue_col='revenue',
            cogs_col='total_cogs',
        )

        # Get recommendations
        recommendations = risk_metrics.get_recommendations(
            item_metrics,
            keep_threshold=settings.sharpe_keep_threshold,
            monitor_threshold=settings.sharpe_monitor_threshold,
        )

        return recommendations

    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}",
        )


@app.post("/predict", response_model=List[PredictionOutput])
async def predict_demand(items: List[MenuItemInput]):
    """Predict demand for menu items."""
    if not state.is_model_trained or state.model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please train the model first.",
        )

    try:
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in items])

        # Get predictions with intervals
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        X, _, _ = preprocessor.prepare_features(df, fit_scaler=True)

        predictions, lower, upper = state.model.get_prediction_intervals(X)

        results = []
        for i, item in enumerate(items):
            results.append(
                PredictionOutput(
                    item_name=item.item_name,
                    predicted_demand=float(predictions[i]),
                    confidence_lower=float(lower[i]),
                    confidence_upper=float(upper[i]),
                    recommendation="UNKNOWN",  # Would need risk metrics
                    sharpe_ratio=None,
                    expected_return=None,
                    risk=None,
                )
            )

        return results

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/sample-data")
async def get_sample_data():
    """Generate sample data for testing."""
    from src.data.loader import DataLoader

    loader = DataLoader()
    sample_df = loader.generate_sample_data(n_items=10, n_days=90)

    return {
        "message": "Sample data generated",
        "rows": len(sample_df),
        "data": sample_df.to_dict(orient='records')[:10],  # First 10 rows
    }


# Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )
