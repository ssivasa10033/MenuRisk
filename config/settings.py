"""
Application settings and configuration management.

Uses Pydantic for validation and environment variable loading.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import os
from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Base configuration settings.

    Settings are loaded from environment variables and .env file.
    """

    # Environment
    environment: Literal["development", "production", "testing"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Application
    app_name: str = Field(default="MenuRisk", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")

    # Financial parameters (Canadian market)
    risk_free_rate: float = Field(
        default=0.0225, description="Bank of Canada overnight rate (Oct 2025)"
    )
    sharpe_keep_threshold: float = Field(
        default=1.5, description="Sharpe ratio threshold for KEEP recommendation"
    )
    sharpe_monitor_threshold: float = Field(
        default=0.8, description="Sharpe ratio threshold for MONITOR recommendation"
    )

    # Model parameters
    model_n_estimators: int = Field(
        default=300, description="Number of trees in Random Forest"
    )
    model_max_depth: Optional[int] = Field(
        default=None, description="Maximum tree depth (None for unlimited)"
    )
    model_random_state: int = Field(
        default=42, description="Random seed for reproducibility"
    )
    tune_hyperparams: bool = Field(
        default=True, description="Whether to tune hyperparameters"
    )

    # Data parameters
    min_observations: int = Field(
        default=30, description="Minimum observations per item"
    )
    test_size_days: int = Field(default=30, description="Number of days for test set")
    cv_splits: int = Field(default=5, description="Number of cross-validation splits")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")

    # File upload
    max_upload_size_mb: float = Field(
        default=16.0, description="Maximum upload file size in MB"
    )
    upload_dir: str = Field(
        default="uploads", description="Directory for uploaded files"
    )

    # Model storage
    model_dir: str = Field(default="models", description="Directory for saved models")
    charts_dir: str = Field(
        default="static/charts", description="Directory for generated charts"
    )

    # MLflow (optional)
    mlflow_enabled: bool = Field(default=False, description="Enable MLflow tracking")
    mlflow_tracking_uri: str = Field(
        default="./mlruns", description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="menurisk-demand-forecasting", description="MLflow experiment name"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "text"] = Field(
        default="text", description="Log format"
    )

    # Security
    secret_key: str = Field(
        default="change-me-in-production", description="Secret key for sessions"
    )
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5000"],
        description="Allowed CORS origins",
    )

    # Canadian-specific
    default_province: str = Field(default="ON", description="Default province code")
    default_season: str = Field(default="Summer", description="Default season")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class DevelopmentSettings(Settings):
    """Development environment settings."""

    environment: Literal["development"] = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    tune_hyperparams: bool = False  # Faster iteration in dev


class ProductionSettings(Settings):
    """Production environment settings."""

    environment: Literal["production"] = "production"
    debug: bool = False
    api_workers: int = 8
    log_level: str = "WARNING"
    log_format: Literal["json", "text"] = "json"
    mlflow_enabled: bool = True


class TestingSettings(Settings):
    """Testing environment settings."""

    environment: Literal["testing"] = "testing"
    debug: bool = True
    log_level: str = "DEBUG"
    tune_hyperparams: bool = False
    min_observations: int = 5  # Lower for testing


@lru_cache()
def get_settings() -> Settings:
    """
    Get settings based on environment.

    Uses lru_cache to ensure settings are loaded only once.

    Returns:
        Settings instance for current environment
    """
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()
