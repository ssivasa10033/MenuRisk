"""
Model registry using MLflow for versioning and tracking.

Manages model versions, parameters, and metrics.

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manage model versions using MLflow.

    This class provides a unified interface for:
    - Logging model training runs
    - Tracking parameters and metrics
    - Model versioning and staging
    - Model comparison and promotion
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "menurisk-demand-forecasting",
    ):
        """
        Initialize the model registry.

        Args:
            tracking_uri: MLflow tracking server URI or local directory
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow_available = False
        self._client = None

        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow if available."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

            self._client = MlflowClient()
            self._mlflow_available = True

            logger.info(f"MLflow configured with tracking URI: {self.tracking_uri}")
            logger.info(f"Experiment: {self.experiment_name}")

        except ImportError:
            logger.warning(
                "MLflow not installed. Model registry features disabled. "
                "Install with: pip install mlflow"
            )
            self._mlflow_available = False

    @property
    def is_available(self) -> bool:
        """Check if MLflow is available."""
        return self._mlflow_available

    def log_model(
        self,
        model: Any,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Log model to MLflow with parameters and metrics.

        Args:
            model: Trained model object
            model_name: Name for the registered model
            params: Training parameters
            metrics: Performance metrics
            artifacts: Additional files to log {name: path}
            tags: Tags for the run

        Returns:
            Run ID if successful, None otherwise
        """
        if not self._mlflow_available:
            logger.warning("MLflow not available. Skipping model logging.")
            return None

        try:
            import mlflow
            import mlflow.sklearn

            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )

                # Log additional artifacts
                if artifacts:
                    for name, path in artifacts.items():
                        if os.path.exists(path):
                            mlflow.log_artifact(path, artifact_path=name)

                # Set tags
                default_tags = {
                    "model_type": "RandomForest",
                    "framework": "scikit-learn",
                }
                if tags:
                    default_tags.update(tags)

                for key, value in default_tags.items():
                    mlflow.set_tag(key, value)

                logger.info(f"Model logged with run_id: {run.info.run_id}")
                return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            return None

    def load_model(
        self,
        model_name: str,
        version: str = "latest",
    ) -> Optional[Any]:
        """
        Load model from registry.

        Args:
            model_name: Registered model name
            version: Version number or "latest"

        Returns:
            Loaded model or None if not found
        """
        if not self._mlflow_available:
            logger.warning("MLflow not available. Cannot load model.")
            return None

        try:
            import mlflow.sklearn

            if version == "latest":
                model_uri = f"models:/{model_name}/Production"
            else:
                model_uri = f"models:/{model_name}/{version}"

            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded from: {model_uri}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None

    def promote_to_production(
        self,
        model_name: str,
        version: str,
    ) -> bool:
        """
        Promote model version to production.

        Args:
            model_name: Registered model name
            version: Version to promote

        Returns:
            True if successful
        """
        if not self._mlflow_available or self._client is None:
            logger.warning("MLflow not available. Cannot promote model.")
            return False

        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"Model {model_name} v{version} promoted to Production")
            return True

        except Exception as e:
            logger.error(f"Failed to promote model: {str(e)}")
            return False

    def get_model_versions(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Get all versions of a registered model.

        Args:
            model_name: Registered model name

        Returns:
            DataFrame with version info
        """
        if not self._mlflow_available or self._client is None:
            return None

        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")

            data = []
            for v in versions:
                data.append(
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "creation_timestamp": v.creation_timestamp,
                        "run_id": v.run_id,
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Failed to get model versions: {str(e)}")
            return None

    def compare_runs(self, run_ids: list) -> Optional[pd.DataFrame]:
        """
        Compare multiple model runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with metrics comparison
        """
        if not self._mlflow_available or self._client is None:
            return None

        try:
            data = []
            for run_id in run_ids:
                run = self._client.get_run(run_id)
                data.append(
                    {
                        "run_id": run_id,
                        "start_time": run.info.start_time,
                        **run.data.params,
                        **run.data.metrics,
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Failed to compare runs: {str(e)}")
            return None

    def get_best_run(
        self,
        metric: str = "test_r2",
        ascending: bool = False,
    ) -> Optional[Dict]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize
            ascending: If True, lower is better

        Returns:
            Dict with best run info
        """
        if not self._mlflow_available:
            return None

        try:
            import mlflow

            order = "ASC" if ascending else "DESC"
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=[f"metrics.{metric} {order}"],
                max_results=1,
            )

            if len(runs) > 0:
                best = runs.iloc[0]
                return {
                    "run_id": best["run_id"],
                    metric: best.get(f"metrics.{metric}"),
                    "params": {
                        k.replace("params.", ""): v
                        for k, v in best.items()
                        if k.startswith("params.")
                    },
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get best run: {str(e)}")
            return None


def log_training_run(
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model_name: str = "demand_forecaster",
    tracking_uri: str = "./mlruns",
) -> Optional[str]:
    """
    Convenience function to log a training run.

    Args:
        model: Trained model
        params: Training parameters
        metrics: Performance metrics
        model_name: Name for registered model
        tracking_uri: MLflow tracking URI

    Returns:
        Run ID if successful
    """
    registry = ModelRegistry(tracking_uri=tracking_uri)
    return registry.log_model(model, model_name, params, metrics)
