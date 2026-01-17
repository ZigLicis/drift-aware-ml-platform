"""MLflow model registry utilities.

This module provides the ModelRegistry class for managing models in the
MLflow model registry, including version comparison and stage management.

Example:
    >>> from src.mlflow_utils.registry import ModelRegistry
    >>> registry = ModelRegistry("http://localhost:5001")
    >>> model = registry.get_latest_model("weather-model", stage="Production")
    >>> info = registry.get_model_info("weather-model")
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found in the registry."""

    pass


class ModelRegistry:
    """Manage models in the MLflow model registry.

    Provides methods to load models, compare versions, and query model metadata.

    Attributes:
        tracking_uri: MLflow tracking server URI.
        client: MLflow tracking client.

    Example:
        >>> registry = ModelRegistry("http://localhost:5001")
        >>> model = registry.get_latest_model("weather-model", stage="Production")
        >>> if model:
        ...     predictions = model.predict(X_new)
    """

    def __init__(self, tracking_uri: str) -> None:
        """Initialize the model registry client.

        Args:
            tracking_uri: MLflow tracking server URI.
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"ModelRegistry initialized: uri={tracking_uri}")

    def get_latest_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Any | None:
        """Load the latest model version from a specific stage.

        Args:
            model_name: Registered model name.
            stage: Stage to load from ("Production", "Staging", "None", "Archived").

        Returns:
            Loaded model object, or None if no model exists at that stage.

        Example:
            >>> model = registry.get_latest_model("weather-model", "Production")
            >>> if model:
            ...     predictions = model.predict(X_new)
        """
        try:
            # Search for model versions at the specified stage
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'"
            )

            # Filter by stage
            stage_versions = [v for v in versions if v.current_stage == stage]

            if not stage_versions:
                logger.info(
                    f"No model '{model_name}' found in stage '{stage}'"
                )
                return None

            # Get the latest version in this stage
            latest = max(stage_versions, key=lambda v: int(v.version))
            model_uri = f"models:/{model_name}/{latest.version}"

            logger.info(
                f"Loading model '{model_name}' v{latest.version} from {stage}"
            )
            return mlflow.sklearn.load_model(model_uri)

        except MlflowException as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return None

    def get_model_by_version(
        self,
        model_name: str,
        version: int | str,
    ) -> Any:
        """Load a specific model version.

        Args:
            model_name: Registered model name.
            version: Version number to load.

        Returns:
            Loaded model object.

        Raises:
            ModelNotFoundError: If model version does not exist.
        """
        try:
            model_uri = f"models:/{model_name}/{version}"
            logger.info(f"Loading model '{model_name}' v{version}")
            return mlflow.sklearn.load_model(model_uri)
        except MlflowException as e:
            raise ModelNotFoundError(
                f"Model '{model_name}' v{version} not found: {e}"
            ) from e

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get comprehensive information about a registered model.

        Returns metadata for all versions including run metrics.

        Args:
            model_name: Registered model name.

        Returns:
            Dictionary containing:
            - name: Model name
            - description: Model description
            - versions: List of version info with stage, metrics, etc.

        Raises:
            ModelNotFoundError: If model does not exist.

        Example:
            >>> info = registry.get_model_info("weather-model")
            >>> for v in info["versions"]:
            ...     print(f"v{v['version']}: {v['stage']} - RMSE: {v['metrics'].get('rmse')}")
        """
        try:
            # Get registered model
            model = self.client.get_registered_model(model_name)

            # Get all versions
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'"
            )

            version_info = []
            for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
                v_info = {
                    "version": int(v.version),
                    "stage": v.current_stage,
                    "status": v.status,
                    "creation_timestamp": v.creation_timestamp,
                    "description": v.description,
                    "run_id": v.run_id,
                    "metrics": {},
                    "params": {},
                }

                # Get run metrics if available
                if v.run_id:
                    try:
                        run = self.client.get_run(v.run_id)
                        v_info["metrics"] = run.data.metrics
                        v_info["params"] = run.data.params
                    except MlflowException:
                        logger.debug(f"Could not fetch run {v.run_id} for version {v.version}")

                version_info.append(v_info)

            return {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "tags": dict(model.tags) if model.tags else {},
                "versions": version_info,
                "latest_versions": {
                    lv.current_stage: int(lv.version)
                    for lv in model.latest_versions
                },
            }

        except MlflowException as e:
            raise ModelNotFoundError(f"Model '{model_name}' not found: {e}") from e

    def compare_models(
        self,
        model_name: str,
        version_a: int | str,
        version_b: int | str,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare metrics between two model versions.

        Args:
            model_name: Registered model name.
            version_a: First version to compare.
            version_b: Second version to compare.
            metrics: Specific metrics to compare. If None, compares all common metrics.

        Returns:
            Dictionary containing:
            - version_a: Metrics for version A
            - version_b: Metrics for version B
            - comparison: Difference (B - A) for each metric
            - better_version: Which version is better for each metric (lower is better)

        Example:
            >>> comparison = registry.compare_models("weather-model", 1, 2)
            >>> print(f"RMSE improved by: {comparison['comparison']['rmse']}")
        """
        version_a = str(version_a)
        version_b = str(version_b)

        # Get version info
        try:
            v_a = self.client.get_model_version(model_name, version_a)
            v_b = self.client.get_model_version(model_name, version_b)
        except MlflowException as e:
            raise ModelNotFoundError(f"Could not find model versions: {e}") from e

        # Get run metrics
        metrics_a = {}
        metrics_b = {}

        if v_a.run_id:
            try:
                run_a = self.client.get_run(v_a.run_id)
                metrics_a = run_a.data.metrics
            except MlflowException:
                logger.warning(f"Could not fetch metrics for version {version_a}")

        if v_b.run_id:
            try:
                run_b = self.client.get_run(v_b.run_id)
                metrics_b = run_b.data.metrics
            except MlflowException:
                logger.warning(f"Could not fetch metrics for version {version_b}")

        # Filter to specified metrics or common metrics
        if metrics:
            common_metrics = [m for m in metrics if m in metrics_a and m in metrics_b]
        else:
            common_metrics = list(set(metrics_a.keys()) & set(metrics_b.keys()))

        # Compute comparison
        comparison = {}
        better_version = {}

        # Metrics where lower is better (errors)
        lower_is_better = {"rmse", "mse", "mae", "mape", "loss"}

        for metric in common_metrics:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)

            if val_a is not None and val_b is not None:
                diff = val_b - val_a
                comparison[metric] = diff

                # Determine which version is better
                if metric.lower() in lower_is_better or "error" in metric.lower():
                    better_version[metric] = version_b if diff < 0 else version_a
                else:  # Higher is better (r2, accuracy, etc.)
                    better_version[metric] = version_b if diff > 0 else version_a

        return {
            "model_name": model_name,
            "version_a": {
                "version": int(version_a),
                "stage": v_a.current_stage,
                "metrics": {m: metrics_a.get(m) for m in common_metrics},
            },
            "version_b": {
                "version": int(version_b),
                "stage": v_b.current_stage,
                "metrics": {m: metrics_b.get(m) for m in common_metrics},
            },
            "comparison": comparison,
            "better_version": better_version,
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models.

        Returns:
            List of model summaries with name, versions count, and latest versions.
        """
        models = []
        for rm in self.client.search_registered_models():
            models.append(
                {
                    "name": rm.name,
                    "description": rm.description,
                    "n_versions": len(rm.latest_versions),
                    "latest_versions": {
                        lv.current_stage: int(lv.version)
                        for lv in rm.latest_versions
                    },
                }
            )
        return models

    def delete_model_version(
        self,
        model_name: str,
        version: int | str,
    ) -> None:
        """Delete a specific model version.

        Args:
            model_name: Registered model name.
            version: Version to delete.
        """
        try:
            self.client.delete_model_version(model_name, str(version))
            logger.info(f"Deleted {model_name} v{version}")
        except MlflowException as e:
            logger.error(f"Failed to delete model version: {e}")
            raise
