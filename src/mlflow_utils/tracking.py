"""MLflow experiment tracking utilities.

This module provides the ExperimentTracker class for managing MLflow experiments,
logging datasets, models, and evaluation artifacts.

Example:
    >>> from src.mlflow_utils.tracking import ExperimentTracker
    >>> tracker = ExperimentTracker(
    ...     tracking_uri="http://localhost:5001",
    ...     experiment_name="weather-prediction"
    ... )
    >>> with tracker.start_run("baseline-model"):
    ...     tracker.log_dataset_info(df, "training_data")
    ...     model.fit(X_train, y_train)
    ...     tracker.log_model_with_signature(model, X_train, "weather-model")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowConnectionError(Exception):
    """Raised when MLflow server connection fails."""

    pass


class ExperimentTracker:
    """Manage MLflow experiment tracking with automatic tagging and artifact logging.

    This class provides a high-level interface for MLflow experiment tracking,
    including automatic tagging, dataset profiling, model signature inference,
    and evaluation visualization.

    Attributes:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Name of the MLflow experiment.
        experiment_id: MLflow experiment ID.
        client: MLflow tracking client.

    Example:
        >>> tracker = ExperimentTracker(
        ...     tracking_uri="http://localhost:5001",
        ...     experiment_name="weather-prediction"
        ... )
        >>> with tracker.start_run("baseline-ridge"):
        ...     tracker.log_dataset_info(train_df, "training_data")
        ...     mlflow.log_params({"alpha": 1.0})
        ...     model.fit(X_train, y_train)
        ...     tracker.log_evaluation_plots(y_test, y_pred)
        ...     tracker.log_feature_importance(model, feature_names)
        ...     tracker.log_model_with_signature(model, X_train, "weather-model")
    """

    DEFAULT_TAGS = {
        "project": "domain-shift-ml-platform",
        "framework": "scikit-learn",
    }

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        environment: str = "development",
    ) -> None:
        """Initialize the experiment tracker.

        Args:
            tracking_uri: MLflow tracking server URI (e.g., "http://localhost:5001").
            experiment_name: Name for the MLflow experiment.
            environment: Environment tag (development, staging, production).

        Raises:
            MLflowConnectionError: If connection to MLflow server fails.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.environment = environment
        self._active_run = None

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Verify connection
        self._verify_connection()

        # Create MLflow client
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment()

        logger.info(
            f"ExperimentTracker initialized: "
            f"uri={tracking_uri}, experiment={experiment_name}, "
            f"experiment_id={self.experiment_id}"
        )

    def _verify_connection(self) -> None:
        """Verify connection to MLflow server.

        Raises:
            MLflowConnectionError: If server is unreachable.
        """
        try:
            # Attempt to list experiments as a connection test
            mlflow.search_experiments(max_results=1)
            logger.debug(f"Successfully connected to MLflow at {self.tracking_uri}")
        except MlflowException as e:
            raise MLflowConnectionError(
                f"Failed to connect to MLflow server at {self.tracking_uri}: {e}"
            ) from e
        except Exception as e:
            raise MLflowConnectionError(
                f"Unexpected error connecting to MLflow at {self.tracking_uri}: {e}"
            ) from e

    def _get_or_create_experiment(self) -> str:
        """Get existing experiment or create new one.

        Returns:
            Experiment ID.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                tags={
                    **self.DEFAULT_TAGS,
                    "environment": self.environment,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.debug(f"Using existing experiment: {self.experiment_name}")

        return experiment_id

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash if available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # Short hash
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Start an MLflow run as a context manager.

        Automatically adds tags for timestamp, git commit, user, and environment.

        Args:
            run_name: Name for the run (displayed in MLflow UI).
            tags: Additional tags to add to the run.
            nested: Whether this is a nested run.

        Yields:
            MLflow ActiveRun context.

        Example:
            >>> with tracker.start_run("baseline-model", tags={"model_type": "ridge"}):
            ...     mlflow.log_param("alpha", 1.0)
            ...     model.fit(X, y)
        """
        # Build run tags
        run_tags = {
            **self.DEFAULT_TAGS,
            "environment": self.environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": os.getenv("USER", "unknown"),
        }

        # Add git commit if available
        git_commit = self._get_git_commit()
        if git_commit:
            run_tags["git_commit"] = git_commit

        # Add custom tags
        if tags:
            run_tags.update(tags)

        try:
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=run_tags,
                nested=nested,
            ) as run:
                self._active_run = run
                logger.info(f"Started MLflow run: {run_name} (id={run.info.run_id[:8]})")
                yield run
        except MlflowException as e:
            logger.error(f"MLflow run failed: {e}")
            raise
        finally:
            self._active_run = None
            logger.info(f"Completed MLflow run: {run_name}")

    def log_dataset_info(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        save_profile: bool = True,
    ) -> None:
        """Log dataset statistics and profile as MLflow artifacts.

        Logs the following:
        - Parameters: n_samples, n_features, date_range (if timestamp present)
        - Metrics: Feature statistics (mean, std, min, max) for numeric columns
        - Artifacts: Dataset profile JSON (if save_profile=True)

        Args:
            df: DataFrame to profile.
            dataset_name: Name for the dataset (used in param/metric names).
            save_profile: Whether to save full profile as JSON artifact.

        Example:
            >>> tracker.log_dataset_info(train_df, "training_data")
        """
        if self._active_run is None:
            raise RuntimeError("No active MLflow run. Use within start_run() context.")

        prefix = f"{dataset_name}_"

        # Basic info as params
        mlflow.log_params(
            {
                f"{prefix}n_samples": len(df),
                f"{prefix}n_features": len(df.columns),
            }
        )

        # Date range if timestamp column exists
        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"])
            mlflow.log_params(
                {
                    f"{prefix}date_start": str(timestamps.min().date()),
                    f"{prefix}date_end": str(timestamps.max().date()),
                }
            )

        # Feature distributions for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        profile_data = {"dataset_name": dataset_name, "n_samples": len(df), "features": {}}

        for col in numeric_cols:
            col_stats = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "null_count": int(df[col].isna().sum()),
            }
            profile_data["features"][col] = col_stats

            # Log key metrics (limit to avoid too many metrics)
            safe_col = col.replace("_", "-")[:20]  # Sanitize metric name
            mlflow.log_metrics(
                {
                    f"{prefix}{safe_col}_mean": col_stats["mean"],
                    f"{prefix}{safe_col}_std": col_stats["std"],
                }
            )

        # Save full profile as artifact
        if save_profile:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(profile_data, f, indent=2, default=str)
                temp_path = f.name

            try:
                mlflow.log_artifact(temp_path, artifact_path="dataset_profiles")
                logger.info(f"Logged dataset profile for {dataset_name}")
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def log_model_with_signature(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        model_name: str,
        register: bool = True,
    ) -> str | None:
        """Log model with inferred signature and optionally register.

        Args:
            model: Trained model object (sklearn-compatible).
            X_sample: Sample input data for signature inference.
            model_name: Name for the model (used in registry).
            register: Whether to register model in MLflow registry.

        Returns:
            Model version if registered, None otherwise.

        Example:
            >>> tracker.log_model_with_signature(
            ...     model, X_train.head(100), "weather-predictor"
            ... )
        """
        if self._active_run is None:
            raise RuntimeError("No active MLflow run. Use within start_run() context.")

        # Infer signature from sample data
        try:
            y_sample = model.predict(X_sample)
            signature = infer_signature(X_sample, y_sample)
            logger.debug(f"Inferred model signature: {signature}")
        except Exception as e:
            logger.warning(f"Could not infer signature: {e}")
            signature = None

        # Log model
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name if register else None,
        )

        logger.info(f"Logged model: {model_name}")

        if register:
            # Get the registered version
            try:
                versions = self.client.search_model_versions(f"name='{model_name}'")
                if versions:
                    latest_version = max(v.version for v in versions)
                    logger.info(f"Registered model {model_name} version {latest_version}")
                    return str(latest_version)
            except MlflowException as e:
                logger.warning(f"Could not get model version: {e}")

        return None

    def log_evaluation_plots(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        prefix: str = "",
        timestamps: pd.Series | None = None,
    ) -> None:
        """Create and log evaluation visualization plots.

        Creates the following plots:
        - Actual vs Predicted scatter plot
        - Residuals distribution histogram
        - Residuals over time (if timestamps provided)

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            prefix: Prefix for artifact filenames.
            timestamps: Optional timestamps for temporal residual plot.

        Example:
            >>> tracker.log_evaluation_plots(y_test, y_pred, prefix="test_")
        """
        if self._active_run is None:
            raise RuntimeError("No active MLflow run. Use within start_run() context.")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_true - y_pred

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Actual vs Predicted scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")

            # Add diagonal line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)

            scatter_path = tmpdir / f"{prefix}actual_vs_predicted.png"
            fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(scatter_path), artifact_path="plots")

            # 2. Residuals distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(x=0, color="r", linestyle="--", lw=2)
            ax.axvline(x=residuals.mean(), color="orange", linestyle="-", lw=2, label=f"Mean: {residuals.mean():.2f}")
            ax.set_xlabel("Residual (Actual - Predicted)")
            ax.set_ylabel("Frequency")
            ax.set_title("Residuals Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            resid_path = tmpdir / f"{prefix}residuals_distribution.png"
            fig.savefig(resid_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(resid_path), artifact_path="plots")

            # 3. Residuals over time (if timestamps available)
            if timestamps is not None:
                fig, ax = plt.subplots(figsize=(12, 6))
                timestamps = pd.to_datetime(timestamps)
                ax.scatter(timestamps, residuals, alpha=0.5, s=10)
                ax.axhline(y=0, color="r", linestyle="--", lw=2)
                ax.set_xlabel("Time")
                ax.set_ylabel("Residual")
                ax.set_title("Residuals Over Time")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

                time_path = tmpdir / f"{prefix}residuals_over_time.png"
                fig.savefig(time_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                mlflow.log_artifact(str(time_path), artifact_path="plots")

        logger.info(f"Logged evaluation plots with prefix '{prefix}'")

    def log_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int = 20,
    ) -> dict[str, float] | None:
        """Extract and log feature importance from model.

        Supports models with feature_importances_ (tree-based) or coef_ (linear).

        Args:
            model: Trained model with feature importance attributes.
            feature_names: List of feature names matching model input.
            top_n: Number of top features to display in plot.

        Returns:
            Dictionary of feature importances, or None if not available.

        Example:
            >>> importance = tracker.log_feature_importance(
            ...     model, X_train.columns.tolist()
            ... )
        """
        if self._active_run is None:
            raise RuntimeError("No active MLflow run. Use within start_run() context.")

        # Extract importance values
        importance_values = None

        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear models, use absolute coefficient values
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef.flatten()
            importance_values = np.abs(coef)

        if importance_values is None:
            logger.warning("Model does not have feature importance attributes")
            return None

        # Create importance dict
        importance_dict = dict(zip(feature_names, importance_values))

        # Sort by importance
        sorted_importance = sorted(
            importance_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Log top features as metrics
        for name, value in sorted_importance[:top_n]:
            safe_name = name.replace("_", "-")[:30]
            mlflow.log_metric(f"importance_{safe_name}", float(value))

        # Create bar chart
        with tempfile.TemporaryDirectory() as tmpdir:
            top_features = sorted_importance[:top_n]
            names = [f[0] for f in top_features]
            values = [f[1] for f in top_features]

            fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)))
            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.invert_yaxis()  # Top feature at top
            ax.set_xlabel("Importance")
            ax.set_title(f"Top {top_n} Feature Importances")
            ax.grid(True, alpha=0.3, axis="x")

            plot_path = Path(tmpdir) / "feature_importance.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

        # Save full importance as JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dict(sorted_importance), f, indent=2)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, artifact_path="feature_importance")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        logger.info(f"Logged feature importance for {len(feature_names)} features")
        return importance_dict

    def promote_model(
        self,
        model_name: str,
        version: int | str,
        stage: str,
        description: str | None = None,
    ) -> None:
        """Promote a model version to a stage.

        Args:
            model_name: Registered model name.
            version: Model version number.
            stage: Target stage ("Staging", "Production", "Archived").
            description: Optional description for the transition.

        Example:
            >>> tracker.promote_model("weather-model", 1, "Production")
        """
        version = str(version)

        # Validate stage
        valid_stages = ["Staging", "Production", "Archived", "None"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production"),
            )

            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description,
                )

            logger.info(f"Promoted {model_name} v{version} to {stage}")

        except MlflowException as e:
            logger.error(f"Failed to promote model: {e}")
            raise

    @property
    def active_run_id(self) -> str | None:
        """Get the current active run ID."""
        if self._active_run:
            return self._active_run.info.run_id
        return None
