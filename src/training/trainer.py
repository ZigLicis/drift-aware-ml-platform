"""Model training pipeline orchestration.

This module provides the ModelTrainer class that orchestrates the complete
training pipeline: data loading, feature engineering, model training,
evaluation, and MLflow logging.

Example:
    >>> from src.training.trainer import ModelTrainer
    >>> trainer = ModelTrainer(
    ...     config_path="config/model_config.yaml",
    ...     db_connection_string="postgresql://...",
    ...     mlflow_tracking_uri="http://localhost:5001"
    ... )
    >>> results = trainer.run_training_pipeline(model_type="ridge")
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.data_ingestion.storage import DataStorage
from src.mlflow_utils.registry import ModelRegistry
from src.mlflow_utils.tracking import ExperimentTracker, MLflowConnectionError
from src.training.evaluation import ModelEvaluator
from src.training.feature_engineering import FeatureEngineer
from src.training.models import BaseModel, create_model

logger = logging.getLogger(__name__)


class TrainerError(Exception):
    """Raised when training pipeline fails."""

    pass


class ModelTrainer:
    """Orchestrate the complete model training pipeline.

    Handles data loading, feature engineering, model training, evaluation,
    and experiment tracking with MLflow.

    Attributes:
        config: Loaded configuration dictionary.
        feature_engineer: Feature engineering instance.
        evaluator: Model evaluation instance.
        tracker: MLflow experiment tracker.
        storage: Database storage instance.

    Example:
        >>> trainer = ModelTrainer(
        ...     config_path="config/model_config.yaml",
        ...     db_connection_string=os.getenv("DATABASE_URL"),
        ...     mlflow_tracking_uri="http://localhost:5001"
        ... )
        >>>
        >>> # Run full pipeline
        >>> results = trainer.run_training_pipeline(
        ...     model_type="ridge",
        ...     register_model=True
        ... )
        >>>
        >>> print(f"Test RMSE: {results['test_metrics']['rmse']:.3f}")
        >>> print(f"Model version: {results.get('model_version')}")
    """

    # Minimum metrics thresholds for model registration
    DEFAULT_THRESHOLDS = {
        "rmse": 5.0,  # Max acceptable RMSE
        "r2": 0.5,    # Min acceptable R2
    }

    def __init__(
        self,
        config_path: str | Path,
        db_connection_string: str,
        mlflow_tracking_uri: str,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize the model trainer.

        Args:
            config_path: Path to model configuration YAML file.
            db_connection_string: PostgreSQL connection string.
            mlflow_tracking_uri: MLflow tracking server URI.
            experiment_name: MLflow experiment name. Uses config if not provided.

        Raises:
            TrainerError: If initialization fails.
        """
        self.config_path = Path(config_path)

        # Load configuration
        self.config = self._load_config()

        # Initialize components
        try:
            self.storage = DataStorage(db_connection_string)
            logger.info("Connected to database")
        except Exception as e:
            raise TrainerError(f"Database connection failed: {e}") from e

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer.from_yaml(config_path)

        # Initialize evaluator
        eval_metrics = self.config.get("evaluation", {}).get(
            "metrics", ["rmse", "mae", "r2", "mape"]
        )
        self.evaluator = ModelEvaluator(metrics=eval_metrics)

        # Initialize MLflow tracker
        exp_name = experiment_name or self.config.get("mlflow", {}).get(
            "experiment_name", "weather-prediction"
        )
        try:
            self.tracker = ExperimentTracker(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=exp_name,
            )
            self.registry = ModelRegistry(mlflow_tracking_uri)
            logger.info(f"Connected to MLflow at {mlflow_tracking_uri}")
        except MLflowConnectionError as e:
            raise TrainerError(f"MLflow connection failed: {e}") from e

        # Store URIs for later use
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_registry_name = self.config.get("mlflow", {}).get(
            "model_registry_name", "weather-forecaster"
        )

        logger.info(f"ModelTrainer initialized with config from {config_path}")

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise TrainerError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        logger.debug(f"Loaded config: {list(config.keys())}")
        return config

    def prepare_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        location_name: str | None = None,
    ) -> dict[str, Any]:
        """Load and prepare data for training.

        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to now.
            location_name: Optional filter by location.

        Returns:
            Dictionary containing:
            - X_train, X_val, X_test: Feature matrices
            - y_train, y_val, y_test: Target vectors
            - feature_names: List of feature names
            - split_info: Train/val/test split information
            - raw_df: Original DataFrame (for timestamps)

        Raises:
            TrainerError: If data loading or preparation fails.
        """
        # Parse dates
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            # Default to 30 days before end
            start_dt = end_dt - timedelta(days=30)

        logger.info(f"Loading data from {start_dt.date()} to {end_dt.date()}")

        # Load data from database
        try:
            df = self.storage.get_data_by_timerange(
                start=start_dt,
                end=end_dt,
                location_name=location_name,
            )
        except Exception as e:
            raise TrainerError(f"Failed to load data: {e}") from e

        if df.empty:
            raise TrainerError(
                f"No data found between {start_dt.date()} and {end_dt.date()}"
            )

        logger.info(f"Loaded {len(df)} records from database")

        # Create features
        df = self.feature_engineer.create_features(df)

        # Create target
        horizon_hours = self.config.get("model", {}).get("prediction_horizon_hours", 24)
        df = self.feature_engineer.create_target(df, horizon_hours=horizon_hours)

        # Temporal split
        training_config = self.config.get("training", {})
        test_days = training_config.get("test_days", 7)
        val_days = training_config.get("validation_days", 3)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.feature_engineer.train_test_split_temporal(
                df,
                test_days=test_days,
                validation_days=val_days,
            )

        split_info = self.feature_engineer.get_split_info()
        feature_names = X_train.columns.tolist()

        logger.info(
            f"Data prepared: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": feature_names,
            "split_info": split_info,
            "raw_df": df,
        }

    def train_model(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[BaseModel, dict[str, float], dict[str, float]]:
        """Train model and evaluate on validation set.

        Args:
            model: Model instance to train.
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            Tuple of (trained_model, train_metrics, val_metrics).
        """
        logger.info(f"Training {model.__class__.__name__}...")

        # Train
        model.fit(X_train, y_train)

        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        train_metrics = self.evaluator.evaluate(y_train, y_train_pred)

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_metrics = self.evaluator.evaluate(y_val, y_val_pred)

        logger.info(
            f"Training complete. "
            f"Train RMSE: {train_metrics.get('rmse', 0):.3f}, "
            f"Val RMSE: {val_metrics.get('rmse', 0):.3f}"
        )

        return model, train_metrics, val_metrics

    def run_training_pipeline(
        self,
        model_type: str = "ridge",
        model_params: dict[str, Any] | None = None,
        register_model: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Run the complete training pipeline.

        This is the main entry point for training. It:
        1. Prepares data with feature engineering
        2. Trains the model
        3. Evaluates on test set
        4. Logs everything to MLflow
        5. Optionally registers the model

        Args:
            model_type: Type of model ("ridge", "random_forest", "gradient_boosting").
            model_params: Optional model hyperparameters.
            register_model: Whether to register model if metrics pass threshold.
            start_date: Data start date (YYYY-MM-DD).
            end_date: Data end date (YYYY-MM-DD).

        Returns:
            Dictionary containing:
            - model: Trained model
            - train_metrics: Training set metrics
            - val_metrics: Validation set metrics
            - test_metrics: Test set metrics
            - feature_importance: Feature importance dict
            - model_version: Registered model version (if registered)
            - run_id: MLflow run ID
            - evaluation_report: Full evaluation report

        Example:
            >>> results = trainer.run_training_pipeline(
            ...     model_type="ridge",
            ...     model_params={"alpha": 0.5},
            ...     register_model=True
            ... )
        """
        # Get model params from config or use provided
        if model_params is None:
            baseline_config = self.config.get("baseline_model", {})
            if model_type == baseline_config.get("type", "ridge"):
                model_params = {
                    k: v for k, v in baseline_config.items()
                    if k not in ["type", "description"]
                }
            else:
                model_params = {}

        # Create run name
        run_name = f"{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        logger.info(f"Starting training pipeline: {run_name}")

        # Prepare data
        data = self.prepare_data(start_date=start_date, end_date=end_date)

        # Create model
        model = create_model(model_type, **model_params)

        # Start MLflow run
        with self.tracker.start_run(
            run_name=run_name,
            tags={"model_type": model_type},
        ) as run:
            run_id = run.info.run_id

            # Log data info
            self.tracker.log_dataset_info(
                data["X_train"],
                "training_data",
            )

            # Log model params
            mlflow.log_params(model.get_params())
            mlflow.log_params({
                "test_days": self.config.get("training", {}).get("test_days", 7),
                "validation_days": self.config.get("training", {}).get("validation_days", 3),
                "prediction_horizon_hours": self.config.get("model", {}).get("prediction_horizon_hours", 24),
            })

            # Log split info
            if data["split_info"]:
                mlflow.log_params({
                    f"split_{k}": v for k, v in data["split_info"].items()
                    if isinstance(v, (str, int, float))
                })

            # Train model
            model, train_metrics, val_metrics = self.train_model(
                model,
                data["X_train"],
                data["y_train"],
                data["X_val"],
                data["y_val"],
            )

            # Evaluate on test set
            y_test_pred = model.predict(data["X_test"])
            test_metrics = self.evaluator.evaluate(data["y_test"], y_test_pred)

            # Log all metrics
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            logger.info(
                f"Test metrics: "
                f"RMSE={test_metrics.get('rmse', 0):.3f}, "
                f"MAE={test_metrics.get('mae', 0):.3f}, "
                f"R2={test_metrics.get('r2', 0):.3f}"
            )

            # Get feature importance
            feature_importance = model.get_feature_importance(data["feature_names"])
            if feature_importance:
                self.tracker.log_feature_importance(model, data["feature_names"])

            # Create evaluation plots
            self.tracker.log_evaluation_plots(
                data["y_test"],
                y_test_pred,
                prefix="test_",
            )

            # Create full evaluation report
            evaluation_report = self.evaluator.create_evaluation_report(
                data["y_test"],
                y_test_pred,
                data["feature_names"],
                model,
            )

            # Save plots as artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Predictions plot
                fig = self.evaluator.plot_predictions(
                    data["y_test"],
                    y_test_pred,
                    title=f"{model_type.title()} Model Predictions",
                )
                fig.savefig(f"{tmpdir}/predictions.png", dpi=150, bbox_inches="tight")
                mlflow.log_artifact(f"{tmpdir}/predictions.png", "plots")

                # Residuals plot
                fig = self.evaluator.plot_residuals(data["y_test"], y_test_pred)
                fig.savefig(f"{tmpdir}/residuals.png", dpi=150, bbox_inches="tight")
                mlflow.log_artifact(f"{tmpdir}/residuals.png", "plots")

                # Feature importance plot
                if feature_importance:
                    fig = self.evaluator.plot_feature_importance(feature_importance)
                    fig.savefig(f"{tmpdir}/feature_importance.png", dpi=150, bbox_inches="tight")
                    mlflow.log_artifact(f"{tmpdir}/feature_importance.png", "plots")

            # Save feature config
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                self.feature_engineer.save_feature_config(f.name)
                mlflow.log_artifact(f.name, "config")

            # Register model if metrics pass threshold
            model_version = None
            should_register = register_model and self._meets_threshold(test_metrics)

            if should_register:
                # Check if better than production
                is_better = self.compare_with_production(test_metrics)

                if is_better:
                    model_version = self.tracker.log_model_with_signature(
                        model.model,  # Get underlying sklearn model
                        data["X_train"].head(100),
                        self.model_registry_name,
                        register=True,
                    )
                    logger.info(f"Registered model version {model_version}")

                    # Promote to staging
                    if model_version:
                        self.tracker.promote_model(
                            self.model_registry_name,
                            model_version,
                            "Staging",
                            description=f"Auto-promoted from training run {run_id[:8]}",
                        )
                else:
                    logger.info("New model not better than production, skipping registration")
                    # Still log the model but don't register
                    mlflow.sklearn.log_model(model.model, "model")
            else:
                if not register_model:
                    logger.info("Model registration disabled")
                else:
                    logger.warning(
                        f"Model metrics below threshold, not registering. "
                        f"RMSE: {test_metrics.get('rmse', 0):.3f} "
                        f"(threshold: {self.DEFAULT_THRESHOLDS['rmse']})"
                    )
                # Log model without registration
                mlflow.sklearn.log_model(model.model, "model")

        # Compile results
        results = {
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "feature_names": data["feature_names"],
            "split_info": data["split_info"],
            "model_version": model_version,
            "run_id": run_id,
            "evaluation_report": evaluation_report,
        }

        logger.info(f"Training pipeline complete. Run ID: {run_id[:8]}")

        return results

    def _meets_threshold(self, metrics: dict[str, float]) -> bool:
        """Check if metrics meet minimum thresholds."""
        rmse = metrics.get("rmse", float("inf"))
        r2 = metrics.get("r2", 0)

        rmse_ok = rmse <= self.DEFAULT_THRESHOLDS["rmse"]
        r2_ok = r2 >= self.DEFAULT_THRESHOLDS["r2"]

        return rmse_ok and r2_ok

    def compare_with_production(self, new_metrics: dict[str, float]) -> bool:
        """Compare new model metrics with current production model.

        Args:
            new_metrics: Metrics from the new model.

        Returns:
            True if new model is better or no production model exists.
        """
        try:
            model_info = self.registry.get_model_info(self.model_registry_name)
            prod_versions = model_info.get("latest_versions", {})

            if "Production" not in prod_versions:
                logger.info("No production model exists, new model wins by default")
                return True

            prod_version = prod_versions["Production"]

            # Get production metrics
            prod_metrics = None
            for v in model_info.get("versions", []):
                if v["version"] == prod_version:
                    prod_metrics = v.get("metrics", {})
                    break

            if not prod_metrics:
                logger.warning("Could not retrieve production metrics")
                return True

            # Compare RMSE (lower is better)
            new_rmse = new_metrics.get("test_rmse", new_metrics.get("rmse", float("inf")))
            prod_rmse = prod_metrics.get("test_rmse", prod_metrics.get("rmse", float("inf")))

            is_better = new_rmse < prod_rmse

            logger.info(
                f"Comparison: New RMSE={new_rmse:.3f}, "
                f"Production RMSE={prod_rmse:.3f}, "
                f"Better={is_better}"
            )

            return is_better

        except Exception as e:
            logger.warning(f"Could not compare with production: {e}")
            return True  # Assume new model is better if comparison fails

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "storage"):
            self.storage.close()
        logger.info("Trainer resources cleaned up")


def train_baseline(
    config_path: str = "config/model_config.yaml",
    db_connection_string: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Convenience function to train the baseline model.

    Args:
        config_path: Path to configuration file.
        db_connection_string: Database connection. Uses DATABASE_URL env var if not provided.
        mlflow_tracking_uri: MLflow URI. Uses MLFLOW_TRACKING_URI env var if not provided.

    Returns:
        Training results dictionary.

    Example:
        >>> results = train_baseline()
        >>> print(f"Test RMSE: {results['test_metrics']['rmse']:.3f}")
    """
    db_conn = db_connection_string or os.getenv("DATABASE_URL")
    mlflow_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    if not db_conn:
        raise TrainerError("DATABASE_URL environment variable not set")

    trainer = ModelTrainer(
        config_path=config_path,
        db_connection_string=db_conn,
        mlflow_tracking_uri=mlflow_uri,
    )

    try:
        return trainer.run_training_pipeline(
            model_type="ridge",
            register_model=True,
        )
    finally:
        trainer.close()
