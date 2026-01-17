"""MLflow utilities for experiment tracking and model registry.

This module provides utilities for:
- Experiment tracking with automatic tagging and artifact logging
- Model registry management and version comparison
- Evaluation metrics and visualization logging

Example:
    >>> from src.mlflow_utils import ExperimentTracker, ModelRegistry
    >>> tracker = ExperimentTracker(
    ...     tracking_uri="http://localhost:5001",
    ...     experiment_name="weather-prediction"
    ... )
    >>> with tracker.start_run("baseline-model"):
    ...     tracker.log_dataset_info(df, "training_data")
    ...     # ... train model ...
    ...     tracker.log_model_with_signature(model, X_sample, "weather-model")
"""

from src.mlflow_utils.tracking import ExperimentTracker
from src.mlflow_utils.registry import ModelRegistry

__all__ = ["ExperimentTracker", "ModelRegistry"]
