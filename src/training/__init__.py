"""Training module for weather prediction models.

This module provides:
- Feature engineering with temporal awareness
- Model implementations (Ridge, Random Forest, Gradient Boosting)
- Model evaluation and visualization
- Complete training pipeline orchestration with MLflow

Example:
    >>> from src.training import ModelTrainer, train_baseline
    >>>
    >>> # Quick baseline training
    >>> results = train_baseline()
    >>>
    >>> # Or full control
    >>> trainer = ModelTrainer(
    ...     config_path="config/model_config.yaml",
    ...     db_connection_string="postgresql://...",
    ...     mlflow_tracking_uri="http://localhost:5001"
    ... )
    >>> results = trainer.run_training_pipeline(model_type="ridge")
"""

from src.training.evaluation import ModelEvaluator
from src.training.feature_engineering import FeatureEngineer
from src.training.models import (
    BaseModel,
    GradientBoostingModel,
    RandomForestModel,
    RidgeBaseline,
    create_model,
)
from src.training.trainer import ModelTrainer, train_baseline

__all__ = [
    "FeatureEngineer",
    "ModelEvaluator",
    "BaseModel",
    "RidgeBaseline",
    "RandomForestModel",
    "GradientBoostingModel",
    "create_model",
    "ModelTrainer",
    "train_baseline",
]
