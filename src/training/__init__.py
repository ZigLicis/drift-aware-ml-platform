"""Training module for weather prediction models.

This module provides:
- Feature engineering with temporal awareness
- Temporal train/validation/test splitting
- Feature configuration management

Example:
    >>> from src.training import FeatureEngineer
    >>> fe = FeatureEngineer()
    >>> df = fe.create_features(raw_df)
    >>> df = fe.create_target(df, horizon_hours=24)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = fe.train_test_split_temporal(df)
"""

from src.training.feature_engineering import FeatureEngineer

__all__ = ["FeatureEngineer"]
