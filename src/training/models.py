"""Model implementations for weather prediction.

This module provides model classes with a consistent interface for training,
prediction, and feature importance extraction.

Example:
    >>> from src.training.models import RidgeBaseline
    >>> model = RidgeBaseline(alpha=1.0)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> importance = model.get_feature_importance(feature_names)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models.

    Provides a consistent interface for model training, prediction,
    and introspection across different model types.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseModel":
        """Train the model.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameter names and values.
        """
        pass

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> dict[str, float] | None:
        """Get feature importance scores.

        Args:
            feature_names: List of feature names corresponding to columns.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not supported.
        """
        return None

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return getattr(self, "_is_fitted", False)


class RidgeBaseline(BaseModel):
    """Ridge regression baseline model.

    A simple regularized linear model that serves as a baseline for
    more complex models. Uses L2 regularization.

    Attributes:
        alpha: Regularization strength.
        model: Underlying sklearn Ridge model.

    Example:
        >>> model = RidgeBaseline(alpha=1.0)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> importance = model.get_feature_importance(feature_names)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Ridge regression model.

        Args:
            alpha: Regularization strength. Larger values = more regularization.
            fit_intercept: Whether to fit an intercept term.
            **kwargs: Additional arguments passed to sklearn.linear_model.Ridge.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._extra_params = kwargs
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            **kwargs,
        )
        self._is_fitted = False
        self._feature_names: list[str] | None = None

        logger.info(f"Initialized RidgeBaseline with alpha={alpha}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "RidgeBaseline":
        """Train the Ridge regression model.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Self for method chaining.
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()

        self.model.fit(X, y)
        self._is_fitted = True

        logger.info(f"Fitted RidgeBaseline on {len(y)} samples")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameter names and values.
        """
        params = {
            "model_type": "ridge",
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            **self._extra_params,
        }

        if self._is_fitted:
            params["n_features"] = len(self.model.coef_)
            params["intercept"] = float(self.model.intercept_)

        return params

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> dict[str, float] | None:
        """Get feature importance based on coefficient magnitudes.

        For linear models, feature importance is the absolute value of
        coefficients (assuming standardized features).

        Args:
            feature_names: List of feature names. Uses stored names if None.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self._is_fitted:
            return None

        names = feature_names or self._feature_names
        if names is None:
            names = [f"feature_{i}" for i in range(len(self.model.coef_))]

        # Use absolute coefficient values as importance
        importance = np.abs(self.model.coef_)

        return dict(zip(names, importance))

    @property
    def coefficients(self) -> np.ndarray | None:
        """Get raw coefficients."""
        if self._is_fitted:
            return self.model.coef_
        return None


class RandomForestModel(BaseModel):
    """Random Forest regression model.

    Ensemble tree-based model that provides feature importance
    and handles non-linear relationships.

    Example:
        >>> model = RandomForestModel(n_estimators=100, max_depth=10)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize Random Forest model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees. None for unlimited.
            min_samples_split: Minimum samples required to split a node.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 for all cores).
            **kwargs: Additional arguments passed to RandomForestRegressor.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._extra_params = kwargs

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self._is_fitted = False
        self._feature_names: list[str] | None = None

        logger.info(
            f"Initialized RandomForestModel: n_estimators={n_estimators}, "
            f"max_depth={max_depth}"
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "RandomForestModel":
        """Train the Random Forest model.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Self for method chaining.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()

        self.model.fit(X, y)
        self._is_fitted = True

        logger.info(f"Fitted RandomForestModel on {len(y)} samples")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        params = {
            "model_type": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            **self._extra_params,
        }

        if self._is_fitted:
            params["n_features"] = self.model.n_features_in_

        return params

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> dict[str, float] | None:
        """Get feature importance from the forest.

        Args:
            feature_names: List of feature names.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self._is_fitted:
            return None

        names = feature_names or self._feature_names
        if names is None:
            names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]

        return dict(zip(names, self.model.feature_importances_))


class GradientBoostingModel(BaseModel):
    """Gradient Boosting regression model.

    Sequential ensemble that builds trees to correct previous errors.
    Often achieves best performance on tabular data.

    Example:
        >>> model = GradientBoostingModel(n_estimators=100, learning_rate=0.1)
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_samples_split: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize Gradient Boosting model.

        Args:
            n_estimators: Number of boosting stages.
            max_depth: Maximum depth of individual trees.
            learning_rate: Shrinkage rate for each tree's contribution.
            min_samples_split: Minimum samples required to split a node.
            random_state: Random seed.
            **kwargs: Additional arguments.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._extra_params = kwargs

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs,
        )
        self._is_fitted = False
        self._feature_names: list[str] | None = None

        logger.info(
            f"Initialized GradientBoostingModel: n_estimators={n_estimators}, "
            f"learning_rate={learning_rate}"
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "GradientBoostingModel":
        """Train the Gradient Boosting model."""
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()

        self.model.fit(X, y)
        self._is_fitted = True

        logger.info(f"Fitted GradientBoostingModel on {len(y)} samples")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        params = {
            "model_type": "gradient_boosting",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            **self._extra_params,
        }

        if self._is_fitted:
            params["n_features"] = self.model.n_features_in_

        return params

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> dict[str, float] | None:
        """Get feature importance."""
        if not self._is_fitted:
            return None

        names = feature_names or self._feature_names
        if names is None:
            names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]

        return dict(zip(names, self.model.feature_importances_))


def create_model(model_type: str, **kwargs: Any) -> BaseModel:
    """Factory function to create model instances.

    Args:
        model_type: Type of model ("ridge", "random_forest", "gradient_boosting").
        **kwargs: Model-specific parameters.

    Returns:
        Configured model instance.

    Raises:
        ValueError: If model_type is unknown.

    Example:
        >>> model = create_model("ridge", alpha=0.5)
        >>> model = create_model("random_forest", n_estimators=200)
    """
    model_classes = {
        "ridge": RidgeBaseline,
        "random_forest": RandomForestModel,
        "gradient_boosting": GradientBoostingModel,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(model_classes.keys())}"
        )

    return model_classes[model_type](**kwargs)
