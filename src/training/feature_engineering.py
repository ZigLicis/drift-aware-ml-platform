"""Feature engineering for weather prediction models.

This module provides the FeatureEngineer class for creating ML features
from weather data with proper temporal handling to prevent data leakage.

IMPORTANT: All operations preserve temporal ordering to prevent data leakage.
- Lag features only look backward in time
- Rolling features only use past observations
- Train/test split is temporal, NOT random

Example:
    >>> from src.training.feature_engineering import FeatureEngineer
    >>> fe = FeatureEngineer()
    >>> df = fe.create_features(raw_df)
    >>> df = fe.create_target(df, horizon_hours=24)
    >>> splits = fe.train_test_split_temporal(df, test_days=7)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FeatureEngineerError(Exception):
    """Raised when feature engineering fails."""

    pass


class FeatureEngineer:
    """Create and manage features for weather prediction models.

    Handles feature creation with temporal awareness to prevent data leakage.
    All operations ensure that only past data is used for feature calculation.

    Attributes:
        feature_config: Dictionary defining feature groups and parameters.
        target_column: Name of the target column to create.

    Example:
        >>> fe = FeatureEngineer()
        >>> df = fe.create_features(raw_df)
        >>> df = fe.create_target(df, horizon_hours=24)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = fe.train_test_split_temporal(df)
    """

    # Default feature configuration
    DEFAULT_CONFIG = {
        "weather_features": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
        ],
        "temporal_features": [
            "hour_of_day",
            "day_of_week",
            "day_of_year",
            "is_weekend",
            "hour_sin",
            "hour_cos",
        ],
        "lag_hours": [1, 6, 24],
        "rolling_windows": [6, 24],
        "lag_columns": ["temperature_2m", "relative_humidity_2m"],
        "rolling_columns": ["temperature_2m"],
        "target_column": "target_next_day_max_temp",
        "prediction_horizon_hours": 24,
    }

    # Columns that should never be used as features
    EXCLUDED_COLUMNS = {
        "id",
        "timestamp",
        "location_name",
        "latitude",
        "longitude",
        "raw_data",
        "batch_id",
        "data_source",
        "ingestion_timestamp",
        "created_at",
        "updated_at",
        "season",  # Categorical - use encoded version instead
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the feature engineer.

        Args:
            config: Feature configuration dictionary. Uses defaults if None.
                Keys:
                - weather_features: List of base weather columns
                - temporal_features: List of temporal feature names
                - lag_hours: List of lag periods in hours
                - rolling_windows: List of rolling window sizes
                - lag_columns: Columns to create lag features for
                - rolling_columns: Columns to create rolling features for
                - target_column: Name for the target variable
                - prediction_horizon_hours: Hours ahead to predict
        """
        self.feature_config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.target_column = self.feature_config["target_column"]
        self._feature_names: list[str] | None = None

        logger.info(
            f"FeatureEngineer initialized: "
            f"lag_hours={self.feature_config['lag_hours']}, "
            f"rolling_windows={self.feature_config['rolling_windows']}"
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "FeatureEngineer":
        """Create FeatureEngineer from YAML config file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Configured FeatureEngineer instance.
        """
        with open(config_path) as f:
            full_config = yaml.safe_load(f)

        # Extract features section
        config = {}
        if "features" in full_config:
            features = full_config["features"]
            config["weather_features"] = features.get("weather", cls.DEFAULT_CONFIG["weather_features"])
            config["temporal_features"] = features.get("temporal", cls.DEFAULT_CONFIG["temporal_features"])
            config["lag_hours"] = features.get("lag_hours", cls.DEFAULT_CONFIG["lag_hours"])
            config["rolling_windows"] = features.get("rolling_windows", cls.DEFAULT_CONFIG["rolling_windows"])

        if "model" in full_config:
            config["prediction_horizon_hours"] = full_config["model"].get(
                "prediction_horizon_hours", cls.DEFAULT_CONFIG["prediction_horizon_hours"]
            )
            config["target_column"] = full_config["model"].get(
                "target", cls.DEFAULT_CONFIG["target_column"]
            )

        return cls(config)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features for the dataset.

        Adds temporal features, lag features, and rolling features.
        All calculations only use past data to prevent leakage.

        Args:
            df: DataFrame with weather data (must have 'timestamp' column).

        Returns:
            DataFrame with all engineered features added.

        Raises:
            FeatureEngineerError: If required columns are missing.
        """
        if "timestamp" not in df.columns:
            raise FeatureEngineerError("DataFrame must have 'timestamp' column")

        df = df.copy()

        # Ensure sorted by timestamp (critical for temporal features)
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Creating features for {len(df)} records")

        # 1. Add temporal features
        df = self._add_temporal_features(df)

        # 2. Add lag features (only looking backward)
        df = self._add_lag_features(df)

        # 3. Add rolling features (only using past data)
        df = self._add_rolling_features(df)

        # 4. Handle any remaining missing values in features
        df = self._handle_missing_values(df)

        # Cache feature names
        self._feature_names = self.get_feature_names()

        logger.info(f"Feature creation complete. Total features: {len(self._feature_names)}")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features.

        These features are derived from the timestamp itself,
        not from other observations, so no leakage risk.
        """
        ts = pd.to_datetime(df["timestamp"])

        # Handle timezone conversion if present
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("America/New_York")

        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_year"] = ts.dt.dayofyear
        df["month"] = ts.dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding (helps models understand hour 23 is close to hour 0)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        logger.debug("Added temporal features")
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features.

        IMPORTANT: Uses shift() which only looks at PAST observations.
        shift(n) takes the value from n rows BEFORE, not after.
        """
        lag_columns = self.feature_config["lag_columns"]
        lag_hours = self.feature_config["lag_hours"]

        for col in lag_columns:
            if col not in df.columns:
                logger.warning(f"Lag column '{col}' not found, skipping")
                continue

            for lag in lag_hours:
                # shift(lag) brings past values forward
                # Row i gets the value from row i-lag (past data only)
                lag_col_name = f"{col}_lag_{lag}h"
                df[lag_col_name] = df[col].shift(lag)

        logger.debug(f"Added lag features for {len(lag_columns)} columns")
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics.

        IMPORTANT: Uses rolling() with default behavior which is BACKWARD-looking.
        For row i, rolling(window=n) uses rows [i-n+1, i] (past data only).
        """
        rolling_columns = self.feature_config["rolling_columns"]
        windows = self.feature_config["rolling_windows"]

        for col in rolling_columns:
            if col not in df.columns:
                logger.warning(f"Rolling column '{col}' not found, skipping")
                continue

            for window in windows:
                # Rolling mean - backward looking by default
                df[f"{col}_rolling_mean_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )

                # Rolling std - need at least 2 observations
                df[f"{col}_rolling_std_{window}h"] = (
                    df[col].rolling(window=window, min_periods=2).std()
                )

                # Rolling min/max
                df[f"{col}_rolling_min_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                df[f"{col}_rolling_max_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )

        logger.debug(f"Added rolling features for {len(rolling_columns)} columns")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features.

        Strategy:
        - Lag features: NaN at start is expected, will be dropped with target
        - Rolling features: Forward fill then backward fill for edge cases
        - Weather features: Interpolate if sparse, otherwise forward fill
        """
        # Get feature columns (exclude target and metadata)
        feature_cols = [
            c for c in df.columns
            if c not in self.EXCLUDED_COLUMNS and c != self.target_column
        ]

        # For rolling std that may have NaN where window is too small
        for col in feature_cols:
            if "_rolling_std_" in col:
                df[col] = df[col].fillna(0)

        # For other numeric features, use forward fill (temporally safe)
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill()

        return df

    def create_target(
        self,
        df: pd.DataFrame,
        horizon_hours: int | None = None,
    ) -> pd.DataFrame:
        """Create the target variable for prediction.

        Target is the maximum temperature in the next `horizon_hours`.
        Rows where target cannot be calculated (end of dataset) are dropped.

        IMPORTANT: Target uses FUTURE data (as expected for prediction).
        The train/test split ensures we never train on data that would
        require knowing the target's future values.

        Args:
            df: DataFrame with weather data.
            horizon_hours: Hours ahead to look for max temperature.
                          Uses config default if not provided.

        Returns:
            DataFrame with target column added, trailing rows dropped.
        """
        horizon_hours = horizon_hours or self.feature_config["prediction_horizon_hours"]

        if "temperature_2m" not in df.columns:
            raise FeatureEngineerError(
                "Cannot create target: 'temperature_2m' column not found"
            )

        df = df.copy()

        # Ensure sorted by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Calculate future max temperature
        # We look AHEAD for the target (this is intentional - it's what we predict)
        # shift(-horizon_hours) would give exact future value
        # Instead, we want max over next horizon_hours

        temps = df["temperature_2m"].values
        n_records = len(temps)
        target_values = np.full(n_records, np.nan)

        for i in range(n_records):
            # Look ahead horizon_hours (future data for target)
            start_idx = i + 1
            end_idx = min(i + horizon_hours + 1, n_records)

            if start_idx < n_records:
                future_temps = temps[start_idx:end_idx]
                if len(future_temps) >= horizon_hours // 2:  # At least half the window
                    target_values[i] = np.nanmax(future_temps)

        df[self.target_column] = target_values

        # Count valid targets before dropping
        valid_before = df[self.target_column].notna().sum()

        # Drop rows where target couldn't be calculated
        df = df.dropna(subset=[self.target_column]).reset_index(drop=True)

        logger.info(
            f"Created target '{self.target_column}' "
            f"(horizon={horizon_hours}h). "
            f"Valid samples: {len(df)}/{n_records} "
            f"(dropped {n_records - len(df)} trailing rows)"
        )

        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix X and target vector y.

        Excludes non-feature columns and handles remaining NaN values.

        Args:
            df: DataFrame with features and target.

        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series.

        Raises:
            FeatureEngineerError: If target column is missing or NaN handling fails.
        """
        if self.target_column not in df.columns:
            raise FeatureEngineerError(
                f"Target column '{self.target_column}' not found. "
                f"Run create_target() first."
            )

        # Get target
        y = df[self.target_column].copy()

        # Get features (exclude metadata and target)
        exclude_cols = self.EXCLUDED_COLUMNS | {self.target_column}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Only keep numeric columns
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()

        # Final NaN check - drop rows with any NaN in features or target
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        n_dropped = (~valid_mask).sum()

        if n_dropped > 0:
            logger.warning(f"Dropping {n_dropped} rows with NaN values")
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)

        logger.info(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def train_test_split_temporal(
        self,
        df: pd.DataFrame,
        test_days: int = 7,
        validation_days: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data temporally into train, validation, and test sets.

        CRITICAL: This is a TEMPORAL split, NOT random.
        - Train: oldest data (majority of dataset)
        - Validation: middle chunk (for hyperparameter tuning)
        - Test: most recent data (final evaluation)

        This prevents data leakage by ensuring:
        1. We never train on data from the future
        2. Validation set is between train and test chronologically
        3. Test set represents truly unseen future data

        Args:
            df: DataFrame with features and target.
            test_days: Number of days for test set (most recent).
            validation_days: Number of days for validation set.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).

        Raises:
            FeatureEngineerError: If insufficient data for splits.
        """
        if "timestamp" not in df.columns:
            raise FeatureEngineerError(
                "DataFrame must have 'timestamp' column for temporal split"
            )

        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Calculate split points based on timestamps
        timestamps = pd.to_datetime(df["timestamp"])
        max_ts = timestamps.max()
        min_ts = timestamps.min()
        total_days = (max_ts - min_ts).days

        if total_days < test_days + validation_days + 1:
            raise FeatureEngineerError(
                f"Insufficient data for split. "
                f"Have {total_days} days, need at least {test_days + validation_days + 1}"
            )

        # Calculate cutoff timestamps
        test_cutoff = max_ts - timedelta(days=test_days)
        val_cutoff = test_cutoff - timedelta(days=validation_days)

        # Create masks
        train_mask = timestamps < val_cutoff
        val_mask = (timestamps >= val_cutoff) & (timestamps < test_cutoff)
        test_mask = timestamps >= test_cutoff

        # Log split information
        logger.info(
            f"Temporal split:\n"
            f"  Train: {min_ts.date()} to {val_cutoff.date()} ({train_mask.sum()} samples)\n"
            f"  Val:   {val_cutoff.date()} to {test_cutoff.date()} ({val_mask.sum()} samples)\n"
            f"  Test:  {test_cutoff.date()} to {max_ts.date()} ({test_mask.sum()} samples)"
        )

        # Verify no overlap (sanity check)
        assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(df), \
            "Split masks don't cover all data!"

        # Get feature matrix for each split
        X_train, y_train = self.get_feature_matrix(df[train_mask])
        X_val, y_val = self.get_feature_matrix(df[val_mask])
        X_test, y_test = self.get_feature_matrix(df[test_mask])

        # Store split info for reproducibility
        self._split_info = {
            "train_start": str(min_ts.date()),
            "train_end": str(val_cutoff.date()),
            "val_start": str(val_cutoff.date()),
            "val_end": str(test_cutoff.date()),
            "test_start": str(test_cutoff.date()),
            "test_end": str(max_ts.date()),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names used.

        Returns:
            List of feature column names.
        """
        if self._feature_names is not None:
            return self._feature_names

        # Build expected feature names from config
        features = []

        # Weather features
        features.extend(self.feature_config["weather_features"])

        # Temporal features
        features.extend(self.feature_config["temporal_features"])

        # Additional temporal (always added)
        features.extend(["month", "day_sin", "day_cos", "month_sin", "month_cos"])

        # Lag features
        for col in self.feature_config["lag_columns"]:
            for lag in self.feature_config["lag_hours"]:
                features.append(f"{col}_lag_{lag}h")

        # Rolling features
        for col in self.feature_config["rolling_columns"]:
            for window in self.feature_config["rolling_windows"]:
                features.append(f"{col}_rolling_mean_{window}h")
                features.append(f"{col}_rolling_std_{window}h")
                features.append(f"{col}_rolling_min_{window}h")
                features.append(f"{col}_rolling_max_{window}h")

        return features

    def save_feature_config(self, path: str | Path) -> None:
        """Save feature configuration for reproducibility.

        Args:
            path: Path to save configuration (JSON format).
        """
        path = Path(path)

        config_to_save = {
            "feature_config": self.feature_config,
            "target_column": self.target_column,
            "excluded_columns": list(self.EXCLUDED_COLUMNS),
            "feature_names": self.get_feature_names(),
        }

        # Add split info if available
        if hasattr(self, "_split_info"):
            config_to_save["split_info"] = self._split_info

        with open(path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        logger.info(f"Saved feature configuration to {path}")

    def get_split_info(self) -> dict[str, Any] | None:
        """Get information about the train/val/test split.

        Returns:
            Dictionary with split boundaries and sample counts,
            or None if split hasn't been performed.
        """
        return getattr(self, "_split_info", None)
