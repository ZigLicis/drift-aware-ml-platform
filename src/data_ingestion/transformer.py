"""Data transformation module for weather data processing.

This module provides transformations for converting API data to database
format and creating features for ML models.

Example:
    >>> from src.data_ingestion.transformer import DataTransformer
    >>> transformer = DataTransformer()
    >>> df_transformed = transformer.transform_for_storage(df)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transform weather data for storage and ML model training.

    Provides methods to convert API response data to database schema format,
    create target variables for prediction, and extract temporal features.

    Attributes:
        default_timezone: Default timezone for data conversion.

    Example:
        >>> transformer = DataTransformer()
        >>> df = transformer.transform_for_storage(raw_df)
        >>> df = transformer.add_temporal_features(df)
        >>> df = transformer.create_target_variable(df)
    """

    # Column mapping from internal names to database schema
    DB_COLUMN_MAPPING = {
        "timestamp": "timestamp",
        "location_name": "location_name",
        "latitude": "latitude",
        "longitude": "longitude",
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "precipitation": "precipitation",
        "wind_speed": "wind_speed_10m",
        "pressure": "surface_pressure",
    }

    def __init__(self, default_timezone: str = "America/New_York") -> None:
        """Initialize the transformer.

        Args:
            default_timezone: Default timezone for conversions.
        """
        self.default_timezone = default_timezone
        try:
            self._tz = pytz.timezone(default_timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            raise ValueError(f"Invalid timezone: {default_timezone}") from e

        logger.info(f"Initialized DataTransformer with timezone {default_timezone}")

    def transform_for_storage(
        self,
        df: pd.DataFrame,
        data_source: str = "open-meteo-api",
        batch_id: str | None = None,
    ) -> pd.DataFrame:
        """Transform API data to database schema format.

        Performs the following transformations:
        - Renames columns to match database schema
        - Adds metadata columns (ingestion_timestamp, data_source, batch_id)
        - Converts timestamps to UTC for storage
        - Rounds float values appropriately
        - Adds raw_data column as JSON

        Args:
            df: DataFrame with weather data from API client.
            data_source: Identifier for the data source.
            batch_id: Unique identifier for this ingestion batch.
                     Generated if not provided.

        Returns:
            DataFrame ready for database insertion.

        Example:
            >>> transformer = DataTransformer()
            >>> df_storage = transformer.transform_for_storage(
            ...     raw_df,
            ...     data_source="open-meteo-api",
            ...     batch_id="batch-2024-01-15-001"
            ... )
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for transformation")
            return df.copy()

        df = df.copy()
        batch_id = batch_id or str(uuid.uuid4())

        logger.info(f"Transforming {len(df)} records for storage (batch: {batch_id[:8]}...)")

        # Rename columns to database schema
        rename_map = {k: v for k, v in self.DB_COLUMN_MAPPING.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Convert timestamp to UTC for consistent storage
        if "timestamp" in df.columns:
            df["timestamp"] = self._ensure_utc(df["timestamp"])

        # Round numeric values
        numeric_columns = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "latitude",
            "longitude",
        ]
        for col in numeric_columns:
            if col in df.columns:
                # Round to appropriate precision
                if col in ["latitude", "longitude"]:
                    df[col] = df[col].round(6)
                elif col == "relative_humidity_2m":
                    df[col] = df[col].round(1)
                else:
                    df[col] = df[col].round(2)

        # Add metadata columns
        df["ingestion_timestamp"] = datetime.now(timezone.utc)
        df["data_source"] = data_source
        df["batch_id"] = batch_id

        # Create raw_data JSON column with original values
        raw_data_cols = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
        ]
        existing_cols = [c for c in raw_data_cols if c in df.columns]
        if existing_cols:
            df["raw_data"] = df[existing_cols].apply(
                lambda row: row.to_dict(), axis=1
            )

        # Add created_at and updated_at
        now = datetime.now(timezone.utc)
        df["created_at"] = now
        df["updated_at"] = now

        logger.info(f"Transformation complete. Columns: {df.columns.tolist()}")
        return df

    def _ensure_utc(self, timestamps: pd.Series) -> pd.Series:
        """Ensure timestamps are in UTC.

        Args:
            timestamps: Series of timestamps.

        Returns:
            Series with UTC timestamps.
        """
        if timestamps.dt.tz is None:
            # Assume local timezone if naive
            timestamps = timestamps.dt.tz_localize(self._tz)

        return timestamps.dt.tz_convert("UTC")

    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_hours: int = 24,
        target_column: str = "temperature_2m",
    ) -> pd.DataFrame:
        """Create target variable for prediction.

        Creates 'next_day_max_temp' as the maximum temperature in the next
        N hours from each observation.

        Args:
            df: DataFrame with weather data.
            target_hours: Hours ahead to look for target calculation.
            target_column: Column to use for target calculation.

        Returns:
            DataFrame with added target column.

        Example:
            >>> transformer = DataTransformer()
            >>> df = transformer.create_target_variable(df, target_hours=24)
            >>> # Now df has 'next_day_max_temp' column
        """
        # Use internal column name if database column name provided
        temp_col = target_column
        if temp_col not in df.columns and "temperature" in df.columns:
            temp_col = "temperature"

        if temp_col not in df.columns:
            logger.error(f"Temperature column not found. Available: {df.columns.tolist()}")
            raise ValueError(f"Column '{target_column}' not found in DataFrame")

        df = df.copy()

        # Ensure sorted by timestamp
        timestamp_col = "timestamp"
        if timestamp_col in df.columns:
            df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Calculate rolling max for next N hours
        # We need to look forward, so we reverse, calculate rolling, then reverse back
        temps = df[temp_col].values
        n_records = len(temps)
        target_values = np.full(n_records, np.nan)

        for i in range(n_records):
            # Look ahead target_hours records (assuming hourly data)
            end_idx = min(i + target_hours + 1, n_records)
            if end_idx > i + 1:
                future_temps = temps[i + 1 : end_idx]
                if len(future_temps) > 0:
                    target_values[i] = np.nanmax(future_temps)

        df["next_day_max_temp"] = target_values

        # Count valid targets
        valid_targets = df["next_day_max_temp"].notna().sum()
        logger.info(
            f"Created target variable. "
            f"Valid targets: {valid_targets}/{n_records} "
            f"({valid_targets / n_records * 100:.1f}%)"
        )

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from timestamp.

        Adds the following features:
        - hour_of_day: 0-23
        - day_of_week: 0-6 (Monday=0)
        - day_of_year: 1-365/366
        - month: 1-12
        - is_weekend: boolean
        - season: categorical (winter, spring, summer, fall)

        Args:
            df: DataFrame with timestamp column.

        Returns:
            DataFrame with added temporal features.

        Example:
            >>> transformer = DataTransformer()
            >>> df = transformer.add_temporal_features(df)
            >>> print(df[['timestamp', 'hour_of_day', 'day_of_week', 'is_weekend']].head())
        """
        timestamp_col = "timestamp"
        if timestamp_col not in df.columns:
            logger.error("No timestamp column found for temporal features")
            raise ValueError("DataFrame must have 'timestamp' column")

        df = df.copy()

        # Convert to local timezone for meaningful features
        ts = df[timestamp_col]
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(self._tz)

        # Extract temporal features
        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_year"] = ts.dt.dayofyear
        df["month"] = ts.dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        # Add season (Northern Hemisphere)
        df["season"] = df["month"].map(self._get_season)

        # Cyclical encoding for hour and day (useful for ML)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        logger.info(f"Added temporal features to {len(df)} records")
        return df

    @staticmethod
    def _get_season(month: int) -> str:
        """Map month to season (Northern Hemisphere)."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        lag_hours: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add lagged features for time series modeling.

        Args:
            df: DataFrame with weather data.
            columns: Columns to create lag features for.
                    Defaults to temperature and humidity.
            lag_hours: List of lag periods in hours.
                      Defaults to [1, 3, 6, 12, 24].

        Returns:
            DataFrame with added lag features.

        Example:
            >>> transformer = DataTransformer()
            >>> df = transformer.add_lag_features(
            ...     df,
            ...     columns=['temperature_2m'],
            ...     lag_hours=[1, 6, 24]
            ... )
        """
        columns = columns or ["temperature_2m", "relative_humidity_2m"]
        lag_hours = lag_hours or [1, 3, 6, 12, 24]

        df = df.copy()

        # Ensure sorted by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        for col in columns:
            if col not in df.columns:
                # Try alternative column name
                alt_col = col.replace("_2m", "").replace("relative_", "")
                if alt_col in df.columns:
                    col = alt_col
                else:
                    logger.warning(f"Column {col} not found, skipping lag features")
                    continue

            for lag in lag_hours:
                lag_col_name = f"{col}_lag_{lag}h"
                df[lag_col_name] = df[col].shift(lag)

        logger.info(
            f"Added lag features for {len(columns)} columns "
            f"with lags {lag_hours}"
        )
        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add rolling statistics features.

        Args:
            df: DataFrame with weather data.
            columns: Columns to calculate rolling stats for.
            windows: Window sizes in hours.
                    Defaults to [6, 12, 24].

        Returns:
            DataFrame with added rolling features.

        Example:
            >>> transformer = DataTransformer()
            >>> df = transformer.add_rolling_features(
            ...     df,
            ...     columns=['temperature_2m'],
            ...     windows=[6, 24]
            ... )
        """
        columns = columns or ["temperature_2m"]
        windows = windows or [6, 12, 24]

        df = df.copy()

        # Ensure sorted by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        for col in columns:
            if col not in df.columns:
                # Try alternative column name
                alt_col = col.replace("_2m", "")
                if alt_col in df.columns:
                    col = alt_col
                else:
                    logger.warning(f"Column {col} not found, skipping rolling features")
                    continue

            for window in windows:
                df[f"{col}_rolling_mean_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_rolling_std_{window}h"] = (
                    df[col].rolling(window=window, min_periods=2).std()
                )
                df[f"{col}_rolling_min_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                df[f"{col}_rolling_max_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )

        logger.info(
            f"Added rolling features for {len(columns)} columns "
            f"with windows {windows}"
        )
        return df
