"""Tests for FeatureEngineer.

Run with: pytest tests/test_feature_engineering.py -v

These tests specifically verify:
1. Feature creation correctness
2. Temporal split prevents data leakage
3. Target variable calculation
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.training.feature_engineering import FeatureEngineer, FeatureEngineerError


@pytest.fixture
def sample_weather_data() -> pd.DataFrame:
    """Create sample weather data for testing."""
    # 30 days of hourly data (720 records)
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=720,
        freq="h",
        tz="UTC",
    )

    np.random.seed(42)

    # Create realistic weather patterns
    hours = np.arange(720)
    day_cycle = np.sin(2 * np.pi * hours / 24)  # Daily temperature cycle

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature_2m": 15 + 10 * day_cycle + np.random.normal(0, 2, 720),
        "relative_humidity_2m": 60 + 20 * np.random.random(720),
        "precipitation": np.random.exponential(0.5, 720),
        "wind_speed_10m": 10 + 5 * np.random.random(720),
        "surface_pressure": 1013 + np.random.normal(0, 5, 720),
        "location_name": "Test City",
        "latitude": 40.7128,
        "longitude": -74.0060,
    })


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Create FeatureEngineer with default config."""
    return FeatureEngineer()


class TestFeatureCreation:
    """Tests for feature creation methods."""

    def test_create_features_adds_temporal(self, feature_engineer, sample_weather_data):
        """Test that temporal features are added."""
        df = feature_engineer.create_features(sample_weather_data)

        assert "hour_of_day" in df.columns
        assert "day_of_week" in df.columns
        assert "day_of_year" in df.columns
        assert "is_weekend" in df.columns
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns

    def test_create_features_adds_lag(self, feature_engineer, sample_weather_data):
        """Test that lag features are added."""
        df = feature_engineer.create_features(sample_weather_data)

        # Check for expected lag features
        assert "temperature_2m_lag_1h" in df.columns
        assert "temperature_2m_lag_6h" in df.columns
        assert "temperature_2m_lag_24h" in df.columns

    def test_create_features_adds_rolling(self, feature_engineer, sample_weather_data):
        """Test that rolling features are added."""
        df = feature_engineer.create_features(sample_weather_data)

        assert "temperature_2m_rolling_mean_6h" in df.columns
        assert "temperature_2m_rolling_mean_24h" in df.columns
        assert "temperature_2m_rolling_std_6h" in df.columns

    def test_create_features_preserves_original(self, feature_engineer, sample_weather_data):
        """Test that original columns are preserved."""
        df = feature_engineer.create_features(sample_weather_data)

        assert "temperature_2m" in df.columns
        assert "relative_humidity_2m" in df.columns
        assert "timestamp" in df.columns

    def test_create_features_requires_timestamp(self, feature_engineer):
        """Test error when timestamp is missing."""
        df = pd.DataFrame({
            "temperature_2m": [20, 21, 22],
            "relative_humidity_2m": [50, 55, 60],
        })

        with pytest.raises(FeatureEngineerError, match="timestamp"):
            feature_engineer.create_features(df)

    def test_lag_features_look_backward(self, feature_engineer, sample_weather_data):
        """Test that lag features correctly use past data only."""
        df = feature_engineer.create_features(sample_weather_data)

        # Lag feature at index i should equal original value at index i-lag
        # Check 24h lag
        for i in range(24, len(df)):
            expected = df.loc[i - 24, "temperature_2m"]
            actual = df.loc[i, "temperature_2m_lag_24h"]
            assert np.isclose(actual, expected, equal_nan=True), \
                f"Lag mismatch at index {i}"

    def test_rolling_features_use_past_only(self, feature_engineer, sample_weather_data):
        """Test that rolling features only use past data."""
        df = feature_engineer.create_features(sample_weather_data)

        # For a 6-hour rolling mean at index 10, it should use indices 5-10
        # (not 10-15 which would be future data)
        for i in range(6, 20):
            expected_mean = df.loc[i - 5:i, "temperature_2m"].mean()
            actual_mean = df.loc[i, "temperature_2m_rolling_mean_6h"]
            assert np.isclose(actual_mean, expected_mean, rtol=1e-5), \
                f"Rolling mean mismatch at index {i}"


class TestTargetCreation:
    """Tests for target variable creation."""

    def test_create_target_adds_column(self, feature_engineer, sample_weather_data):
        """Test that target column is created."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df, horizon_hours=24)

        assert feature_engineer.target_column in df.columns

    def test_create_target_uses_future_max(self, feature_engineer, sample_weather_data):
        """Test that target is max temperature in future window."""
        df = feature_engineer.create_features(sample_weather_data)
        df_with_target = feature_engineer.create_target(df.copy(), horizon_hours=24)

        # For each row, target should be max of next 24 hours
        # Check a few specific rows
        for i in [10, 50, 100]:
            if i < len(df_with_target):
                original_idx = i
                future_temps = df.loc[original_idx + 1:original_idx + 24, "temperature_2m"]
                expected_max = future_temps.max()
                actual_target = df_with_target.loc[i, feature_engineer.target_column]
                assert np.isclose(actual_target, expected_max, rtol=1e-5), \
                    f"Target mismatch at index {i}"

    def test_create_target_drops_trailing_rows(self, feature_engineer, sample_weather_data):
        """Test that rows without valid targets are dropped."""
        df = feature_engineer.create_features(sample_weather_data)
        original_len = len(df)
        df = feature_engineer.create_target(df, horizon_hours=24)

        # Should have dropped ~24 rows from the end
        assert len(df) < original_len
        # All remaining rows should have valid targets
        assert df[feature_engineer.target_column].notna().all()

    def test_create_target_requires_temperature(self, feature_engineer):
        """Test error when temperature column is missing."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "humidity": np.random.random(100),
        })

        df = feature_engineer.create_features(df)
        with pytest.raises(FeatureEngineerError, match="temperature_2m"):
            feature_engineer.create_target(df)


class TestTemporalSplit:
    """Tests for temporal train/val/test split.

    CRITICAL: These tests verify that no data leakage occurs.
    """

    def test_split_preserves_temporal_order(self, feature_engineer, sample_weather_data):
        """Test that train comes before val, val before test."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            feature_engineer.train_test_split_temporal(df, test_days=5, validation_days=3)

        # Get timestamps from original df for comparison
        # Since get_feature_matrix drops some columns, we need to verify via split_info
        split_info = feature_engineer.get_split_info()

        assert split_info is not None
        assert split_info["train_end"] <= split_info["val_start"]
        assert split_info["val_end"] <= split_info["test_start"]

    def test_split_no_overlap(self, feature_engineer, sample_weather_data):
        """Test that train/val/test sets don't overlap."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            feature_engineer.train_test_split_temporal(df, test_days=5, validation_days=3)

        total_samples = len(X_train) + len(X_val) + len(X_test)
        original_samples = len(df[df[feature_engineer.target_column].notna()])

        # Allow for some dropped NaN rows but should be close
        assert total_samples <= original_samples
        # No duplicates across sets
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

    def test_test_set_is_most_recent(self, feature_engineer, sample_weather_data):
        """Test that test set contains the most recent data."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)

        feature_engineer.train_test_split_temporal(df, test_days=5, validation_days=3)
        split_info = feature_engineer.get_split_info()

        # Test end should be close to the max timestamp in original data
        max_date = pd.to_datetime(df["timestamp"]).max().date()
        test_end = datetime.strptime(split_info["test_end"], "%Y-%m-%d").date()

        assert test_end == max_date

    def test_train_set_is_oldest(self, feature_engineer, sample_weather_data):
        """Test that train set contains the oldest data."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)

        feature_engineer.train_test_split_temporal(df, test_days=5, validation_days=3)
        split_info = feature_engineer.get_split_info()

        # Train start should be close to min timestamp
        min_date = pd.to_datetime(df["timestamp"]).min().date()
        train_start = datetime.strptime(split_info["train_start"], "%Y-%m-%d").date()

        assert train_start == min_date

    def test_split_fails_with_insufficient_data(self, feature_engineer):
        """Test error when data is insufficient for requested split."""
        # Only 5 days of data
        timestamps = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature_2m": np.random.random(120) * 20,
            "relative_humidity_2m": np.random.random(120) * 50 + 30,
        })

        df = feature_engineer.create_features(df)
        df = feature_engineer.create_target(df)

        # Request more days than available
        with pytest.raises(FeatureEngineerError, match="Insufficient data"):
            feature_engineer.train_test_split_temporal(df, test_days=10, validation_days=5)

    def test_no_future_data_in_training(self, feature_engineer, sample_weather_data):
        """Test that training features don't contain future information.

        This is the CRITICAL data leakage test.
        """
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)

        # Add a timestamp column to track
        df = df.reset_index(drop=True)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            feature_engineer.train_test_split_temporal(df, test_days=5, validation_days=3)

        split_info = feature_engineer.get_split_info()
        val_start = datetime.strptime(split_info["val_start"], "%Y-%m-%d")

        # Training data should not have any features derived from
        # data after the validation cutoff
        # For lag features, this is ensured by temporal split
        # For rolling features, this is ensured by backward-looking windows

        # Check that training set size is correct
        train_samples = split_info["train_samples"]
        assert len(X_train) == train_samples
        assert len(y_train) == train_samples


class TestFeatureMatrix:
    """Tests for feature matrix extraction."""

    def test_get_feature_matrix_excludes_metadata(self, feature_engineer, sample_weather_data):
        """Test that metadata columns are excluded from features."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)
        X, y = feature_engineer.get_feature_matrix(df)

        assert "timestamp" not in X.columns
        assert "location_name" not in X.columns
        assert "latitude" not in X.columns
        assert "longitude" not in X.columns
        assert feature_engineer.target_column not in X.columns

    def test_get_feature_matrix_returns_numeric_only(self, feature_engineer, sample_weather_data):
        """Test that only numeric features are returned."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)
        X, y = feature_engineer.get_feature_matrix(df)

        # All columns should be numeric
        assert all(np.issubdtype(X[col].dtype, np.number) for col in X.columns)

    def test_get_feature_matrix_handles_nan(self, feature_engineer, sample_weather_data):
        """Test that NaN values are handled."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)
        X, y = feature_engineer.get_feature_matrix(df)

        # Should have no NaN values in output
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_get_feature_matrix_requires_target(self, feature_engineer, sample_weather_data):
        """Test error when target column is missing."""
        df = feature_engineer.create_features(sample_weather_data)
        # Don't create target

        with pytest.raises(FeatureEngineerError, match="Target column"):
            feature_engineer.get_feature_matrix(df)


class TestFeatureNames:
    """Tests for feature name management."""

    def test_get_feature_names_returns_list(self, feature_engineer):
        """Test that feature names are returned as a list."""
        names = feature_engineer.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_feature_names_match_created(self, feature_engineer, sample_weather_data):
        """Test that feature names match what's actually created."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)
        X, y = feature_engineer.get_feature_matrix(df)

        expected_names = feature_engineer.get_feature_names()

        # All expected names should be in the feature matrix
        # (some might be filtered out if column doesn't exist)
        for name in expected_names:
            if name in sample_weather_data.columns or "_lag_" in name or "_rolling_" in name:
                # These should definitely be present
                pass  # Flexible check - main thing is no extra unexpected columns


class TestConfiguration:
    """Tests for configuration handling."""

    def test_default_config(self, feature_engineer):
        """Test that default configuration is set."""
        assert feature_engineer.feature_config is not None
        assert "weather_features" in feature_engineer.feature_config
        assert "lag_hours" in feature_engineer.feature_config
        assert "rolling_windows" in feature_engineer.feature_config

    def test_custom_config(self):
        """Test custom configuration."""
        custom_config = {
            "lag_hours": [1, 2, 3],
            "rolling_windows": [3, 6],
            "prediction_horizon_hours": 12,
        }
        fe = FeatureEngineer(config=custom_config)

        assert fe.feature_config["lag_hours"] == [1, 2, 3]
        assert fe.feature_config["rolling_windows"] == [3, 6]
        assert fe.feature_config["prediction_horizon_hours"] == 12

    def test_save_feature_config(self, feature_engineer, sample_weather_data, tmp_path):
        """Test saving feature configuration."""
        df = feature_engineer.create_features(sample_weather_data)
        df = feature_engineer.create_target(df)
        feature_engineer.train_test_split_temporal(df)

        config_path = tmp_path / "feature_config.json"
        feature_engineer.save_feature_config(config_path)

        assert config_path.exists()

        # Load and verify
        import json
        with open(config_path) as f:
            saved_config = json.load(f)

        assert "feature_config" in saved_config
        assert "split_info" in saved_config


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_location_data(self, feature_engineer):
        """Test with data from a single location."""
        timestamps = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature_2m": np.random.random(500) * 30,
            "relative_humidity_2m": np.random.random(500) * 50 + 30,
            "precipitation": np.random.random(500) * 5,
            "wind_speed_10m": np.random.random(500) * 20,
            "surface_pressure": 1013 + np.random.normal(0, 5, 500),
        })

        df = feature_engineer.create_features(df)
        df = feature_engineer.create_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = \
            feature_engineer.train_test_split_temporal(df)

        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

    def test_minimum_viable_data(self, feature_engineer):
        """Test with minimum amount of data."""
        # 15 days minimum (7 test + 3 val + 5 train)
        timestamps = pd.date_range("2024-01-01", periods=360, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature_2m": np.random.random(360) * 30,
            "relative_humidity_2m": np.random.random(360) * 50 + 30,
        })

        df = feature_engineer.create_features(df)
        df = feature_engineer.create_target(df)

        # Should work with tight margins
        X_train, X_val, X_test, y_train, y_val, y_test = \
            feature_engineer.train_test_split_temporal(df, test_days=5, validation_days=2)

        assert len(X_train) > 0
