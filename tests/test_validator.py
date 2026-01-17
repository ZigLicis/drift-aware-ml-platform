"""Tests for DataValidator.

Run with: pytest tests/test_validator.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.validator import (
    DataValidator,
    ValidationConfig,
    ValidationResult,
)


@pytest.fixture
def valid_weather_data() -> pd.DataFrame:
    """Create valid sample weather data."""
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=24,
        freq="h",
        tz="UTC"
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.random.uniform(-10, 30, 24),
        "humidity": np.random.uniform(30, 90, 24),
        "precipitation": np.random.uniform(0, 10, 24),
        "wind_speed": np.random.uniform(0, 50, 24),
        "pressure": np.random.uniform(990, 1030, 24),
    })


@pytest.fixture
def validator() -> DataValidator:
    """Create validator with default config."""
    return DataValidator()


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_str_valid(self):
        """Test string representation for valid result."""
        result = ValidationResult(
            is_valid=True,
            quality_score=0.95,
            total_records=100,
            valid_records=98,
            issues=[],
            anomaly_count=2,
        )
        str_repr = str(result)
        assert "VALID" in str_repr
        assert "95" in str_repr

    def test_str_invalid(self):
        """Test string representation for invalid result."""
        result = ValidationResult(
            is_valid=False,
            quality_score=0.65,
            total_records=100,
            valid_records=60,
            issues=["Missing columns"],
            anomaly_count=5,
        )
        str_repr = str(result)
        assert "INVALID" in str_repr


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_valid_data(self, validator, valid_weather_data):
        """Test validation of valid data."""
        result = validator.validate_weather_data(valid_weather_data)

        assert result.is_valid
        assert result.quality_score >= 0.8
        assert result.total_records == 24
        assert result.valid_records > 0
        assert len(result.issues) == 0

    def test_validate_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        result = validator.validate_weather_data(df)

        assert not result.is_valid
        assert result.quality_score == 0.0
        assert result.total_records == 0
        assert "empty" in result.issues[0].lower()

    def test_validate_missing_columns(self, validator):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "temperature": np.random.randn(10),
            # Missing: humidity, precipitation, wind_speed, pressure
        })

        result = validator.validate_weather_data(df)

        assert not result.is_valid
        assert any("missing" in issue.lower() for issue in result.issues)

    def test_validate_out_of_range_temperature(self, validator, valid_weather_data):
        """Test detection of out-of-range temperature values."""
        df = valid_weather_data.copy()
        df.loc[0, "temperature"] = 100  # Way above max (60°C)
        df.loc[1, "temperature"] = -100  # Way below min (-50°C)

        result = validator.validate_weather_data(df)

        assert any("temperature" in issue.lower() for issue in result.issues)
        assert result.details["range_validation"]["temperature"]["out_of_range_count"] == 2

    def test_validate_out_of_range_humidity(self, validator, valid_weather_data):
        """Test detection of out-of-range humidity values."""
        df = valid_weather_data.copy()
        df.loc[0, "humidity"] = 150  # Above 100%
        df.loc[1, "humidity"] = -10  # Below 0%

        result = validator.validate_weather_data(df)

        assert any("humidity" in issue.lower() for issue in result.issues)

    def test_validate_negative_precipitation(self, validator, valid_weather_data):
        """Test detection of negative precipitation."""
        df = valid_weather_data.copy()
        df.loc[0, "precipitation"] = -5  # Invalid

        result = validator.validate_weather_data(df)

        assert any("precipitation" in issue.lower() for issue in result.issues)

    def test_validate_out_of_range_pressure(self, validator, valid_weather_data):
        """Test detection of out-of-range pressure values."""
        df = valid_weather_data.copy()
        df.loc[0, "pressure"] = 800  # Below min (900)
        df.loc[1, "pressure"] = 1200  # Above max (1100)

        result = validator.validate_weather_data(df)

        assert any("pressure" in issue.lower() for issue in result.issues)

    def test_validate_with_nulls(self, validator, valid_weather_data):
        """Test validation handles null values."""
        df = valid_weather_data.copy()
        df.loc[0, "temperature"] = np.nan
        df.loc[1, "humidity"] = np.nan
        df.loc[2, "temperature"] = np.nan

        result = validator.validate_weather_data(df)

        # Should still be valid if nulls are below threshold
        assert "null_analysis" in result.details
        assert result.details["null_analysis"]["temperature"]["null_count"] == 2

    def test_validate_temporal_gaps(self, validator):
        """Test detection of temporal gaps."""
        # Create data with a large gap
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=10, freq="h", tz="UTC"))
        # Add a 5-hour gap
        timestamps.append(pd.Timestamp("2024-01-01 15:00", tz="UTC"))

        df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature": np.random.uniform(0, 20, len(timestamps)),
            "humidity": np.random.uniform(50, 80, len(timestamps)),
            "precipitation": np.random.uniform(0, 5, len(timestamps)),
            "wind_speed": np.random.uniform(0, 30, len(timestamps)),
            "pressure": np.random.uniform(1000, 1020, len(timestamps)),
        })

        result = validator.validate_weather_data(df)

        assert any("gap" in issue.lower() for issue in result.issues)
        assert result.details["temporal_analysis"]["gaps_found"] > 0


class TestQualityScore:
    """Tests for quality score calculation."""

    def test_perfect_quality_score(self, validator, valid_weather_data):
        """Test quality score for perfect data."""
        score = validator.calculate_quality_score(valid_weather_data)

        assert 0.9 <= score <= 1.0

    def test_quality_score_empty_df(self, validator):
        """Test quality score for empty DataFrame."""
        df = pd.DataFrame()
        score = validator.calculate_quality_score(df)

        assert score == 0.0

    def test_quality_score_with_nulls(self, validator, valid_weather_data):
        """Test quality score decreases with nulls."""
        df = valid_weather_data.copy()
        perfect_score = validator.calculate_quality_score(df)

        # Add significant nulls
        df.loc[0:5, "temperature"] = np.nan
        df.loc[0:5, "humidity"] = np.nan

        imperfect_score = validator.calculate_quality_score(df)

        assert imperfect_score < perfect_score

    def test_quality_score_breakdown(self, validator, valid_weather_data):
        """Test quality score breakdown components."""
        breakdown = validator._get_quality_breakdown(valid_weather_data)

        assert "completeness" in breakdown
        assert "validity" in breakdown
        assert "consistency" in breakdown
        assert all(0 <= v <= 1 for v in breakdown.values())


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    def test_flag_anomalies_normal_data(self, validator, valid_weather_data):
        """Test that normal data has few anomalies."""
        df_flagged = validator.flag_anomalies(valid_weather_data)

        assert "is_anomaly" in df_flagged.columns
        # With random normal data, should have very few anomalies
        anomaly_rate = df_flagged["is_anomaly"].mean()
        assert anomaly_rate < 0.2  # Less than 20% anomalies

    def test_flag_anomalies_with_outliers(self, validator, valid_weather_data):
        """Test detection of obvious outliers."""
        df = valid_weather_data.copy()

        # Add obvious outliers
        df.loc[0, "temperature"] = 1000  # Extreme outlier
        df.loc[1, "pressure"] = 0  # Extreme outlier

        df_flagged = validator.flag_anomalies(df)

        assert df_flagged.loc[0, "is_anomaly"]
        assert df_flagged.loc[1, "is_anomaly"]

    def test_flag_anomalies_columns(self, validator, valid_weather_data):
        """Test that per-column anomaly flags are created."""
        df_flagged = validator.flag_anomalies(valid_weather_data)

        assert "temperature_anomaly" in df_flagged.columns
        assert "humidity_anomaly" in df_flagged.columns
        assert "pressure_anomaly" in df_flagged.columns


class TestValidationConfig:
    """Tests for custom validation configuration."""

    def test_custom_config(self):
        """Test validator with custom configuration."""
        config = ValidationConfig(
            temperature_min=-20,
            temperature_max=40,
            min_quality_score=0.9,
        )
        validator = DataValidator(config=config)

        assert validator.config.temperature_min == -20
        assert validator.config.temperature_max == 40
        assert validator.config.min_quality_score == 0.9

    def test_stricter_threshold(self, valid_weather_data):
        """Test that stricter threshold affects validity."""
        lenient_config = ValidationConfig(min_quality_score=0.5)
        strict_config = ValidationConfig(min_quality_score=0.99)

        lenient_validator = DataValidator(lenient_config)
        strict_validator = DataValidator(strict_config)

        # Add some issues to the data
        df = valid_weather_data.copy()
        df.loc[0:2, "temperature"] = np.nan

        lenient_result = lenient_validator.validate_weather_data(df)
        strict_result = strict_validator.validate_weather_data(df)

        # Lenient should pass, strict might fail
        assert lenient_result.is_valid
        # Quality score should be the same, validity might differ
        assert lenient_result.quality_score == strict_result.quality_score


class TestValidationReport:
    """Tests for validation report generation."""

    def test_generate_report(self, validator, valid_weather_data):
        """Test report generation."""
        result = validator.validate_weather_data(valid_weather_data)
        report = validator.get_validation_report(result)

        assert "VALIDATION REPORT" in report
        assert "Quality Score" in report
        assert "Total Records" in report

    def test_report_with_issues(self, validator, valid_weather_data):
        """Test report includes issues."""
        df = valid_weather_data.copy()
        df.loc[0, "temperature"] = 100  # Out of range

        result = validator.validate_weather_data(df)
        report = validator.get_validation_report(result)

        assert "Issues Found" in report
        assert "temperature" in report.lower()
