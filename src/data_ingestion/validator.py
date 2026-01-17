"""Data validation module for weather data quality assurance.

This module provides comprehensive validation for weather data including
range checks, anomaly detection, and quality scoring.

Example:
    >>> from src.data_ingestion.validator import DataValidator
    >>> validator = DataValidator()
    >>> result = validator.validate_weather_data(df)
    >>> print(f"Quality Score: {result.quality_score:.2%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation operation.

    Attributes:
        is_valid: Whether the data passes minimum quality thresholds.
        quality_score: Overall quality score from 0.0 to 1.0.
        total_records: Total number of records validated.
        valid_records: Number of records passing all validation rules.
        issues: List of validation issues found.
        anomaly_count: Number of records flagged as anomalies.
        details: Additional validation details and statistics.
    """

    is_valid: bool
    quality_score: float
    total_records: int
    valid_records: int
    issues: list[str] = field(default_factory=list)
    anomaly_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return (
            f"ValidationResult({status}, "
            f"score={self.quality_score:.2%}, "
            f"records={self.valid_records}/{self.total_records}, "
            f"anomalies={self.anomaly_count}, "
            f"issues={len(self.issues)})"
        )


@dataclass
class ValidationConfig:
    """Configuration for data validation rules.

    Attributes:
        temperature_min: Minimum valid temperature in °C.
        temperature_max: Maximum valid temperature in °C.
        humidity_min: Minimum valid humidity percentage.
        humidity_max: Maximum valid humidity percentage.
        precipitation_min: Minimum valid precipitation in mm.
        precipitation_max: Maximum valid precipitation in mm.
        wind_speed_min: Minimum valid wind speed in km/h.
        wind_speed_max: Maximum valid wind speed in km/h.
        pressure_min: Minimum valid pressure in hPa.
        pressure_max: Maximum valid pressure in hPa.
        max_time_gap_hours: Maximum allowed gap between consecutive records.
        anomaly_std_threshold: Number of standard deviations for anomaly detection.
        min_quality_score: Minimum quality score for data to be considered valid.
    """

    # Temperature bounds (°C)
    temperature_min: float = -50.0
    temperature_max: float = 60.0

    # Humidity bounds (%)
    humidity_min: float = 0.0
    humidity_max: float = 100.0

    # Precipitation bounds (mm)
    precipitation_min: float = 0.0
    precipitation_max: float = 500.0

    # Wind speed bounds (km/h)
    wind_speed_min: float = 0.0
    wind_speed_max: float = 400.0

    # Pressure bounds (hPa)
    pressure_min: float = 900.0
    pressure_max: float = 1100.0

    # Temporal consistency
    max_time_gap_hours: float = 2.0

    # Anomaly detection
    anomaly_std_threshold: float = 3.0

    # Quality thresholds
    min_quality_score: float = 0.8

    # Quality score weights
    completeness_weight: float = 0.4
    validity_weight: float = 0.4
    consistency_weight: float = 0.2


class DataValidator:
    """Validator for weather data quality assurance.

    Provides comprehensive validation including range checks, null detection,
    anomaly flagging, and quality scoring.

    Attributes:
        config: Validation configuration with thresholds and bounds.

    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate_weather_data(df)
        >>> if result.is_valid:
        ...     print("Data quality acceptable")
        ... else:
        ...     for issue in result.issues:
        ...         print(f"Issue: {issue}")
    """

    # Required columns for weather data
    REQUIRED_COLUMNS = [
        "timestamp",
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
        "pressure",
    ]

    # Numeric columns for validation
    NUMERIC_COLUMNS = [
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
        "pressure",
    ]

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize the validator with configuration.

        Args:
            config: Validation configuration. Uses defaults if not provided.
        """
        self.config = config or ValidationConfig()
        logger.info("Initialized DataValidator with config")

    def validate_weather_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate weather data comprehensively.

        Performs the following checks:
        1. Required columns exist
        2. Data types are correct
        3. Values are within expected ranges
        4. Null value assessment
        5. Temporal consistency (no large gaps)
        6. Anomaly detection

        Args:
            df: DataFrame with weather data to validate.

        Returns:
            ValidationResult with validation status and details.

        Example:
            >>> validator = DataValidator()
            >>> result = validator.validate_weather_data(df)
            >>> print(result)
            ValidationResult(✓ VALID, score=95.20%, records=168/168, anomalies=2)
        """
        issues: list[str] = []
        details: dict[str, Any] = {}

        if df.empty:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                total_records=0,
                valid_records=0,
                issues=["DataFrame is empty"],
                anomaly_count=0,
            )

        total_records = len(df)
        logger.info(f"Validating {total_records} weather records")

        # 1. Check required columns
        column_issues = self._validate_columns(df)
        issues.extend(column_issues)

        if column_issues:
            # Can't proceed without required columns
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                total_records=total_records,
                valid_records=0,
                issues=issues,
                anomaly_count=0,
            )

        # 2. Validate data types
        type_issues = self._validate_data_types(df)
        issues.extend(type_issues)

        # 3. Check value ranges
        range_results = self._validate_ranges(df)
        issues.extend(range_results["issues"])
        details["range_validation"] = range_results["details"]

        # 4. Check for nulls
        null_results = self._check_nulls(df)
        issues.extend(null_results["issues"])
        details["null_analysis"] = null_results["details"]

        # 5. Check temporal consistency
        temporal_results = self._check_temporal_consistency(df)
        issues.extend(temporal_results["issues"])
        details["temporal_analysis"] = temporal_results["details"]

        # 6. Detect anomalies
        anomaly_count = self._count_anomalies(df)
        details["anomaly_count"] = anomaly_count

        # Calculate quality score
        quality_score = self.calculate_quality_score(df)
        details["quality_score_breakdown"] = self._get_quality_breakdown(df)

        # Determine validity
        valid_records = range_results["valid_count"]
        is_valid = (
            quality_score >= self.config.min_quality_score
            and len(column_issues) == 0
            and len(type_issues) == 0
        )

        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            total_records=total_records,
            valid_records=valid_records,
            issues=issues,
            anomaly_count=anomaly_count,
            details=details,
        )

        logger.info(f"Validation complete: {result}")
        return result

    def _validate_columns(self, df: pd.DataFrame) -> list[str]:
        """Check that required columns exist."""
        issues = []
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            issues.append(f"Missing required columns: {sorted(missing)}")
            logger.error(f"Missing columns: {missing}")

        return issues

    def _validate_data_types(self, df: pd.DataFrame) -> list[str]:
        """Validate data types for each column."""
        issues = []

        # Check timestamp is datetime
        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                issues.append("'timestamp' column is not datetime type")

        # Check numeric columns
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"'{col}' column is not numeric type")

        return issues

    def _validate_ranges(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate that values are within expected ranges."""
        issues = []
        details = {}
        invalid_mask = pd.Series(False, index=df.index)

        # Define range checks
        range_checks = [
            ("temperature", self.config.temperature_min, self.config.temperature_max),
            ("humidity", self.config.humidity_min, self.config.humidity_max),
            ("precipitation", self.config.precipitation_min, self.config.precipitation_max),
            ("wind_speed", self.config.wind_speed_min, self.config.wind_speed_max),
            ("pressure", self.config.pressure_min, self.config.pressure_max),
        ]

        for col, min_val, max_val in range_checks:
            if col not in df.columns:
                continue

            col_data = df[col].dropna()
            below_min = (col_data < min_val).sum()
            above_max = (col_data > max_val).sum()
            out_of_range = below_min + above_max

            details[col] = {
                "min": col_data.min() if len(col_data) > 0 else None,
                "max": col_data.max() if len(col_data) > 0 else None,
                "expected_min": min_val,
                "expected_max": max_val,
                "below_min_count": int(below_min),
                "above_max_count": int(above_max),
                "out_of_range_count": int(out_of_range),
            }

            if out_of_range > 0:
                pct = out_of_range / len(df) * 100
                issues.append(
                    f"{col}: {out_of_range} values ({pct:.1f}%) out of range "
                    f"[{min_val}, {max_val}]"
                )
                invalid_mask |= (df[col] < min_val) | (df[col] > max_val)

        valid_count = (~invalid_mask).sum()

        return {
            "issues": issues,
            "details": details,
            "valid_count": int(valid_count),
        }

    def _check_nulls(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check for null values in the dataset."""
        issues = []
        details = {}

        total_records = len(df)

        for col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = null_count / total_records * 100 if total_records > 0 else 0

            details[col] = {
                "null_count": int(null_count),
                "null_percentage": round(null_pct, 2),
            }

            if null_count > 0:
                # Only add as issue if significant (> 5%)
                if null_pct > 5:
                    issues.append(
                        f"{col}: {null_count} null values ({null_pct:.1f}%)"
                    )
                logger.debug(f"{col}: {null_count} nulls ({null_pct:.1f}%)")

        return {"issues": issues, "details": details}

    def _check_temporal_consistency(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check for gaps in time series data."""
        issues = []
        details = {}

        if "timestamp" not in df.columns or len(df) < 2:
            return {"issues": [], "details": {"checked": False}}

        # Sort by timestamp
        df_sorted = df.sort_values("timestamp")
        time_diffs = df_sorted["timestamp"].diff()

        # Convert to hours
        time_diffs_hours = time_diffs.dt.total_seconds() / 3600

        # Find gaps larger than threshold
        max_gap = self.config.max_time_gap_hours
        gaps = time_diffs_hours[time_diffs_hours > max_gap]

        details = {
            "checked": True,
            "total_intervals": len(time_diffs) - 1,
            "gaps_found": len(gaps),
            "max_gap_hours": float(time_diffs_hours.max()) if len(time_diffs_hours) > 1 else 0,
            "expected_interval_hours": 1.0,
            "gap_threshold_hours": max_gap,
        }

        if len(gaps) > 0:
            issues.append(
                f"Found {len(gaps)} time gaps > {max_gap} hours "
                f"(max gap: {details['max_gap_hours']:.1f} hours)"
            )

        return {"issues": issues, "details": details}

    def _count_anomalies(self, df: pd.DataFrame) -> int:
        """Count anomalies using IQR method."""
        anomaly_count = 0

        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue

            col_data = df[col].dropna()
            if len(col_data) < 4:
                continue

            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            col_anomalies = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            anomaly_count += col_anomalies

        return int(anomaly_count)

    def calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score.

        Score is weighted average of:
        - Completeness: Percentage of non-null values
        - Validity: Percentage of values within expected ranges
        - Consistency: Temporal consistency (no large gaps)

        Args:
            df: DataFrame with weather data.

        Returns:
            Quality score from 0.0 to 1.0.

        Example:
            >>> validator = DataValidator()
            >>> score = validator.calculate_quality_score(df)
            >>> print(f"Quality: {score:.2%}")
            Quality: 94.50%
        """
        if df.empty:
            return 0.0

        breakdown = self._get_quality_breakdown(df)

        # Weighted average
        score = (
            breakdown["completeness"] * self.config.completeness_weight
            + breakdown["validity"] * self.config.validity_weight
            + breakdown["consistency"] * self.config.consistency_weight
        )

        return round(score, 4)

    def _get_quality_breakdown(self, df: pd.DataFrame) -> dict[str, float]:
        """Get breakdown of quality score components."""
        # Completeness: % non-null across numeric columns
        completeness_scores = []
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                non_null_pct = 1 - (df[col].isna().sum() / len(df))
                completeness_scores.append(non_null_pct)
        completeness = np.mean(completeness_scores) if completeness_scores else 0.0

        # Validity: % values within expected ranges
        validity_scores = []
        range_checks = [
            ("temperature", self.config.temperature_min, self.config.temperature_max),
            ("humidity", self.config.humidity_min, self.config.humidity_max),
            ("precipitation", self.config.precipitation_min, self.config.precipitation_max),
            ("wind_speed", self.config.wind_speed_min, self.config.wind_speed_max),
            ("pressure", self.config.pressure_min, self.config.pressure_max),
        ]
        for col, min_val, max_val in range_checks:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    valid_pct = ((col_data >= min_val) & (col_data <= max_val)).mean()
                    validity_scores.append(valid_pct)
        validity = np.mean(validity_scores) if validity_scores else 0.0

        # Consistency: Based on time gaps
        consistency = 1.0
        if "timestamp" in df.columns and len(df) >= 2:
            df_sorted = df.sort_values("timestamp")
            time_diffs = df_sorted["timestamp"].diff().dt.total_seconds() / 3600
            # Penalize for gaps > threshold
            gaps = time_diffs[time_diffs > self.config.max_time_gap_hours]
            if len(gaps) > 0:
                # Reduce score based on number of gaps relative to total intervals
                gap_ratio = len(gaps) / (len(df) - 1)
                consistency = max(0.0, 1.0 - gap_ratio)

        return {
            "completeness": float(completeness),
            "validity": float(validity),
            "consistency": float(consistency),
        }

    def flag_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly flags to DataFrame.

        Flags records with values more than N standard deviations from the mean,
        using the threshold from config.

        Args:
            df: DataFrame with weather data.

        Returns:
            DataFrame with added 'is_anomaly' and per-column anomaly columns.

        Example:
            >>> validator = DataValidator()
            >>> df_flagged = validator.flag_anomalies(df)
            >>> anomalies = df_flagged[df_flagged['is_anomaly']]
            >>> print(f"Found {len(anomalies)} anomalous records")
        """
        df = df.copy()
        threshold = self.config.anomaly_std_threshold

        # Initialize anomaly flag
        df["is_anomaly"] = False

        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue

            col_data = df[col]
            mean = col_data.mean()
            std = col_data.std()

            if std > 0:
                z_scores = np.abs((col_data - mean) / std)
                col_anomaly = z_scores > threshold
                df[f"{col}_anomaly"] = col_anomaly
                df["is_anomaly"] |= col_anomaly
            else:
                df[f"{col}_anomaly"] = False

        anomaly_count = df["is_anomaly"].sum()
        logger.info(
            f"Flagged {anomaly_count} anomalies ({anomaly_count / len(df) * 100:.1f}%)"
        )

        return df

    def get_validation_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report.

        Args:
            result: ValidationResult from validate_weather_data().

        Returns:
            Formatted string report.
        """
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            "",
            f"Status: {'✓ VALID' if result.is_valid else '✗ INVALID'}",
            f"Quality Score: {result.quality_score:.2%}",
            f"Total Records: {result.total_records:,}",
            f"Valid Records: {result.valid_records:,}",
            f"Anomalies Detected: {result.anomaly_count:,}",
            "",
        ]

        if result.issues:
            lines.append("Issues Found:")
            for issue in result.issues:
                lines.append(f"  • {issue}")
            lines.append("")

        if "quality_score_breakdown" in result.details:
            breakdown = result.details["quality_score_breakdown"]
            lines.extend([
                "Quality Score Breakdown:",
                f"  • Completeness: {breakdown.get('completeness', 0):.2%}",
                f"  • Validity: {breakdown.get('validity', 0):.2%}",
                f"  • Consistency: {breakdown.get('consistency', 0):.2%}",
                "",
            ])

        lines.append("=" * 60)

        return "\n".join(lines)
