"""Weather data ingestion pipeline orchestration.

This module provides the main pipeline class that orchestrates all
data ingestion components: fetching, validation, transformation, and storage.

Example:
    >>> from src.data_ingestion.pipeline import WeatherDataPipeline
    >>> pipeline = WeatherDataPipeline(
    ...     config_path="config/data_config.yaml",
    ...     db_connection_string="postgresql://user:pass@localhost/db"
    ... )
    >>> result = pipeline.run_incremental_ingestion()
    >>> print(f"Stored {result.records_stored} records")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import mlflow
import pandas as pd
import yaml

from src.data_ingestion.weather_client import WeatherAPIClient
from src.data_ingestion.validator import DataValidator, ValidationResult
from src.data_ingestion.transformer import DataTransformer
from src.data_ingestion.storage import DataStorage, DataStorageError

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of a data ingestion operation.

    Attributes:
        success: Whether the ingestion completed successfully.
        records_fetched: Number of records fetched from API.
        records_stored: Number of records stored in database.
        quality_score: Data quality score from validation.
        execution_time_seconds: Total execution time.
        batch_id: Unique identifier for this batch.
        error_message: Error details if failed.
        validation_result: Full validation result details.
        metadata: Additional metadata about the run.
    """

    success: bool
    records_fetched: int
    records_stored: int
    quality_score: float
    execution_time_seconds: float
    batch_id: str
    error_message: str | None = None
    validation_result: ValidationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        return (
            f"IngestionResult({status}, "
            f"fetched={self.records_fetched}, "
            f"stored={self.records_stored}, "
            f"quality={self.quality_score:.2%}, "
            f"time={self.execution_time_seconds:.1f}s)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Remove non-serializable validation_result
        result.pop("validation_result", None)
        return result


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline.

    Attributes:
        api_base_url: Base URL for weather API.
        api_timeout: Request timeout in seconds.
        api_retry_attempts: Number of retry attempts.
        location_name: Human-readable location name.
        latitude: Location latitude.
        longitude: Location longitude.
        timezone: Timezone for the location.
        features: Weather features to collect.
        min_quality_score: Minimum quality score for data acceptance.
        mlflow_experiment_name: Name of MLflow experiment.
        mlflow_tracking_uri: MLflow tracking server URI.
    """

    api_base_url: str = "https://api.open-meteo.com/v1/forecast"
    api_timeout: int = 30
    api_retry_attempts: int = 3
    location_name: str = "Washington DC"
    latitude: float = 38.9072
    longitude: float = -77.0369
    timezone: str = "America/New_York"
    features: list[str] = field(default_factory=lambda: [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "pressure_msl",
    ])
    min_quality_score: float = 0.8
    mlflow_experiment_name: str = "data-ingestion"
    mlflow_tracking_uri: str | None = None

    @classmethod
    def from_yaml(cls, config_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        api_config = data.get("api", {})
        location_config = data.get("location", {})
        features_config = data.get("features", {})

        return cls(
            api_base_url=api_config.get("base_url", cls.api_base_url),
            api_timeout=api_config.get("timeout", cls.api_timeout),
            api_retry_attempts=api_config.get("retry_attempts", cls.api_retry_attempts),
            location_name=location_config.get("name", cls.location_name),
            latitude=location_config.get("latitude", cls.latitude),
            longitude=location_config.get("longitude", cls.longitude),
            timezone=location_config.get("timezone", cls.timezone),
            features=features_config.get("hourly", cls.features),
            min_quality_score=data.get("data_quality", {}).get(
                "min_completeness", cls.min_quality_score
            ),
        )


class WeatherDataPipeline:
    """Orchestrates the complete weather data ingestion pipeline.

    Coordinates fetching, validation, transformation, and storage of
    weather data with MLflow experiment tracking.

    Attributes:
        config: Pipeline configuration.
        client: Weather API client.
        validator: Data validator.
        transformer: Data transformer.
        storage: Database storage.

    Example:
        >>> pipeline = WeatherDataPipeline(
        ...     config_path="config/data_config.yaml",
        ...     db_connection_string="postgresql://user:pass@localhost/db"
        ... )
        >>> # Run historical ingestion
        >>> result = pipeline.run_historical_ingestion("2024-01-01", "2024-01-31")
        >>> print(f"Quality: {result.quality_score:.2%}")
        >>>
        >>> # Run incremental ingestion
        >>> result = pipeline.run_incremental_ingestion()
    """

    def __init__(
        self,
        config_path: str | None = None,
        db_connection_string: str | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the pipeline with all components.

        Args:
            config_path: Path to YAML configuration file.
            db_connection_string: PostgreSQL connection string.
            config: Direct configuration object (overrides config_path).

        Raises:
            ValueError: If neither config_path nor config is provided.
            FileNotFoundError: If config file doesn't exist.
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            self.config = PipelineConfig.from_yaml(config_path)
        else:
            self.config = PipelineConfig()

        self.db_connection_string = db_connection_string

        # Initialize components
        self._init_components()

        # Setup MLflow
        self._setup_mlflow()

        logger.info(
            f"Initialized WeatherDataPipeline for {self.config.location_name}"
        )

    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Weather API client
        self.client = WeatherAPIClient(
            base_url=self.config.api_base_url,
            latitude=self.config.latitude,
            longitude=self.config.longitude,
            timezone=self.config.timezone,
            timeout=self.config.api_timeout,
            retry_attempts=self.config.api_retry_attempts,
            location_name=self.config.location_name,
        )

        # Validator
        self.validator = DataValidator()

        # Transformer
        self.transformer = DataTransformer(
            default_timezone=self.config.timezone
        )

        # Storage (optional - can run without DB)
        self.storage: DataStorage | None = None
        if self.db_connection_string:
            try:
                self.storage = DataStorage(self.db_connection_string)
                logger.info("Database storage initialized")
            except DataStorageError as e:
                logger.warning(f"Database unavailable: {e}. Running without storage.")

    def _setup_mlflow(self) -> None:
        """Configure MLflow experiment tracking."""
        tracking_uri = (
            self.config.mlflow_tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI")
        )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(
                self.config.mlflow_experiment_name
            )
            if experiment is None:
                mlflow.create_experiment(self.config.mlflow_experiment_name)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            logger.info(f"MLflow experiment: {self.config.mlflow_experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Continuing without tracking.")

    def _generate_batch_id(self) -> str:
        """Generate unique batch identifier."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"batch_{timestamp}"

    def run_historical_ingestion(
        self,
        start_date: str,
        end_date: str,
        skip_mlflow: bool = False,
    ) -> IngestionResult:
        """Run historical data ingestion for a date range.

        Fetches, validates, transforms, and stores historical weather data.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            skip_mlflow: If True, skip MLflow logging.

        Returns:
            IngestionResult with details of the operation.

        Example:
            >>> result = pipeline.run_historical_ingestion("2024-01-01", "2024-01-31")
            >>> if result.success:
            ...     print(f"Ingested {result.records_stored} records")
        """
        batch_id = self._generate_batch_id()
        start_time = time.time()

        logger.info(
            f"Starting historical ingestion: {start_date} to {end_date} "
            f"(batch: {batch_id})"
        )

        try:
            # Step 1: Fetch data
            logger.info("Step 1/4: Fetching data from API...")
            df_raw = self.client.fetch_historical(start_date, end_date)
            records_fetched = len(df_raw)
            logger.info(f"Fetched {records_fetched} records")

            if records_fetched == 0:
                return IngestionResult(
                    success=False,
                    records_fetched=0,
                    records_stored=0,
                    quality_score=0.0,
                    execution_time_seconds=time.time() - start_time,
                    batch_id=batch_id,
                    error_message="No data returned from API",
                )

            # Step 2: Validate data
            logger.info("Step 2/4: Validating data quality...")
            validation_result = self.validator.validate_weather_data(df_raw)
            logger.info(f"Validation: {validation_result}")

            if not validation_result.is_valid:
                logger.warning(
                    f"Data failed validation with score {validation_result.quality_score:.2%}"
                )

            # Step 3: Transform data
            logger.info("Step 3/4: Transforming data...")
            df_transformed = self.transformer.transform_for_storage(
                df_raw,
                data_source="open-meteo-api",
                batch_id=batch_id,
            )

            # Step 4: Store data
            records_stored = 0
            if self.storage:
                logger.info("Step 4/4: Storing data in database...")
                records_stored = self.storage.store_weather_data(
                    df_transformed,
                    batch_id=batch_id,
                )
                logger.info(f"Stored {records_stored} records")

                # Log ingestion metadata
                self.storage.log_ingestion_metadata(
                    batch_id=batch_id,
                    metadata={
                        "start_time": datetime.now(timezone.utc).isoformat(),
                        "end_time": datetime.now(timezone.utc).isoformat(),
                        "records_count": records_stored,
                        "quality_score": validation_result.quality_score,
                        "status": "success",
                        "date_range": f"{start_date} to {end_date}",
                    },
                )
            else:
                logger.info("Step 4/4: Skipping storage (no database configured)")

            execution_time = time.time() - start_time

            result = IngestionResult(
                success=True,
                records_fetched=records_fetched,
                records_stored=records_stored,
                quality_score=validation_result.quality_score,
                execution_time_seconds=execution_time,
                batch_id=batch_id,
                validation_result=validation_result,
                metadata={
                    "start_date": start_date,
                    "end_date": end_date,
                    "location": self.config.location_name,
                    "ingestion_type": "historical",
                },
            )

            # Log to MLflow
            if not skip_mlflow:
                self._log_to_mlflow(result, df_raw, validation_result)

            logger.info(f"Historical ingestion complete: {result}")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.exception(f"Historical ingestion failed: {error_msg}")

            return IngestionResult(
                success=False,
                records_fetched=0,
                records_stored=0,
                quality_score=0.0,
                execution_time_seconds=execution_time,
                batch_id=batch_id,
                error_message=error_msg,
            )

    def run_incremental_ingestion(
        self,
        hours: int = 24,
        skip_mlflow: bool = False,
    ) -> IngestionResult:
        """Run incremental data ingestion.

        Fetches data since the last stored timestamp, or the last N hours
        if no data exists.

        Args:
            hours: Hours of data to fetch if no existing data.
            skip_mlflow: If True, skip MLflow logging.

        Returns:
            IngestionResult with details of the operation.

        Example:
            >>> result = pipeline.run_incremental_ingestion()
            >>> print(f"Added {result.records_stored} new records")
        """
        batch_id = self._generate_batch_id()
        start_time = time.time()

        logger.info(f"Starting incremental ingestion (batch: {batch_id})")

        try:
            # Determine start point
            if self.storage:
                latest_ts = self.storage.get_latest_timestamp()
                if latest_ts:
                    # Fetch from last timestamp (with 1 hour overlap for safety)
                    start_dt = latest_ts - timedelta(hours=1)
                    logger.info(f"Last data timestamp: {latest_ts}")
                else:
                    # No existing data, use default lookback
                    start_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
                    logger.info(f"No existing data, fetching last {hours} hours")
            else:
                start_dt = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Calculate date strings
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Run the ingestion
            result = self.run_historical_ingestion(
                start_date=start_date,
                end_date=end_date,
                skip_mlflow=True,  # We'll log separately
            )

            # Update metadata
            result.metadata["ingestion_type"] = "incremental"
            result.metadata["hours_requested"] = hours

            # Log to MLflow with incremental tags
            if not skip_mlflow and result.validation_result:
                self._log_to_mlflow(
                    result,
                    None,  # Don't re-log data
                    result.validation_result,
                    tags={"ingestion_type": "incremental"},
                )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.exception(f"Incremental ingestion failed: {error_msg}")

            return IngestionResult(
                success=False,
                records_fetched=0,
                records_stored=0,
                quality_score=0.0,
                execution_time_seconds=execution_time,
                batch_id=batch_id,
                error_message=error_msg,
            )

    def backfill(
        self,
        start_date: str,
        end_date: str,
        batch_days: int = 7,
        progress_callback: Callable[[int, int, IngestionResult], None] | None = None,
    ) -> list[IngestionResult]:
        """Run backfill ingestion in batches.

        For large historical loads, breaks the date range into smaller
        batches to avoid timeouts and enable progress tracking.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            batch_days: Days per batch.
            progress_callback: Optional callback(batch_num, total_batches, result).

        Returns:
            List of IngestionResult for each batch.

        Example:
            >>> results = pipeline.backfill(
            ...     "2024-01-01",
            ...     "2024-12-31",
            ...     batch_days=7
            ... )
            >>> total_stored = sum(r.records_stored for r in results)
            >>> print(f"Backfilled {total_stored} records in {len(results)} batches")
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculate batches
        batches: list[tuple[str, str]] = []
        current = start_dt
        while current <= end_dt:
            batch_end = min(current + timedelta(days=batch_days - 1), end_dt)
            batches.append((
                current.strftime("%Y-%m-%d"),
                batch_end.strftime("%Y-%m-%d"),
            ))
            current = batch_end + timedelta(days=1)

        logger.info(
            f"Starting backfill: {start_date} to {end_date} "
            f"({len(batches)} batches of {batch_days} days)"
        )

        results: list[IngestionResult] = []

        for i, (batch_start, batch_end) in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{len(batches)}: {batch_start} to {batch_end}")

            result = self.run_historical_ingestion(
                start_date=batch_start,
                end_date=batch_end,
                skip_mlflow=True,  # Log summary at end
            )
            results.append(result)

            if progress_callback:
                progress_callback(i, len(batches), result)

            if not result.success:
                logger.error(f"Batch {i} failed: {result.error_message}")
                # Continue with next batch instead of stopping

            # Small delay between batches to be nice to the API
            if i < len(batches):
                time.sleep(0.5)

        # Log summary to MLflow
        self._log_backfill_summary(results, start_date, end_date)

        # Summary
        successful = sum(1 for r in results if r.success)
        total_fetched = sum(r.records_fetched for r in results)
        total_stored = sum(r.records_stored for r in results)

        logger.info(
            f"Backfill complete: {successful}/{len(batches)} batches successful, "
            f"{total_fetched} fetched, {total_stored} stored"
        )

        return results

    def _log_to_mlflow(
        self,
        result: IngestionResult,
        df: pd.DataFrame | None,
        validation_result: ValidationResult,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log ingestion results to MLflow."""
        try:
            with mlflow.start_run(run_name=result.batch_id):
                # Log parameters
                mlflow.log_params({
                    "location": self.config.location_name,
                    "latitude": self.config.latitude,
                    "longitude": self.config.longitude,
                    "start_date": result.metadata.get("start_date", ""),
                    "end_date": result.metadata.get("end_date", ""),
                    "ingestion_type": result.metadata.get("ingestion_type", "unknown"),
                })

                # Log metrics
                mlflow.log_metrics({
                    "records_fetched": result.records_fetched,
                    "records_stored": result.records_stored,
                    "quality_score": result.quality_score,
                    "execution_time_seconds": result.execution_time_seconds,
                    "validation_pass": 1 if validation_result.is_valid else 0,
                    "anomaly_count": validation_result.anomaly_count,
                })

                # Log tags
                mlflow.set_tags({
                    "batch_id": result.batch_id,
                    "success": str(result.success),
                    **(tags or {}),
                })

                # Log validation report as artifact
                report = self.validator.get_validation_report(validation_result)
                report_path = f"/tmp/validation_report_{result.batch_id}.txt"
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path)

                # Log data summary if available
                if df is not None and len(df) > 0:
                    summary_path = f"/tmp/data_summary_{result.batch_id}.json"
                    summary = {
                        "record_count": len(df),
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "null_counts": df.isnull().sum().to_dict(),
                    }
                    # Add numeric column stats
                    for col in df.select_dtypes(include=["number"]).columns:
                        summary[f"{col}_stats"] = {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                        }
                    with open(summary_path, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    mlflow.log_artifact(summary_path)

                logger.debug(f"Logged run to MLflow: {result.batch_id}")

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def _log_backfill_summary(
        self,
        results: list[IngestionResult],
        start_date: str,
        end_date: str,
    ) -> None:
        """Log backfill summary to MLflow."""
        try:
            with mlflow.start_run(run_name=f"backfill_{start_date}_{end_date}"):
                successful = sum(1 for r in results if r.success)
                total_fetched = sum(r.records_fetched for r in results)
                total_stored = sum(r.records_stored for r in results)
                avg_quality = (
                    sum(r.quality_score for r in results) / len(results)
                    if results else 0
                )
                total_time = sum(r.execution_time_seconds for r in results)

                mlflow.log_params({
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_batches": len(results),
                    "ingestion_type": "backfill",
                })

                mlflow.log_metrics({
                    "successful_batches": successful,
                    "failed_batches": len(results) - successful,
                    "total_records_fetched": total_fetched,
                    "total_records_stored": total_stored,
                    "average_quality_score": avg_quality,
                    "total_execution_time": total_time,
                })

                mlflow.set_tags({
                    "backfill": "true",
                    "success": str(successful == len(results)),
                })

        except Exception as e:
            logger.warning(f"Failed to log backfill summary to MLflow: {e}")

    def health_check(self) -> dict[str, bool]:
        """Check health of all pipeline components.

        Returns:
            Dictionary with component health status.
        """
        health = {
            "api": False,
            "database": False,
            "mlflow": False,
        }

        # Check API
        try:
            health["api"] = self.client.health_check()
        except Exception as e:
            logger.error(f"API health check failed: {e}")

        # Check database
        if self.storage:
            try:
                health["database"] = self.storage.health_check()
            except Exception as e:
                logger.error(f"Database health check failed: {e}")

        # Check MLflow
        try:
            mlflow.get_tracking_uri()
            health["mlflow"] = True
        except Exception as e:
            logger.error(f"MLflow health check failed: {e}")

        return health

    def close(self) -> None:
        """Close all connections."""
        self.client.close()
        if self.storage:
            self.storage.close()
        logger.info("Pipeline connections closed")

    def __enter__(self) -> WeatherDataPipeline:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
