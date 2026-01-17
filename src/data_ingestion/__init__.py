"""Data ingestion module for Domain-Shift ML Platform.

This module provides components for fetching, validating, and storing
weather data from external APIs.

Components:
    - WeatherAPIClient: Client for Open-Meteo weather API
    - DataValidator: Validates incoming weather data
    - DataTransformer: Transforms raw API data to database format
    - DataStorage: Persists weather data to PostgreSQL
    - WeatherDataPipeline: Orchestrates the complete ingestion workflow

Example:
    >>> from src.data_ingestion import WeatherDataPipeline
    >>> # Using the pipeline (recommended)
    >>> pipeline = WeatherDataPipeline(
    ...     config_path="config/data_config.yaml",
    ...     db_connection_string="postgresql://user:pass@localhost/db"
    ... )
    >>> result = pipeline.run_incremental_ingestion()
    >>> print(f"Stored {result.records_stored} records")
    >>>
    >>> # Or use components individually
    >>> from src.data_ingestion import (
    ...     WeatherAPIClient,
    ...     DataValidator,
    ...     DataTransformer,
    ...     DataStorage,
    ... )
    >>> client = WeatherAPIClient()
    >>> df = client.fetch_latest(hours=24)
    >>> validator = DataValidator()
    >>> result = validator.validate_weather_data(df)
"""

from src.data_ingestion.weather_client import WeatherAPIClient
from src.data_ingestion.validator import (
    DataValidator,
    ValidationConfig,
    ValidationResult,
)
from src.data_ingestion.transformer import DataTransformer
from src.data_ingestion.storage import DataStorage, DataStorageError
from src.data_ingestion.pipeline import (
    WeatherDataPipeline,
    IngestionResult,
    PipelineConfig,
)

__all__ = [
    # Pipeline (high-level)
    "WeatherDataPipeline",
    "IngestionResult",
    "PipelineConfig",
    # Components (low-level)
    "WeatherAPIClient",
    "DataValidator",
    "ValidationConfig",
    "ValidationResult",
    "DataTransformer",
    "DataStorage",
    "DataStorageError",
]
