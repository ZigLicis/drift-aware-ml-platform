"""Weather API client for Open-Meteo API.

This module provides a robust client for fetching weather data from the
Open-Meteo API with proper error handling, retry logic, and rate limiting.

Example:
    >>> from src.data_ingestion.weather_client import WeatherAPIClient
    >>> client = WeatherAPIClient()
    >>> df = client.fetch_latest(hours=24)
    >>> print(df.head())
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class WeatherAPIError(Exception):
    """Base exception for Weather API errors."""

    pass


class WeatherAPIConnectionError(WeatherAPIError):
    """Raised when connection to API fails."""

    pass


class WeatherAPIResponseError(WeatherAPIError):
    """Raised when API returns invalid response."""

    pass


class WeatherAPIRateLimitError(WeatherAPIError):
    """Raised when API rate limit is exceeded."""

    pass


class WeatherAPIClient:
    """Client for fetching weather data from Open-Meteo API.

    This client provides methods to fetch historical and recent weather data
    with built-in retry logic, rate limiting, and comprehensive error handling.

    Attributes:
        base_url: Base URL for the Open-Meteo API.
        latitude: Location latitude.
        longitude: Location longitude.
        timezone: Timezone for data conversion.
        timeout: Request timeout in seconds.
        retry_attempts: Number of retry attempts for failed requests.

    Example:
        >>> client = WeatherAPIClient(
        ...     latitude=38.9072,
        ...     longitude=-77.0369,
        ...     timezone="America/New_York"
        ... )
        >>> df = client.fetch_historical("2024-01-01", "2024-01-07")
        >>> print(f"Fetched {len(df)} records")
        Fetched 168 records
    """

    # Default configuration
    DEFAULT_BASE_URL = "https://api.open-meteo.com/v1/forecast"
    DEFAULT_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    DEFAULT_LATITUDE = 38.9072  # Washington DC
    DEFAULT_LONGITUDE = -77.0369
    DEFAULT_TIMEZONE = "America/New_York"
    DEFAULT_TIMEOUT = 30
    DEFAULT_RETRY_ATTEMPTS = 3

    # Weather variables to fetch
    WEATHER_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "pressure_msl",
    ]

    # Column mapping from API names to internal names
    COLUMN_MAPPING = {
        "time": "timestamp",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "precipitation": "precipitation",
        "wind_speed_10m": "wind_speed",
        "pressure_msl": "pressure",
    }

    def __init__(
        self,
        base_url: str | None = None,
        archive_url: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        timezone: str | None = None,
        timeout: int | None = None,
        retry_attempts: int | None = None,
        location_name: str = "Washington DC",
    ) -> None:
        """Initialize the Weather API client.

        Args:
            base_url: Base URL for the API. Defaults to Open-Meteo forecast endpoint.
            archive_url: URL for the archive API (historical data). Defaults to
                Open-Meteo archive endpoint for data from 2020 onwards.
            latitude: Location latitude. Defaults to Washington DC.
            longitude: Location longitude. Defaults to Washington DC.
            timezone: Timezone for data. Defaults to America/New_York.
            timeout: Request timeout in seconds. Defaults to 30.
            retry_attempts: Number of retry attempts. Defaults to 3.
            location_name: Human-readable location name for logging.

        Example:
            >>> client = WeatherAPIClient(
            ...     latitude=40.7128,
            ...     longitude=-74.0060,
            ...     timezone="America/New_York",
            ...     location_name="New York City"
            ... )
        """
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.archive_url = archive_url or self.DEFAULT_ARCHIVE_URL
        self.latitude = latitude or self.DEFAULT_LATITUDE
        self.longitude = longitude or self.DEFAULT_LONGITUDE
        self.timezone = timezone or self.DEFAULT_TIMEZONE
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.retry_attempts = retry_attempts or self.DEFAULT_RETRY_ATTEMPTS
        self.location_name = location_name

        # Validate timezone
        try:
            self._tz = pytz.timezone(self.timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            raise ValueError(f"Invalid timezone: {self.timezone}") from e

        # Create session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "DomainShiftMLPlatform/1.0",
            "Accept": "application/json",
        })

        logger.info(
            "Initialized WeatherAPIClient",
            extra={
                "location": self.location_name,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "timezone": self.timezone,
            },
        )

    def __enter__(self) -> WeatherAPIClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close session."""
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
        logger.debug("Closed WeatherAPIClient session")

    @retry(
        retry=retry_if_exception_type((
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            WeatherAPIConnectionError,
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _make_request(
        self, params: dict[str, Any], url: str | None = None
    ) -> dict[str, Any]:
        """Make HTTP request to the API with retry logic.

        Args:
            params: Query parameters for the API request.
            url: Optional URL to use. Defaults to self.base_url.

        Returns:
            Parsed JSON response from the API.

        Raises:
            WeatherAPIConnectionError: If connection fails after retries.
            WeatherAPIRateLimitError: If rate limit is exceeded.
            WeatherAPIResponseError: If response is invalid.
        """
        request_url = url or self.base_url
        try:
            logger.debug(f"Making API request to {request_url} with params: {params}")

            response = self._session.get(
                request_url,
                params=params,
                timeout=self.timeout,
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                logger.warning(f"Rate limited. Retry after {retry_after} seconds")
                raise WeatherAPIRateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds"
                )

            # Handle other HTTP errors
            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                raise WeatherAPIResponseError(
                    f"API returned status {response.status_code}: {response.text[:200]}"
                )

            data = response.json()

            # Validate response structure
            if not self.validate_api_response(data):
                raise WeatherAPIResponseError("Invalid API response structure")

            return data

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out after {self.timeout} seconds")
            raise WeatherAPIConnectionError(f"Request timed out: {e}") from e

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise WeatherAPIConnectionError(f"Connection failed: {e}") from e

        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise WeatherAPIResponseError(f"Invalid JSON response: {e}") from e

    def validate_api_response(self, response: dict[str, Any]) -> bool:
        """Validate the structure of an API response.

        Checks that the response contains required fields and has valid data.

        Args:
            response: Parsed JSON response from the API.

        Returns:
            True if response is valid, False otherwise.

        Example:
            >>> client = WeatherAPIClient()
            >>> valid_response = {"hourly": {"time": [...], "temperature_2m": [...]}}
            >>> client.validate_api_response(valid_response)
            True
            >>> client.validate_api_response({})
            False
        """
        # Check for error response
        if "error" in response:
            logger.error(f"API returned error: {response.get('reason', 'Unknown error')}")
            return False

        # Check for required hourly data
        if "hourly" not in response:
            logger.error("Response missing 'hourly' field")
            return False

        hourly = response["hourly"]

        # Check for time series
        if "time" not in hourly:
            logger.error("Response missing 'hourly.time' field")
            return False

        # Check that we have at least some weather variables
        has_data = any(var in hourly for var in self.WEATHER_VARIABLES)
        if not has_data:
            logger.error("Response missing all weather variables")
            return False

        # Check data length consistency
        time_length = len(hourly["time"])
        for var in self.WEATHER_VARIABLES:
            if var in hourly and len(hourly[var]) != time_length:
                logger.error(
                    f"Data length mismatch: time has {time_length} entries, "
                    f"{var} has {len(hourly[var])}"
                )
                return False

        logger.debug(f"Response validation passed with {time_length} data points")
        return True

    def _build_params(
        self,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Build query parameters for API request.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            Dictionary of query parameters.
        """
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.WEATHER_VARIABLES),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",  # Always fetch in UTC, convert later
        }

    def _response_to_dataframe(self, response: dict[str, Any]) -> pd.DataFrame:
        """Convert API response to pandas DataFrame.

        Args:
            response: Validated API response.

        Returns:
            DataFrame with weather data, timestamps converted to local timezone.
        """
        hourly = response["hourly"]

        # Build dataframe from hourly data
        data = {"time": hourly["time"]}
        for var in self.WEATHER_VARIABLES:
            if var in hourly:
                data[var] = hourly[var]

        df = pd.DataFrame(data)

        # Convert timestamp to datetime and localize
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].dt.tz_localize("UTC").dt.tz_convert(self._tz)

        # Rename columns to internal names
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Add metadata columns
        df["location_name"] = self.location_name
        df["latitude"] = self.latitude
        df["longitude"] = self.longitude

        # Reorder columns
        column_order = [
            "timestamp",
            "location_name",
            "latitude",
            "longitude",
            "temperature",
            "humidity",
            "precipitation",
            "wind_speed",
            "pressure",
        ]
        df = df[[col for col in column_order if col in df.columns]]

        logger.debug(f"Converted response to DataFrame with {len(df)} rows")
        return df

    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical hourly weather data for a date range.

        Retrieves hourly weather observations from the Open-Meteo API for the
        specified date range. Data includes temperature, humidity, precipitation,
        wind speed, and pressure.

        Automatically selects the appropriate API:
        - Archive API (archive-api.open-meteo.com) for dates older than 5 days
        - Forecast API (api.open-meteo.com) for recent dates

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format (inclusive).

        Returns:
            DataFrame with columns:
                - timestamp: datetime with timezone
                - location_name: str
                - latitude: float
                - longitude: float
                - temperature: float (°C)
                - humidity: float (%)
                - precipitation: float (mm)
                - wind_speed: float (km/h)
                - pressure: float (hPa)

        Raises:
            WeatherAPIError: If API request fails.
            ValueError: If date format is invalid.

        Example:
            >>> client = WeatherAPIClient()
            >>> df = client.fetch_historical("2024-01-01", "2024-01-07")
            >>> print(df.columns.tolist())
            ['timestamp', 'location_name', 'latitude', 'longitude',
             'temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
            >>> print(f"Records: {len(df)}")
            Records: 168
        """
        # Validate date format
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(
                f"Invalid date format. Use YYYY-MM-DD. Error: {e}"
            ) from e

        if end_dt < start_dt:
            raise ValueError(
                f"end_date ({end_date}) must be >= start_date ({start_date})"
            )

        # Use archive API for dates older than 5 days ago
        cutoff = datetime.now() - timedelta(days=5)
        use_archive = start_dt < cutoff
        api_url = self.archive_url if use_archive else self.base_url
        api_name = "archive" if use_archive else "forecast"

        logger.info(
            f"Fetching historical data for {self.location_name} using {api_name} API",
            extra={
                "start_date": start_date,
                "end_date": end_date,
                "days": (end_dt - start_dt).days + 1,
                "api": api_name,
            },
        )

        params = self._build_params(start_date, end_date)
        response = self._make_request(params, url=api_url)
        df = self._response_to_dataframe(response)

        logger.info(
            f"Successfully fetched {len(df)} hourly records",
            extra={
                "location": self.location_name,
                "start_date": start_date,
                "end_date": end_date,
                "record_count": len(df),
            },
        )

        return df

    def fetch_latest(self, hours: int = 24) -> pd.DataFrame:
        """Fetch the most recent N hours of weather data.

        Useful for incremental data ingestion. Fetches data from (now - hours)
        to now.

        Args:
            hours: Number of hours of historical data to fetch. Defaults to 24.
                   Maximum is typically 16 days (384 hours) for forecast API.

        Returns:
            DataFrame with weather data for the specified time window.
            See fetch_historical() for column details.

        Raises:
            WeatherAPIError: If API request fails.
            ValueError: If hours is invalid.

        Example:
            >>> client = WeatherAPIClient()
            >>> df = client.fetch_latest(hours=48)
            >>> print(f"Fetched {len(df)} records from last 48 hours")
            Fetched 48 records from last 48 hours
        """
        if hours <= 0:
            raise ValueError(f"hours must be positive, got {hours}")

        if hours > 384:  # ~16 days
            logger.warning(
                f"Requested {hours} hours exceeds typical API limit. "
                "Consider using fetch_historical() with archive API."
            )

        # Calculate date range
        now = datetime.now(self._tz)
        start_dt = now - timedelta(hours=hours)

        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

        logger.info(
            f"Fetching latest {hours} hours of data for {self.location_name}",
            extra={
                "hours": hours,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        df = self.fetch_historical(start_date, end_date)

        # Filter to exactly the requested hours
        cutoff_time = now - timedelta(hours=hours)
        df = df[df["timestamp"] >= cutoff_time].copy()

        logger.info(
            f"Filtered to {len(df)} records within {hours}-hour window",
            extra={"record_count": len(df)},
        )

        return df

    def health_check(self) -> bool:
        """Check if the API is accessible.

        Makes a minimal request to verify connectivity.

        Returns:
            True if API is accessible, False otherwise.

        Example:
            >>> client = WeatherAPIClient()
            >>> if client.health_check():
            ...     print("API is healthy")
            API is healthy
        """
        try:
            # Make minimal request - just today's data
            today = datetime.now().strftime("%Y-%m-%d")
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": "temperature_2m",
                "start_date": today,
                "end_date": today,
            }

            response = self._session.get(
                self.base_url,
                params=params,
                timeout=10,
            )

            is_healthy = response.status_code == 200
            logger.info(f"Health check: {'passed' if is_healthy else 'failed'}")
            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get summary statistics for weather data.

        Args:
            df: DataFrame from fetch_historical or fetch_latest.

        Returns:
            Dictionary with summary statistics.

        Example:
            >>> client = WeatherAPIClient()
            >>> df = client.fetch_latest(hours=24)
            >>> stats = client.get_stats(df)
            >>> print(f"Avg temp: {stats['temperature']['mean']:.1f}°C")
        """
        stats: dict[str, Any] = {
            "record_count": len(df),
            "time_range": {
                "start": df["timestamp"].min().isoformat() if len(df) > 0 else None,
                "end": df["timestamp"].max().isoformat() if len(df) > 0 else None,
            },
            "location": self.location_name,
        }

        # Compute stats for numeric columns
        numeric_cols = ["temperature", "humidity", "precipitation", "wind_speed", "pressure"]
        for col in numeric_cols:
            if col in df.columns:
                stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "null_count": df[col].isna().sum(),
                }

        return stats
