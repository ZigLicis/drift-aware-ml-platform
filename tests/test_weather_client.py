"""Tests for WeatherAPIClient.

Run with: pytest tests/test_weather_client.py -v

These tests verify:
1. API response validation logic (unit tests - no network)
2. Health check functionality (requires network)
3. Data fetching methods (requires network)

Network tests are marked with @pytest.mark.network and can be skipped with:
    pytest tests/test_weather_client.py -v -m "not network"
"""

from datetime import datetime, timedelta

import pytest

from src.data_ingestion.weather_client import WeatherAPIClient


# =============================================================================
# Unit Tests (no network required)
# =============================================================================


class TestResponseValidation:
    """Test API response validation logic."""

    def test_valid_response_passes(self):
        """Valid response with hourly data should pass validation."""
        client = WeatherAPIClient()

        valid_response = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [10.5, 11.0],
                "relative_humidity_2m": [80, 82],
            }
        }

        assert client.validate_api_response(valid_response) is True
        client.close()

    def test_missing_hourly_field_rejected(self):
        """Response without hourly field should be rejected."""
        client = WeatherAPIClient()

        invalid_response = {"daily": {}}

        assert client.validate_api_response(invalid_response) is False
        client.close()

    def test_error_response_rejected(self):
        """Response with error field should be rejected."""
        client = WeatherAPIClient()

        error_response = {"error": True, "reason": "Invalid parameters"}

        assert client.validate_api_response(error_response) is False
        client.close()

    def test_empty_response_rejected(self):
        """Empty response should be rejected."""
        client = WeatherAPIClient()

        assert client.validate_api_response({}) is False
        client.close()

    def test_none_response_raises_error(self):
        """None response should raise an error."""
        client = WeatherAPIClient()

        with pytest.raises((TypeError, AttributeError)):
            client.validate_api_response(None)
        client.close()


class TestClientConfiguration:
    """Test client configuration and initialization."""

    def test_default_configuration(self):
        """Client should initialize with default Open-Meteo URLs."""
        client = WeatherAPIClient()

        assert "open-meteo.com" in client.base_url
        assert "archive-api.open-meteo.com" in client.archive_url
        assert client.timeout > 0
        client.close()

    def test_custom_configuration(self):
        """Client should accept custom configuration."""
        custom_url = "https://custom.api.com/forecast"
        custom_archive = "https://custom.api.com/archive"

        client = WeatherAPIClient(
            base_url=custom_url,
            archive_url=custom_archive,
            timeout=60
        )

        assert client.base_url == custom_url
        assert client.archive_url == custom_archive
        assert client.timeout == 60
        client.close()

    def test_context_manager(self):
        """Client should work as context manager."""
        with WeatherAPIClient() as client:
            assert client is not None
            assert hasattr(client, 'fetch_latest')


# =============================================================================
# Network Tests (require API access)
# =============================================================================


@pytest.mark.network
class TestHealthCheck:
    """Test API health check (requires network)."""

    def test_health_check_returns_boolean(self):
        """Health check should return True when API is accessible."""
        with WeatherAPIClient() as client:
            result = client.health_check()
            assert isinstance(result, bool)
            assert result is True  # Open-Meteo should be available


@pytest.mark.network
class TestFetchLatest:
    """Test fetching latest data (requires network)."""

    def test_fetch_latest_returns_dataframe(self):
        """fetch_latest should return DataFrame with expected structure."""
        with WeatherAPIClient() as client:
            df = client.fetch_latest(hours=24)

            # Should return data
            assert len(df) > 0

            # Should have timestamp column
            assert "timestamp" in df.columns or "time" in df.columns

            # Should have weather measurements
            weather_cols = ["temperature_2m", "relative_humidity_2m"]
            for col in weather_cols:
                assert col in df.columns, f"Missing column: {col}"

    def test_fetch_latest_data_types(self):
        """fetch_latest should return correct data types."""
        with WeatherAPIClient() as client:
            df = client.fetch_latest(hours=6)

            # Temperature should be numeric
            assert df["temperature_2m"].dtype in ["float64", "float32", "int64"]

            # Humidity should be numeric
            assert df["relative_humidity_2m"].dtype in ["float64", "float32", "int64"]


@pytest.mark.network
class TestFetchHistorical:
    """Test fetching historical data (requires network)."""

    def test_fetch_historical_returns_data(self):
        """fetch_historical should return DataFrame for date range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        with WeatherAPIClient() as client:
            df = client.fetch_historical(start_str, end_str)

            # Should return data (at least some hours)
            assert len(df) > 0

            # Should have multiple days of data
            # 3 days = ~72-96 hours depending on timezone
            assert len(df) >= 48, f"Expected >= 48 records, got {len(df)}"

    def test_fetch_historical_uses_archive_for_old_dates(self):
        """fetch_historical should use archive API for dates > 5 days old."""
        # Test with dates from 2 months ago
        end_date = datetime.now() - timedelta(days=60)
        start_date = end_date - timedelta(days=3)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        with WeatherAPIClient() as client:
            # This should automatically use archive API
            df = client.fetch_historical(start_str, end_str)

            # Should still return data
            assert len(df) > 0


@pytest.mark.network
class TestGetStats:
    """Test statistics calculation (requires network)."""

    def test_get_stats_returns_dict(self):
        """get_stats should return dictionary with expected keys."""
        with WeatherAPIClient() as client:
            df = client.fetch_latest(hours=24)
            stats = client.get_stats(df)

            assert isinstance(stats, dict)
            assert "location" in stats
            assert "record_count" in stats
            assert "time_range" in stats
            assert stats["record_count"] == len(df)
