#!/usr/bin/env python3
"""Test script for WeatherAPIClient.

Run from project root:
    python -m scripts.test_weather_client
"""

import logging
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.weather_client import WeatherAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_health_check():
    """Test API health check."""
    print("\n" + "=" * 60)
    print("TEST: Health Check")
    print("=" * 60)

    with WeatherAPIClient() as client:
        is_healthy = client.health_check()
        print(f"API Health: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")
        return is_healthy


def test_fetch_latest():
    """Test fetching latest 24 hours of data."""
    print("\n" + "=" * 60)
    print("TEST: Fetch Latest 24 Hours")
    print("=" * 60)

    with WeatherAPIClient() as client:
        df = client.fetch_latest(hours=24)

        print(f"\nFetched {len(df)} records")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nLast 5 rows:\n{df.tail()}")

        # Get stats
        stats = client.get_stats(df)
        print(f"\nStatistics:")
        print(f"  Location: {stats['location']}")
        print(f"  Records: {stats['record_count']}")
        print(f"  Time range: {stats['time_range']['start']} to {stats['time_range']['end']}")

        if "temperature" in stats:
            t = stats["temperature"]
            print(f"  Temperature: {t['mean']:.1f}°C (min: {t['min']:.1f}, max: {t['max']:.1f})")

        return df


def test_fetch_historical():
    """Test fetching historical data for a week."""
    print("\n" + "=" * 60)
    print("TEST: Fetch Historical (Last 7 Days)")
    print("=" * 60)

    # Calculate dates for last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    with WeatherAPIClient() as client:
        df = client.fetch_historical(start_str, end_str)

        print(f"\nFetched {len(df)} records for {start_str} to {end_str}")
        print(f"Expected ~{8 * 24} records (8 days * 24 hours)")

        # Summary statistics
        print(f"\nSummary Statistics:")
        print(df.describe())

        # Check for nulls
        null_counts = df.isnull().sum()
        print(f"\nNull counts per column:\n{null_counts}")

        return df


def test_response_validation():
    """Test response validation logic."""
    print("\n" + "=" * 60)
    print("TEST: Response Validation")
    print("=" * 60)

    client = WeatherAPIClient()

    # Test valid response
    valid_response = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [10.5, 11.0],
            "relative_humidity_2m": [80, 82],
        }
    }
    is_valid = client.validate_api_response(valid_response)
    print(f"Valid response: {'✓ Passed' if is_valid else '✗ Failed'}")

    # Test invalid response - missing hourly
    invalid_response = {"daily": {}}
    is_valid = client.validate_api_response(invalid_response)
    print(f"Missing hourly field: {'✓ Correctly rejected' if not is_valid else '✗ Should have rejected'}")

    # Test error response
    error_response = {"error": True, "reason": "Invalid parameters"}
    is_valid = client.validate_api_response(error_response)
    print(f"Error response: {'✓ Correctly rejected' if not is_valid else '✗ Should have rejected'}")

    client.close()


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# WeatherAPIClient Test Suite")
    print("#" * 60)

    try:
        # Run tests
        test_response_validation()
        test_health_check()
        df_latest = test_fetch_latest()
        df_historical = test_fetch_historical()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 60)

    except Exception as e:
        logger.exception("Test failed with error")
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
