#!/usr/bin/env python3
"""Verify data ingestion by checking database and MLflow.

Usage:
    python scripts/verify_ingestion.py
    python scripts/verify_ingestion.py --db-url "postgresql://user:pass@localhost/db"
    python scripts/verify_ingestion.py --check-mlflow
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Optional imports
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    from src.data_ingestion.storage import DataStorage
    HAS_STORAGE = True
except ImportError:
    HAS_STORAGE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_database(db_url: str) -> bool:
    """Check database for ingested data."""
    print_section("Database Verification")

    if not HAS_STORAGE:
        print("✗ DataStorage module not available")
        return False

    try:
        storage = DataStorage(db_url)

        # Check connection
        if not storage.health_check():
            print("✗ Database connection failed")
            return False
        print("✓ Database connection successful")

        # Get record count
        total_records = storage.get_record_count()
        print(f"\nTotal records in weather_data: {total_records:,}")

        if total_records == 0:
            print("⚠ No data found in database")
            storage.close()
            return True

        # Get latest timestamp
        latest_ts = storage.get_latest_timestamp()
        if latest_ts:
            print(f"Latest timestamp: {latest_ts}")
            age = datetime.now(timezone.utc) - latest_ts.replace(tzinfo=timezone.utc)
            print(f"Data age: {age}")

        # Get sample data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)

        df = storage.get_data_by_timerange(start_time, end_time)
        print(f"\nRecords in last 7 days: {len(df):,}")

        if len(df) > 0:
            print("\nData Summary (last 7 days):")
            print("-" * 40)

            # Show time range
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Show location
            if "location_name" in df.columns:
                locations = df["location_name"].unique()
                print(f"  Locations: {', '.join(str(l) for l in locations)}")

            # Show numeric column stats
            numeric_cols = ["temperature_2m", "relative_humidity_2m", "precipitation",
                           "wind_speed_10m", "surface_pressure"]

            print("\n  Statistics:")
            for col in numeric_cols:
                if col in df.columns:
                    mean_val = df[col].mean()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    null_count = df[col].isna().sum()
                    print(f"    {col}:")
                    print(f"      Mean: {mean_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}")
                    if null_count > 0:
                        print(f"      Nulls: {null_count}")

            # Show sample rows
            print("\n  Sample records (latest 3):")
            sample_cols = ["timestamp", "location_name", "temperature_2m", "relative_humidity_2m"]
            sample_cols = [c for c in sample_cols if c in df.columns]
            print(df[sample_cols].tail(3).to_string(index=False))

        # Check ingestion log
        try:
            log_query = f"""
                SELECT batch_id, start_time, records_count, quality_score, status
                FROM dsml.ingestion_log
                ORDER BY created_at DESC
                LIMIT 5
            """
            from sqlalchemy import text
            with storage.engine.connect() as conn:
                log_df = pd.read_sql(text(log_query), conn)

            if len(log_df) > 0:
                print("\n  Recent ingestion runs:")
                print(log_df.to_string(index=False))
        except Exception as e:
            logger.debug(f"Could not read ingestion log: {e}")

        storage.close()
        print("\n✓ Database verification complete")
        return True

    except Exception as e:
        print(f"✗ Database error: {e}")
        logger.exception("Database verification failed")
        return False


def check_mlflow(tracking_uri: str | None = None) -> bool:
    """Check MLflow for logged runs."""
    print_section("MLflow Verification")

    if not HAS_MLFLOW:
        print("✗ MLflow not installed")
        return False

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        tracking_uri = mlflow.get_tracking_uri()
        print(f"Tracking URI: {tracking_uri}")

        # Get experiment
        experiment_name = "data-ingestion"
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            print(f"⚠ Experiment '{experiment_name}' not found")
            return True

        print(f"✓ Found experiment: {experiment_name}")
        print(f"  Experiment ID: {experiment.experiment_id}")
        print(f"  Artifact Location: {experiment.artifact_location}")

        # Get recent runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=10,
            order_by=["start_time DESC"],
        )

        if len(runs) == 0:
            print("\n⚠ No runs found in experiment")
            return True

        print(f"\nRecent runs ({len(runs)}):")
        print("-" * 40)

        # Show run summary
        display_cols = ["run_id", "start_time", "status"]

        # Add metric columns if available
        metric_cols = ["metrics.records_fetched", "metrics.records_stored",
                       "metrics.quality_score"]
        for col in metric_cols:
            if col in runs.columns:
                display_cols.append(col)

        # Add tag columns
        if "tags.batch_id" in runs.columns:
            display_cols.append("tags.batch_id")

        available_cols = [c for c in display_cols if c in runs.columns]
        print(runs[available_cols].head(5).to_string(index=False))

        # Summary stats
        if "metrics.records_stored" in runs.columns:
            total_stored = runs["metrics.records_stored"].sum()
            print(f"\n  Total records logged: {int(total_stored):,}")

        if "metrics.quality_score" in runs.columns:
            avg_quality = runs["metrics.quality_score"].mean()
            print(f"  Average quality score: {avg_quality:.2%}")

        print("\n✓ MLflow verification complete")
        return True

    except Exception as e:
        print(f"✗ MLflow error: {e}")
        logger.exception("MLflow verification failed")
        return False


def check_api() -> bool:
    """Check API connectivity."""
    print_section("API Verification")

    try:
        from src.data_ingestion.weather_client import WeatherAPIClient

        client = WeatherAPIClient()
        is_healthy = client.health_check()

        if is_healthy:
            print("✓ Open-Meteo API is accessible")

            # Fetch a small sample
            df = client.fetch_latest(hours=1)
            print(f"  Sample fetch: {len(df)} records")
            if len(df) > 0:
                print(f"  Latest temp: {df['temperature'].iloc[-1]:.1f}°C")

        else:
            print("✗ API health check failed")

        client.close()
        return is_healthy

    except Exception as e:
        print(f"✗ API error: {e}")
        logger.exception("API verification failed")
        return False


def check_data_quality(db_url: str) -> bool:
    """Run data quality checks on stored data."""
    print_section("Data Quality Checks")

    if not HAS_STORAGE:
        print("✗ DataStorage module not available")
        return False

    try:
        from src.data_ingestion.validator import DataValidator

        storage = DataStorage(db_url)
        validator = DataValidator()

        # Get recent data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)
        df = storage.get_data_by_timerange(start_time, end_time)

        if len(df) == 0:
            print("⚠ No data available for quality check")
            storage.close()
            return True

        # Rename columns for validator
        column_mapping = {
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
            "surface_pressure": "pressure",
        }
        df_renamed = df.rename(columns=column_mapping)

        # Run validation
        result = validator.validate_weather_data(df_renamed)

        print(f"\nValidation Result: {result}")
        print(f"\nQuality Score: {result.quality_score:.2%}")

        if result.details.get("quality_score_breakdown"):
            breakdown = result.details["quality_score_breakdown"]
            print(f"  - Completeness: {breakdown['completeness']:.2%}")
            print(f"  - Validity:     {breakdown['validity']:.2%}")
            print(f"  - Consistency:  {breakdown['consistency']:.2%}")

        if result.issues:
            print(f"\nIssues ({len(result.issues)}):")
            for issue in result.issues[:5]:  # Limit to 5
                print(f"  • {issue}")

        print(f"\nAnomalies: {result.anomaly_count}")

        storage.close()

        status = "✓" if result.is_valid else "⚠"
        print(f"\n{status} Data quality check complete")
        return result.is_valid

    except Exception as e:
        print(f"✗ Quality check error: {e}")
        logger.exception("Data quality check failed")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify data ingestion"
    )

    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL"),
        help="Database connection string (default: $DATABASE_URL)",
    )

    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)",
    )

    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Check API connectivity",
    )

    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Check database",
    )

    parser.add_argument(
        "--check-mlflow",
        action="store_true",
        help="Check MLflow runs",
    )

    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="Run data quality checks",
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all checks",
    )

    args = parser.parse_args()

    # Default to all checks if none specified
    if not any([args.check_api, args.check_db, args.check_mlflow, args.check_quality, args.all]):
        args.all = True

    print("""
╔══════════════════════════════════════════════════════════════╗
║       DOMAIN-SHIFT ML PLATFORM - Ingestion Verification      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = []

    # Run checks
    if args.all or args.check_api:
        results.append(("API", check_api()))

    if args.all or args.check_db:
        if args.db_url:
            results.append(("Database", check_database(args.db_url)))
        else:
            print("\n⚠ Skipping database check - no DATABASE_URL provided")
            results.append(("Database", None))

    if args.all or args.check_mlflow:
        results.append(("MLflow", check_mlflow(args.mlflow_uri)))

    if args.all or args.check_quality:
        if args.db_url:
            results.append(("Data Quality", check_data_quality(args.db_url)))
        else:
            print("\n⚠ Skipping quality check - no DATABASE_URL provided")
            results.append(("Data Quality", None))

    # Summary
    print_section("Verification Summary")

    all_passed = True
    for name, passed in results:
        if passed is None:
            status = "⚠ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        print(f"  {name:15} {status}")

    print("=" * 60)

    if all_passed:
        print("\n✓ All verifications passed!")
        return 0
    else:
        print("\n⚠ Some verifications failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
