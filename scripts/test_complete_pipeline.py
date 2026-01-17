#!/usr/bin/env python3
"""Complete end-to-end pipeline test.

This script validates the entire data ingestion pipeline:
1. Docker services health
2. Data ingestion (7 days historical)
3. Database verification
4. MLflow tracking verification
5. Generates comprehensive test report

Run from project root:
    python scripts/test_complete_pipeline.py

Or via Docker:
    docker-compose exec app python scripts/test_complete_pipeline.py
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class TestReport:
    """Complete test report."""
    results: list[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def add(self, result: TestResult) -> None:
        self.results.append(result)

    def print_report(self) -> None:
        """Print formatted test report."""
        self.end_time = datetime.now(timezone.utc)
        duration = (self.end_time - self.start_time).total_seconds()

        print("\n" + "=" * 70)
        print(" DOMAIN-SHIFT ML PLATFORM - End-to-End Test Report")
        print("=" * 70)
        print(f" Started:  {self.start_time.isoformat()}")
        print(f" Duration: {duration:.1f} seconds")
        print(f" Results:  {self.passed} passed, {self.failed} failed")
        print("=" * 70)

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n[{status}] {result.name}")
            print(f"  {result.message}")
            if result.details:
                for key, value in result.details.items():
                    print(f"  • {key}: {value}")

        print("\n" + "=" * 70)
        if self.all_passed:
            print(" ✓ ALL TESTS PASSED - Pipeline is fully operational!")
        else:
            print(f" ✗ {self.failed} TEST(S) FAILED - Please check the issues above")
        print("=" * 70 + "\n")


def print_section(title: str) -> None:
    """Print section header."""
    print("\n" + "-" * 60)
    print(f" {title}")
    print("-" * 60)


def get_db_url() -> str:
    """Get database URL from environment or defaults."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://dsml_user:mlpassword@localhost:5432/dsml_db"
    )


def get_mlflow_uri() -> str:
    """Get MLflow URI from environment or defaults."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")


# =============================================================================
# Test Functions
# =============================================================================

def test_docker_services(report: TestReport) -> bool:
    """Test 1: Check Docker services are accessible."""
    print_section("Test 1: Docker Services Health")
    import time
    start = time.time()

    results = {
        "postgres": False,
        "mlflow": False,
    }

    # Test PostgreSQL
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(get_db_url())
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        results["postgres"] = True
        print("  ✓ PostgreSQL: Connected")
        engine.dispose()
    except Exception as e:
        print(f"  ✗ PostgreSQL: {e}")

    # Test MLflow
    try:
        import requests
        mlflow_url = get_mlflow_uri()
        # Try health endpoint first, then root
        try:
            resp = requests.get(f"{mlflow_url}/health", timeout=5)
            results["mlflow"] = resp.status_code == 200
        except:
            resp = requests.get(f"{mlflow_url}/api/2.0/mlflow/experiments/search", timeout=5)
            results["mlflow"] = resp.status_code in [200, 400]  # 400 means it's running

        if results["mlflow"]:
            print(f"  ✓ MLflow: Accessible at {mlflow_url}")
        else:
            print(f"  ✗ MLflow: Not responding (status: {resp.status_code})")
    except Exception as e:
        print(f"  ✗ MLflow: {e}")

    all_healthy = all(results.values())
    duration = time.time() - start

    report.add(TestResult(
        name="Docker Services Health",
        passed=all_healthy,
        message="All services accessible" if all_healthy else "Some services unavailable",
        details=results,
        duration_seconds=duration,
    ))

    return all_healthy


def test_data_ingestion(report: TestReport) -> tuple[bool, str | None]:
    """Test 2: Run data ingestion for last 7 days."""
    print_section("Test 2: Data Ingestion (7 days)")
    import time
    start = time.time()

    try:
        from src.data_ingestion.pipeline import WeatherDataPipeline

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        print(f"  Ingesting data from {start_str} to {end_str}...")

        # Initialize pipeline
        pipeline = WeatherDataPipeline(
            db_connection_string=get_db_url(),
        )

        # Configure MLflow
        import mlflow
        mlflow.set_tracking_uri(get_mlflow_uri())

        # Run ingestion
        result = pipeline.run_historical_ingestion(start_str, end_str)
        pipeline.close()

        duration = time.time() - start

        if result.success:
            print(f"  ✓ Fetched: {result.records_fetched} records")
            print(f"  ✓ Stored:  {result.records_stored} records")
            print(f"  ✓ Quality: {result.quality_score:.2%}")
            print(f"  ✓ Batch:   {result.batch_id}")

            report.add(TestResult(
                name="Data Ingestion",
                passed=True,
                message=f"Successfully ingested {result.records_stored} records",
                details={
                    "records_fetched": result.records_fetched,
                    "records_stored": result.records_stored,
                    "quality_score": f"{result.quality_score:.2%}",
                    "batch_id": result.batch_id,
                    "date_range": f"{start_str} to {end_str}",
                },
                duration_seconds=duration,
            ))
            return True, result.batch_id
        else:
            print(f"  ✗ Ingestion failed: {result.error_message}")
            report.add(TestResult(
                name="Data Ingestion",
                passed=False,
                message=f"Ingestion failed: {result.error_message}",
                duration_seconds=duration,
            ))
            return False, None

    except Exception as e:
        duration = time.time() - start
        print(f"  ✗ Error: {e}")
        logger.exception("Ingestion test failed")
        report.add(TestResult(
            name="Data Ingestion",
            passed=False,
            message=f"Exception: {str(e)}",
            duration_seconds=duration,
        ))
        return False, None


def test_database_verification(report: TestReport) -> bool:
    """Test 3: Verify data in database."""
    print_section("Test 3: Database Verification")
    import time
    start = time.time()

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(get_db_url())

        with engine.connect() as conn:
            # Check total record count
            result = conn.execute(text("""
                SELECT COUNT(*) as count,
                       MIN(timestamp) as min_ts,
                       MAX(timestamp) as max_ts
                FROM dsml.weather_data
            """))
            row = result.fetchone()
            total_count = row[0]
            min_ts = row[1]
            max_ts = row[2]

            print(f"  Total records: {total_count:,}")
            if min_ts and max_ts:
                print(f"  Time range: {min_ts} to {max_ts}")

            # Check data quality - null counts
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN temperature_2m IS NULL THEN 1 ELSE 0 END) as temp_nulls,
                    SUM(CASE WHEN relative_humidity_2m IS NULL THEN 1 ELSE 0 END) as humidity_nulls,
                    AVG(temperature_2m) as avg_temp,
                    MIN(temperature_2m) as min_temp,
                    MAX(temperature_2m) as max_temp
                FROM dsml.weather_data
            """))
            row = result.fetchone()
            temp_nulls = row[1]
            humidity_nulls = row[2]
            avg_temp = row[3]
            min_temp = row[4]
            max_temp = row[5]

            print(f"  Temperature nulls: {temp_nulls}")
            print(f"  Humidity nulls: {humidity_nulls}")
            if avg_temp:
                print(f"  Temperature range: {min_temp:.1f}°C to {max_temp:.1f}°C (avg: {avg_temp:.1f}°C)")

            # Check recent 7 days
            result = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM dsml.weather_data
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            """))
            recent_count = result.fetchone()[0]
            print(f"  Records (last 7 days): {recent_count}")

            # Check unique locations
            result = conn.execute(text("""
                SELECT DISTINCT location_name FROM dsml.weather_data
            """))
            locations = [row[0] for row in result.fetchall()]
            print(f"  Locations: {', '.join(locations)}")

        engine.dispose()

        duration = time.time() - start

        # Validation criteria
        passed = total_count > 0 and recent_count > 0

        report.add(TestResult(
            name="Database Verification",
            passed=passed,
            message=f"{total_count:,} records in database, {recent_count} from last 7 days",
            details={
                "total_records": total_count,
                "recent_records": recent_count,
                "locations": locations,
                "temp_range": f"{min_temp}°C to {max_temp}°C" if min_temp else "N/A",
            },
            duration_seconds=duration,
        ))

        return passed

    except Exception as e:
        duration = time.time() - start
        print(f"  ✗ Error: {e}")
        logger.exception("Database verification failed")
        report.add(TestResult(
            name="Database Verification",
            passed=False,
            message=f"Exception: {str(e)}",
            duration_seconds=duration,
        ))
        return False


def test_mlflow_verification(report: TestReport, batch_id: str | None) -> bool:
    """Test 4: Verify MLflow tracking."""
    print_section("Test 4: MLflow Verification")
    import time
    start = time.time()

    try:
        import mlflow

        mlflow.set_tracking_uri(get_mlflow_uri())
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")

        # Check experiment
        experiment_name = "data-ingestion"
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            print(f"  ⚠ Experiment '{experiment_name}' not found")
            # Try to create it
            mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment:
            print(f"  ✓ Experiment: {experiment_name} (ID: {experiment.experiment_id})")

            # Search for recent runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=5,
                order_by=["start_time DESC"],
            )

            print(f"  ✓ Total runs: {len(runs)}")

            if len(runs) > 0:
                latest = runs.iloc[0]
                print(f"\n  Latest run:")
                print(f"    Run ID: {latest['run_id'][:8]}...")
                print(f"    Status: {latest['status']}")

                # Show metrics if available
                metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
                if metric_cols:
                    print("    Metrics:")
                    for col in metric_cols[:5]:
                        metric_name = col.replace("metrics.", "")
                        value = latest[col]
                        if pd.notna(value):
                            if "score" in metric_name or "rate" in metric_name:
                                print(f"      - {metric_name}: {value:.2%}")
                            else:
                                print(f"      - {metric_name}: {value}")

                # Check if our batch was logged
                if batch_id:
                    batch_runs = runs[runs.get("tags.batch_id", "") == batch_id]
                    if len(batch_runs) > 0:
                        print(f"\n  ✓ Current batch ({batch_id[:8]}...) logged to MLflow")
                    else:
                        print(f"\n  ⚠ Current batch not found in recent runs")

        duration = time.time() - start

        passed = experiment is not None

        report.add(TestResult(
            name="MLflow Verification",
            passed=passed,
            message=f"Experiment '{experiment_name}' active with {len(runs)} runs",
            details={
                "experiment_name": experiment_name,
                "experiment_id": experiment.experiment_id if experiment else None,
                "total_runs": len(runs) if experiment else 0,
                "tracking_uri": mlflow.get_tracking_uri(),
            },
            duration_seconds=duration,
        ))

        return passed

    except Exception as e:
        duration = time.time() - start
        print(f"  ✗ Error: {e}")
        logger.exception("MLflow verification failed")
        report.add(TestResult(
            name="MLflow Verification",
            passed=False,
            message=f"Exception: {str(e)}",
            duration_seconds=duration,
        ))
        return False


def test_data_quality(report: TestReport) -> bool:
    """Test 5: Run data quality validation on stored data."""
    print_section("Test 5: Data Quality Validation")
    import time
    start = time.time()

    try:
        from src.data_ingestion.validator import DataValidator
        from src.data_ingestion.storage import DataStorage

        storage = DataStorage(get_db_url())

        # Get recent data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)
        df = storage.get_data_by_timerange(start_time, end_time)
        storage.close()

        if len(df) == 0:
            print("  ⚠ No data available for quality check")
            report.add(TestResult(
                name="Data Quality Validation",
                passed=False,
                message="No data available",
                duration_seconds=time.time() - start,
            ))
            return False

        # Rename columns for validator
        column_mapping = {
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
            "surface_pressure": "pressure",
        }
        df_renamed = df.rename(columns=column_mapping)

        # Run validation
        validator = DataValidator()
        result = validator.validate_weather_data(df_renamed)

        print(f"  Quality Score: {result.quality_score:.2%}")
        print(f"  Valid Records: {result.valid_records}/{result.total_records}")
        print(f"  Anomalies: {result.anomaly_count}")

        if result.details.get("quality_score_breakdown"):
            breakdown = result.details["quality_score_breakdown"]
            print(f"  Breakdown:")
            print(f"    - Completeness: {breakdown['completeness']:.2%}")
            print(f"    - Validity: {breakdown['validity']:.2%}")
            print(f"    - Consistency: {breakdown['consistency']:.2%}")

        if result.issues:
            print(f"  Issues ({len(result.issues)}):")
            for issue in result.issues[:3]:
                print(f"    • {issue}")

        duration = time.time() - start

        report.add(TestResult(
            name="Data Quality Validation",
            passed=result.is_valid,
            message=f"Quality score: {result.quality_score:.2%}",
            details={
                "quality_score": f"{result.quality_score:.2%}",
                "valid_records": f"{result.valid_records}/{result.total_records}",
                "anomalies": result.anomaly_count,
                "issues": len(result.issues),
            },
            duration_seconds=duration,
        ))

        return result.is_valid

    except Exception as e:
        duration = time.time() - start
        print(f"  ✗ Error: {e}")
        logger.exception("Data quality test failed")
        report.add(TestResult(
            name="Data Quality Validation",
            passed=False,
            message=f"Exception: {str(e)}",
            duration_seconds=duration,
        ))
        return False


def generate_sql_verification_queries() -> None:
    """Print SQL queries for manual verification."""
    print_section("SQL Queries for Manual Verification")
    print("""
  Connect to database:
    docker-compose exec postgres psql -U dsml_user -d dsml_db

  Queries:
    -- Total record count
    SELECT COUNT(*) FROM dsml.weather_data;

    -- Recent data summary
    SELECT
        DATE(timestamp) as date,
        COUNT(*) as records,
        ROUND(AVG(temperature_2m)::numeric, 1) as avg_temp,
        ROUND(MIN(temperature_2m)::numeric, 1) as min_temp,
        ROUND(MAX(temperature_2m)::numeric, 1) as max_temp
    FROM dsml.weather_data
    WHERE timestamp >= NOW() - INTERVAL '7 days'
    GROUP BY DATE(timestamp)
    ORDER BY date;

    -- Ingestion batches
    SELECT batch_id, COUNT(*) as records, MIN(timestamp) as start, MAX(timestamp) as end
    FROM dsml.weather_data
    GROUP BY batch_id
    ORDER BY MIN(timestamp) DESC
    LIMIT 5;

    -- Ingestion log
    SELECT * FROM dsml.ingestion_log ORDER BY created_at DESC LIMIT 5;
    """)


def generate_mlflow_instructions() -> None:
    """Print MLflow UI verification instructions."""
    print_section("MLflow UI Verification")
    print(f"""
  Access MLflow UI:
    http://localhost:5001

  What to check:
    1. Experiments page:
       - Look for "data-ingestion" experiment
       - Should see recent runs listed

    2. Latest run:
       - Click on the most recent run
       - Check Parameters: location, date range
       - Check Metrics: records_stored, quality_score
       - Check Artifacts: validation_report.txt

    3. Compare runs:
       - Select multiple runs
       - Compare quality scores over time
    """)


def main() -> int:
    """Run complete end-to-end test."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         DOMAIN-SHIFT ML PLATFORM - Complete Pipeline Test            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    report = TestReport()
    batch_id = None

    # Test 1: Docker Services
    services_ok = test_docker_services(report)
    if not services_ok:
        print("\n⚠ Docker services not ready. Please ensure containers are running:")
        print("  docker-compose up -d")
        print("  docker-compose ps")

    # Test 2: Data Ingestion (only if services are OK)
    if services_ok:
        ingestion_ok, batch_id = test_data_ingestion(report)
    else:
        report.add(TestResult(
            name="Data Ingestion",
            passed=False,
            message="Skipped - services not available",
        ))

    # Test 3: Database Verification
    if services_ok:
        test_database_verification(report)
    else:
        report.add(TestResult(
            name="Database Verification",
            passed=False,
            message="Skipped - services not available",
        ))

    # Test 4: MLflow Verification
    if services_ok:
        test_mlflow_verification(report, batch_id)
    else:
        report.add(TestResult(
            name="MLflow Verification",
            passed=False,
            message="Skipped - services not available",
        ))

    # Test 5: Data Quality
    if services_ok:
        test_data_quality(report)
    else:
        report.add(TestResult(
            name="Data Quality Validation",
            passed=False,
            message="Skipped - services not available",
        ))

    # Print report
    report.print_report()

    # Print verification instructions
    generate_sql_verification_queries()
    generate_mlflow_instructions()

    # Save report to file
    report_path = "/tmp/pipeline_test_report.json"
    try:
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "passed": report.passed,
                "failed": report.failed,
                "all_passed": report.all_passed,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "message": r.message,
                        "details": r.details,
                        "duration": r.duration_seconds,
                    }
                    for r in report.results
                ],
            }, f, indent=2, default=str)
        print(f"\n  Report saved to: {report_path}")
    except Exception as e:
        print(f"\n  Could not save report: {e}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
