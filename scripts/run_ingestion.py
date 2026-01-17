#!/usr/bin/env python3
"""CLI tool for running weather data ingestion.

Usage:
    # Historical ingestion
    python scripts/run_ingestion.py --mode historical --start 2024-12-01 --end 2025-01-15

    # Incremental ingestion (fetch latest data)
    python scripts/run_ingestion.py --mode incremental

    # Backfill large date ranges
    python scripts/run_ingestion.py --mode backfill --start 2024-01-01 --end 2024-12-31

    # With custom config
    python scripts/run_ingestion.py --mode incremental --config config/data_config.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.pipeline import WeatherDataPipeline, IngestionResult

# Optional: try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)


def validate_date(date_str: str) -> str:
    """Validate date format YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD"
        )


def print_banner() -> None:
    """Print application banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       DOMAIN-SHIFT ML PLATFORM - Data Ingestion CLI          ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_result(result: IngestionResult) -> None:
    """Print ingestion result summary."""
    status = "✓ SUCCESS" if result.success else "✗ FAILED"

    print("\n" + "=" * 60)
    print(f" Ingestion Result: {status}")
    print("=" * 60)
    print(f"  Batch ID:         {result.batch_id}")
    print(f"  Records Fetched:  {result.records_fetched:,}")
    print(f"  Records Stored:   {result.records_stored:,}")
    print(f"  Quality Score:    {result.quality_score:.2%}")
    print(f"  Execution Time:   {result.execution_time_seconds:.2f}s")

    if result.error_message:
        print(f"  Error:            {result.error_message}")

    if result.metadata:
        print("  Metadata:")
        for key, value in result.metadata.items():
            print(f"    - {key}: {value}")

    print("=" * 60)


def print_backfill_summary(results: list[IngestionResult]) -> None:
    """Print summary of backfill results."""
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_fetched = sum(r.records_fetched for r in results)
    total_stored = sum(r.records_stored for r in results)
    avg_quality = (
        sum(r.quality_score for r in results) / len(results) if results else 0
    )
    total_time = sum(r.execution_time_seconds for r in results)

    print("\n" + "=" * 60)
    print(" Backfill Summary")
    print("=" * 60)
    print(f"  Total Batches:    {len(results)}")
    print(f"  Successful:       {successful}")
    print(f"  Failed:           {failed}")
    print(f"  Records Fetched:  {total_fetched:,}")
    print(f"  Records Stored:   {total_stored:,}")
    print(f"  Avg Quality:      {avg_quality:.2%}")
    print(f"  Total Time:       {total_time:.1f}s")

    if failed > 0:
        print("\n  Failed batches:")
        for i, r in enumerate(results, 1):
            if not r.success:
                print(f"    Batch {i}: {r.error_message}")

    print("=" * 60)


def run_historical(
    pipeline: WeatherDataPipeline,
    start_date: str,
    end_date: str,
) -> int:
    """Run historical ingestion."""
    print(f"\n[Historical Ingestion] {start_date} to {end_date}")
    print("-" * 60)

    result = pipeline.run_historical_ingestion(start_date, end_date)
    print_result(result)

    return 0 if result.success else 1


def run_incremental(
    pipeline: WeatherDataPipeline,
    hours: int = 24,
) -> int:
    """Run incremental ingestion."""
    print(f"\n[Incremental Ingestion] Last {hours} hours")
    print("-" * 60)

    result = pipeline.run_incremental_ingestion(hours=hours)
    print_result(result)

    return 0 if result.success else 1


def run_backfill(
    pipeline: WeatherDataPipeline,
    start_date: str,
    end_date: str,
    batch_days: int = 7,
) -> int:
    """Run backfill ingestion with progress tracking."""
    print(f"\n[Backfill Ingestion] {start_date} to {end_date}")
    print(f"  Batch size: {batch_days} days")
    print("-" * 60)

    # Calculate number of batches for progress
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1
    total_batches = (total_days + batch_days - 1) // batch_days

    if HAS_TQDM:
        pbar = tqdm(total=total_batches, desc="Backfill Progress", unit="batch")

        def progress_callback(batch_num: int, total: int, result: IngestionResult):
            pbar.update(1)
            status = "✓" if result.success else "✗"
            pbar.set_postfix({
                "status": status,
                "records": result.records_stored,
            })

        results = pipeline.backfill(
            start_date=start_date,
            end_date=end_date,
            batch_days=batch_days,
            progress_callback=progress_callback,
        )
        pbar.close()
    else:
        def progress_callback(batch_num: int, total: int, result: IngestionResult):
            status = "✓" if result.success else "✗"
            print(
                f"  Batch {batch_num}/{total}: {status} "
                f"(stored {result.records_stored} records)"
            )

        results = pipeline.backfill(
            start_date=start_date,
            end_date=end_date,
            batch_days=batch_days,
            progress_callback=progress_callback,
        )

    print_backfill_summary(results)

    # Return success only if all batches succeeded
    all_success = all(r.success for r in results)
    return 0 if all_success else 1


def run_health_check(pipeline: WeatherDataPipeline) -> int:
    """Run health check on all components."""
    print("\n[Health Check]")
    print("-" * 60)

    health = pipeline.health_check()

    for component, is_healthy in health.items():
        status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
        print(f"  {component.upper():12} {status}")

    print("-" * 60)

    all_healthy = all(health.values())
    return 0 if all_healthy else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Weather Data Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode historical --start 2024-12-01 --end 2025-01-15
  %(prog)s --mode incremental
  %(prog)s --mode backfill --start 2024-01-01 --end 2024-12-31 --batch-days 14
  %(prog)s --mode health
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["historical", "incremental", "backfill", "health"],
        required=True,
        help="Ingestion mode",
    )

    parser.add_argument(
        "--start",
        type=validate_date,
        help="Start date (YYYY-MM-DD) for historical/backfill",
    )

    parser.add_argument(
        "--end",
        type=validate_date,
        help="End date (YYYY-MM-DD) for historical/backfill",
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours of data for incremental mode (default: 24)",
    )

    parser.add_argument(
        "--batch-days",
        type=int,
        default=7,
        help="Days per batch for backfill mode (default: 7)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL"),
        help="Database connection string (default: $DATABASE_URL)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without storing to database",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print banner
    print_banner()

    # Validate arguments
    if args.mode in ["historical", "backfill"]:
        if not args.start or not args.end:
            parser.error(f"--start and --end are required for {args.mode} mode")

    # Database URL
    db_url = None if args.dry_run else args.db_url
    if not db_url and not args.dry_run:
        logger.warning(
            "No database URL provided. Running without storage. "
            "Set DATABASE_URL or use --db-url"
        )

    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = WeatherDataPipeline(
            config_path=args.config,
            db_connection_string=db_url,
        )

        # Run appropriate mode
        if args.mode == "health":
            return run_health_check(pipeline)

        elif args.mode == "historical":
            return run_historical(pipeline, args.start, args.end)

        elif args.mode == "incremental":
            return run_incremental(pipeline, args.hours)

        elif args.mode == "backfill":
            return run_backfill(
                pipeline,
                args.start,
                args.end,
                args.batch_days,
            )

    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        return 130

    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        print(f"\n✗ FATAL ERROR: {e}")
        return 1

    finally:
        if "pipeline" in locals():
            pipeline.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
