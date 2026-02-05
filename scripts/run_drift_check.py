#!/usr/bin/env python3
"""
Drift Detection CLI - Domain-Shift ML Platform

Production CLI tool for detecting domain shift in weather data.

Usage:
    # Create reference from training data
    python scripts/run_drift_check.py create-reference \\
        --name baseline_v1 --start 2024-01-01 --end 2024-12-31

    # Run drift check against last 7 days
    python scripts/run_drift_check.py check \\
        --reference baseline_v1 --window-hours 168

    # Run drift check against specific date range
    python scripts/run_drift_check.py check \\
        --reference baseline_v1 --start 2025-01-01 --end 2025-01-15

    # Show drift history from MLflow
    python scripts/run_drift_check.py history --runs 10

    # List available references
    python scripts/run_drift_check.py list-references
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.data_ingestion.storage import DataStorage
from src.drift_detection import (
    DriftDetector,
    DriftMLflowLogger,
    DriftSeverity,
    ReferenceManager,
    ReferenceNotFoundError,
)

logger = logging.getLogger(__name__)

# Default features to monitor (raw weather measurements subject to drift)
DEFAULT_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
]

# Status indicators
STATUS_ICONS = {
    DriftSeverity.NONE: ("🟢", "STABLE"),
    DriftSeverity.LOW: ("🟢", "STABLE"),
    DriftSeverity.MODERATE: ("🟡", "WATCH"),
    DriftSeverity.SIGNIFICANT: ("🔴", "DRIFTED"),
    DriftSeverity.SEVERE: ("🔴", "DRIFTED"),
}

SEVERITY_DISPLAY = {
    DriftSeverity.NONE: "NONE",
    DriftSeverity.LOW: "LOW",
    DriftSeverity.MODERATE: "MODERATE",
    DriftSeverity.SIGNIFICANT: "SIGNIF.",
    DriftSeverity.SEVERE: "SEVERE",
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


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
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║        DOMAIN-SHIFT ML PLATFORM - Drift Detection            ║
╚══════════════════════════════════════════════════════════════╝
"""
    )


def load_config() -> dict:
    """Load drift configuration from YAML."""
    config_path = Path("config/drift_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_features_to_monitor(args, config: dict) -> list[str]:
    """Determine which features to monitor."""
    # Priority: CLI args > config > defaults
    if hasattr(args, "features") and args.features:
        return args.features

    monitored = config.get("drift_detection", {}).get("monitored_features", [])
    if monitored:
        return monitored

    return DEFAULT_FEATURES


def create_reference(args, config: dict) -> int:
    """Create reference profiles from historical data."""
    print_banner()
    print("📊 Creating Reference Profile")
    print("=" * 62)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return 2

    features = get_features_to_monitor(args, config)

    print(f"  📁 Reference Name: {args.name}")
    print(f"  📅 Date Range:     {args.start} to {args.end}")
    print(f"  🔍 Features:       {', '.join(features)}")
    print()

    try:
        # Connect to database
        print("🔄 Connecting to database...")
        storage = DataStorage(db_url)

        # Fetch data
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

        print(f"🔄 Fetching data from {args.start} to {args.end}...")
        df = storage.get_data_by_timerange(start=start_dt, end=end_dt)

        if df.empty:
            print(f"❌ ERROR: No data found for date range {args.start} to {args.end}")
            return 2

        print(f"   ✓ Retrieved {len(df):,} records")

        # Validate features exist
        missing = set(features) - set(df.columns)
        if missing:
            print(f"❌ ERROR: Features not found in data: {missing}")
            print(f"   Available: {list(df.columns)}")
            return 2

        # Create reference
        print("🔄 Creating reference profiles...")
        manager = ReferenceManager(storage_path="data/references")

        profiles = manager.create_reference_from_dataframe(
            df=df,
            feature_columns=features,
            reference_name=args.name,
            store_raw_values=True,  # Store for KS-test
        )

        # Save reference
        metadata = {
            "date_start": args.start,
            "date_end": args.end,
            "created_by": "run_drift_check.py",
        }
        save_path = manager.save_reference(profiles, args.name, metadata=metadata)

        print()
        print("=" * 62)
        print("✅ Reference Created Successfully!")
        print("=" * 62)
        print(f"  📁 Name:        {args.name}")
        print(f"  📂 Path:        {save_path}")
        print(f"  📊 Samples:     {len(df):,}")
        print(f"  🔍 Features:    {len(features)}")
        print()
        print("  Feature Statistics:")
        print("  " + "-" * 58)
        for name, profile in profiles.items():
            print(
                f"    {name:25s} mean={profile.mean:8.2f}  std={profile.std:6.2f}"
            )
        print("=" * 62)

        return 0

    except Exception as e:
        print(f"❌ ERROR: {e}")
        logger.exception("Failed to create reference")
        return 2


def check_drift(args, config: dict) -> int:
    """Run drift detection against a reference."""
    print_banner()

    db_url = os.getenv("DATABASE_URL")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return 2

    # Initialize components
    try:
        storage = DataStorage(db_url)
        manager = ReferenceManager(storage_path="data/references")

        # Check reference exists
        if not manager.reference_exists(args.reference):
            print(f"❌ ERROR: Reference '{args.reference}' not found")
            print()
            available = manager.list_references()
            if available:
                print("Available references:")
                for ref in available:
                    print(f"  - {ref['reference_name']}")
            else:
                print("No references available. Create one with:")
                print("  python scripts/run_drift_check.py create-reference --name <name> --start <date> --end <date>")
            return 2

        # Load reference metadata
        ref_metadata = manager.get_reference_metadata(args.reference)
        ref_samples = ref_metadata.get("total_samples", "?")
        ref_start = ref_metadata.get("date_start", "?")
        ref_end = ref_metadata.get("date_end", "?")
        features = ref_metadata.get("features", DEFAULT_FEATURES)

        # Determine current data window
        if args.start and args.end:
            current_start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            current_end = datetime.strptime(args.end, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            window_desc = f"{args.start} to {args.end}"
        else:
            current_end = datetime.now(timezone.utc)
            current_start = current_end - timedelta(hours=args.window_hours)
            window_desc = f"Last {args.window_hours} hours"

        # Fetch current data
        print(f"🔄 Fetching current data ({window_desc})...")
        current_df = storage.get_data_by_timerange(start=current_start, end=current_end)

        if current_df.empty:
            print(f"❌ ERROR: No data found for current window")
            return 2

        # Run drift detection
        print(f"🔄 Running drift detection against '{args.reference}'...")
        detector = DriftDetector(
            reference_manager=manager,
            config=config.get("drift_detection"),
        )

        report = detector.detect_drift(
            reference_name=args.reference,
            current_data=current_df,
            feature_columns=features,
        )

        # Log to MLflow (unless disabled)
        mlflow_run_id = None
        mlflow_logged = False
        if not args.no_mlflow:
            try:
                mlflow_logger = DriftMLflowLogger(
                    tracking_uri=mlflow_uri,
                    experiment_name=config.get("mlflow", {}).get(
                        "experiment_name", "drift-monitoring"
                    ),
                )
                mlflow_run_id = mlflow_logger.log_drift_report(
                    report,
                    model_name=args.model_name,
                    model_version=args.model_version,
                )
                mlflow_logged = True
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
                print(f"⚠️  Warning: Could not log to MLflow: {e}")

        # Print report
        print()
        print_drift_report(
            report,
            ref_name=args.reference,
            ref_start=ref_start,
            ref_end=ref_end,
            ref_samples=ref_samples,
            window_desc=window_desc,
            current_samples=len(current_df),
            mlflow_logged=mlflow_logged,
            mlflow_run_id=mlflow_run_id,
            mlflow_uri=mlflow_uri,
        )

        # Return based on drift status
        return 1 if report.drift_detected else 0

    except ReferenceNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return 2
    except Exception as e:
        print(f"❌ ERROR: {e}")
        logger.exception("Drift check failed")
        return 2


def print_drift_report(
    report,
    ref_name: str,
    ref_start: str,
    ref_end: str,
    ref_samples,
    window_desc: str,
    current_samples: int,
    mlflow_logged: bool,
    mlflow_run_id: str | None,
    mlflow_uri: str,
) -> None:
    """Print formatted drift report to console."""
    print("=" * 62)
    print(f"📊 Reference:   {ref_name} ({ref_start} to {ref_end}, n={ref_samples:,})")
    print(f"📈 Current:     {window_desc} (n={current_samples:,})")
    print(f"🔍 Features:    {report.n_features_total} monitored")
    print()

    # Feature table
    print("┌────────────────────────────┬────────┬────────────┬──────────────┐")
    print("│ Feature                    │ PSI    │ Severity   │ Status       │")
    print("├────────────────────────────┼────────┼────────────┼──────────────┤")

    # Sort by severity (worst first)
    sorted_results = sorted(
        report.feature_results.values(),
        key=lambda r: list(DriftSeverity).index(r.severity),
        reverse=True,
    )

    for result in sorted_results:
        icon, status = STATUS_ICONS[result.severity]
        severity_str = SEVERITY_DISPLAY[result.severity]
        name = result.feature_name[:26]
        print(
            f"│ {name:26s} │ {result.psi:6.3f} │ {severity_str:10s} │ {icon} {status:8s} │"
        )

    print("└────────────────────────────┴────────┴────────────┴──────────────┘")

    # Overall assessment
    print()
    print("═" * 62)
    print("OVERALL ASSESSMENT")
    print("═" * 62)

    action = "ACTION REQUIRED" if report.drift_detected else "CONTINUE MONITORING"
    print(f"  Drift Score:      {report.overall_drift_score:.2f} / 1.00")
    print(f"  Severity:         {report.overall_severity.value.upper()}")
    print(
        f"  Features Drifted: {report.n_features_drifted} / {report.n_features_total} "
        f"({100 * report.n_features_drifted / report.n_features_total:.0f}%)"
    )
    print(f"  Action Required:  {action}")

    if report.recommendations:
        print()
        print("  Recommendations:")
        for rec in report.recommendations:
            print(f"    • {rec}")

    if report.warnings:
        print()
        print("  Warnings:")
        for warn in report.warnings:
            print(f"    ⚠️  {warn}")

    print("═" * 62)

    # MLflow info
    if mlflow_logged and mlflow_run_id:
        print()
        print("📊 Results logged to MLflow")
        print(f"   Experiment: drift-monitoring")
        print(f"   Run ID:     {mlflow_run_id[:16]}...")
        # Extract experiment ID from a simple format
        print(f"   View at:    {mlflow_uri}")
    elif not mlflow_logged:
        print()
        print("⚠️  Results NOT logged to MLflow (use --no-mlflow to suppress this)")


def show_history(args, config: dict) -> int:
    """Show drift detection history from MLflow."""
    print_banner()
    print("📊 Drift Detection History")
    print("=" * 62)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    try:
        mlflow_logger = DriftMLflowLogger(
            tracking_uri=mlflow_uri,
            experiment_name=config.get("mlflow", {}).get(
                "experiment_name", "drift-monitoring"
            ),
        )

        df = mlflow_logger.get_drift_history(
            n_runs=args.runs,
            model_name=args.model_name,
        )

        if df.empty:
            print("No drift detection runs found.")
            print()
            print("Run a drift check first:")
            print("  python scripts/run_drift_check.py check --reference <name>")
            return 0

        print(f"Showing last {len(df)} runs:")
        print()
        print("┌──────────────────────┬────────┬────────────┬──────────┐")
        print("│ Timestamp            │ Score  │ Severity   │ Drifted  │")
        print("├──────────────────────┼────────┼────────────┼──────────┤")

        for _, row in df.iterrows():
            ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            score = row.get("overall_drift_score", 0)
            severity = row.get("overall_severity", "unknown")
            drifted = "Yes" if row.get("drift_detected", False) else "No"
            print(f"│ {ts:20s} │ {score:6.3f} │ {severity:10s} │ {drifted:8s} │")

        print("└──────────────────────┴────────┴────────────┴──────────┘")
        print()
        print(f"View details at: {mlflow_uri}")

        return 0

    except Exception as e:
        print(f"❌ ERROR: Could not connect to MLflow: {e}")
        print(f"   Ensure MLflow is running at {mlflow_uri}")
        return 2


def list_references(args, config: dict) -> int:
    """List available reference profiles."""
    print_banner()
    print("📁 Available Reference Profiles")
    print("=" * 62)

    manager = ReferenceManager(storage_path="data/references")
    references = manager.list_references()

    if not references:
        print("No references found.")
        print()
        print("Create one with:")
        print("  python scripts/run_drift_check.py create-reference \\")
        print("      --name baseline_v1 --start 2024-01-01 --end 2024-12-31")
        return 0

    print()
    print("┌──────────────────────┬────────────┬────────────┬──────────┐")
    print("│ Name                 │ Start      │ End        │ Samples  │")
    print("├──────────────────────┼────────────┼────────────┼──────────┤")

    for ref in references:
        name = ref.get("reference_name", "?")[:20]
        start = ref.get("date_start", "?")
        end = ref.get("date_end", "?")
        samples = ref.get("total_samples", 0)
        print(f"│ {name:20s} │ {start:10s} │ {end:10s} │ {samples:>8,} │")

    print("└──────────────────────┴────────────┴────────────┴──────────┘")
    print()
    print(f"Total: {len(references)} reference(s)")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Domain-Shift ML Platform - Drift Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create reference from training data:
    %(prog)s create-reference --name baseline_v1 --start 2024-01-01 --end 2024-12-31

  Run drift check against last 7 days:
    %(prog)s check --reference baseline_v1 --window-hours 168

  Run drift check against specific dates:
    %(prog)s check --reference baseline_v1 --start 2025-01-01 --end 2025-01-15

  Show drift history:
    %(prog)s history --runs 10

  List references:
    %(prog)s list-references
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create-reference subcommand
    create_parser = subparsers.add_parser(
        "create-reference",
        help="Create reference profile from historical data",
    )
    create_parser.add_argument(
        "--name", required=True, help="Name for the reference profile"
    )
    create_parser.add_argument(
        "--start",
        type=validate_date,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    create_parser.add_argument(
        "--end",
        type=validate_date,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    create_parser.add_argument(
        "--features",
        nargs="+",
        help="Features to monitor (default: temperature_2m, humidity, etc.)",
    )

    # check subcommand
    check_parser = subparsers.add_parser(
        "check",
        help="Run drift detection against a reference",
    )
    check_parser.add_argument(
        "--reference", required=True, help="Reference profile name"
    )
    check_parser.add_argument(
        "--window-hours",
        type=int,
        default=168,
        help="Hours of recent data to analyze (default: 168 = 7 days)",
    )
    check_parser.add_argument(
        "--start",
        type=validate_date,
        help="Start date for current window (YYYY-MM-DD)",
    )
    check_parser.add_argument(
        "--end",
        type=validate_date,
        help="End date for current window (YYYY-MM-DD)",
    )
    check_parser.add_argument(
        "--model-name",
        help="Associated model name (for MLflow tagging)",
    )
    check_parser.add_argument(
        "--model-version",
        help="Associated model version (for MLflow tagging)",
    )
    check_parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip logging to MLflow",
    )

    # history subcommand
    history_parser = subparsers.add_parser(
        "history",
        help="Show drift detection history from MLflow",
    )
    history_parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of recent runs to show (default: 10)",
    )
    history_parser.add_argument(
        "--model-name",
        help="Filter by model name",
    )

    # list-references subcommand
    subparsers.add_parser(
        "list-references",
        help="List available reference profiles",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(args.verbose)
    config = load_config()

    try:
        if args.command == "create-reference":
            return create_reference(args, config)
        elif args.command == "check":
            return check_drift(args, config)
        elif args.command == "history":
            return show_history(args, config)
        elif args.command == "list-references":
            return list_references(args, config)
        else:
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
