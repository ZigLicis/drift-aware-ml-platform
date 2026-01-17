#!/usr/bin/env python3
"""CLI tool for evaluating trained models from the registry.

Usage:
    # Evaluate latest production model on recent data
    python scripts/evaluate_model.py --model weather-forecaster --stage Production

    # Evaluate staging model
    python scripts/evaluate_model.py --model weather-forecaster --stage Staging

    # Evaluate specific version
    python scripts/evaluate_model.py --model weather-forecaster --version 1

    # Evaluate on specific date range
    python scripts/evaluate_model.py --model weather-forecaster --stage Production \\
        --start 2025-01-10 --end 2025-01-15

    # Compare two versions
    python scripts/evaluate_model.py --model weather-forecaster --compare 1 2
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.data_ingestion.storage import DataStorage
from src.mlflow_utils.registry import ModelRegistry, ModelNotFoundError
from src.training.evaluation import ModelEvaluator
from src.training.feature_engineering import FeatureEngineer

# Load environment variables
load_dotenv()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
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
║       DOMAIN-SHIFT ML PLATFORM - Model Evaluation CLI        ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_metrics(metrics: dict, title: str) -> None:
    """Print metrics in a formatted table."""
    print(f"\n  {title}:")
    print("  " + "-" * 40)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"    {name:20s}: {value:>10.4f}")
        else:
            print(f"    {name:20s}: {value:>10}")


def print_model_info(info: dict) -> None:
    """Print model registry information."""
    print("\n" + "=" * 60)
    print(f" MODEL: {info['name']}")
    print("=" * 60)

    if info.get("description"):
        print(f"  Description: {info['description']}")

    print("\n  Versions:")
    print("  " + "-" * 50)

    for v in info.get("versions", [])[:5]:  # Show latest 5
        stage = v.get("stage", "None")
        stage_badge = {
            "Production": "🟢",
            "Staging": "🟡",
            "Archived": "⚫",
            "None": "⚪",
        }.get(stage, "⚪")

        metrics = v.get("metrics", {})
        rmse = metrics.get("test_rmse", metrics.get("rmse", "N/A"))
        r2 = metrics.get("test_r2", metrics.get("r2", "N/A"))

        rmse_str = f"{rmse:.3f}" if isinstance(rmse, float) else rmse
        r2_str = f"{r2:.3f}" if isinstance(r2, float) else r2

        print(f"    v{v['version']:3d} {stage_badge} {stage:12s}  RMSE: {rmse_str:>8}  R²: {r2_str:>8}")


def print_comparison(comparison: dict) -> None:
    """Print model comparison results."""
    print("\n" + "=" * 60)
    print(f" MODEL COMPARISON: v{comparison['version_a']['version']} vs v{comparison['version_b']['version']}")
    print("=" * 60)

    print(f"\n  Version A (v{comparison['version_a']['version']}) - {comparison['version_a']['stage']}")
    print_metrics(comparison['version_a']['metrics'], "Metrics")

    print(f"\n  Version B (v{comparison['version_b']['version']}) - {comparison['version_b']['stage']}")
    print_metrics(comparison['version_b']['metrics'], "Metrics")

    print("\n  Comparison (B - A):")
    print("  " + "-" * 40)
    for metric, diff in comparison['comparison'].items():
        better = comparison['better_version'].get(metric, "N/A")
        indicator = "✅" if str(better) == str(comparison['version_b']['version']) else "❌"
        print(f"    {metric:20s}: {diff:>+10.4f}  {indicator} v{better} is better")


def evaluate_model_on_data(
    model,
    storage: DataStorage,
    feature_engineer: FeatureEngineer,
    evaluator: ModelEvaluator,
    start_date: str | None,
    end_date: str | None,
) -> dict:
    """Load data and evaluate model."""
    # Parse dates
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=7)

    print(f"\n📅 Evaluating on data from {start_dt.date()} to {end_dt.date()}")

    # Load data
    df = storage.get_data_by_timerange(start_dt, end_dt)

    if df.empty:
        raise ValueError(f"No data found between {start_dt.date()} and {end_dt.date()}")

    print(f"📦 Loaded {len(df)} records")

    # Create features
    df = feature_engineer.create_features(df)
    df = feature_engineer.create_target(df)

    # Get feature matrix
    X, y = feature_engineer.get_feature_matrix(df)
    print(f"📊 Prepared {len(X)} samples with {len(X.columns)} features")

    # Predict
    y_pred = model.predict(X)

    # Evaluate
    metrics = evaluator.evaluate(y, y_pred)

    return {
        "metrics": metrics,
        "n_samples": len(X),
        "date_range": f"{start_dt.date()} to {end_dt.date()}",
        "y_true": y,
        "y_pred": y_pred,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models from the registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="weather-forecaster",
        help="Registered model name (default: weather-forecaster)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["Production", "Staging", "None", "Archived"],
        default=None,
        help="Model stage to evaluate",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific model version to evaluate",
    )

    # Data selection
    parser.add_argument(
        "--start",
        type=validate_date,
        default=None,
        help="Start date for evaluation data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=validate_date,
        default=None,
        help="End date for evaluation data (YYYY-MM-DD)",
    )

    # Comparison
    parser.add_argument(
        "--compare",
        type=int,
        nargs=2,
        metavar=("V1", "V2"),
        help="Compare two model versions",
    )

    # Info only
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model registry info without evaluation",
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    print_banner()

    # Get connection strings
    db_url = os.getenv("DATABASE_URL")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    if not db_url and not args.info and not args.compare:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return 1

    # Initialize registry
    try:
        registry = ModelRegistry(mlflow_uri)
    except Exception as e:
        print(f"❌ Failed to connect to MLflow: {e}")
        return 1

    # Just show comparison
    if args.compare:
        try:
            comparison = registry.compare_models(args.model, args.compare[0], args.compare[1])
            print_comparison(comparison)
            return 0
        except ModelNotFoundError as e:
            print(f"❌ {e}")
            return 1

    # Just show info
    if args.info:
        try:
            info = registry.get_model_info(args.model)
            print_model_info(info)
            return 0
        except ModelNotFoundError as e:
            print(f"❌ {e}")
            return 1

    # Load model
    print(f"🔍 Loading model: {args.model}")
    try:
        if args.version:
            model = registry.get_model_by_version(args.model, args.version)
            print(f"   Version: {args.version}")
        elif args.stage:
            model = registry.get_latest_model(args.model, args.stage)
            if model is None:
                print(f"❌ No model found in stage '{args.stage}'")
                return 1
            print(f"   Stage: {args.stage}")
        else:
            # Default to Production, fallback to Staging
            model = registry.get_latest_model(args.model, "Production")
            if model is None:
                model = registry.get_latest_model(args.model, "Staging")
                if model is None:
                    print("❌ No model found in Production or Staging")
                    return 1
                print("   Stage: Staging (no Production model)")
            else:
                print("   Stage: Production")
    except ModelNotFoundError as e:
        print(f"❌ {e}")
        return 1

    # Initialize components
    storage = DataStorage(db_url)
    feature_engineer = FeatureEngineer()
    evaluator = ModelEvaluator()

    # Evaluate
    try:
        results = evaluate_model_on_data(
            model,
            storage,
            feature_engineer,
            evaluator,
            args.start,
            args.end,
        )
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        storage.close()

    # Print results
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Model:        {args.model}")
    print(f"  Date Range:   {results['date_range']}")
    print(f"  Samples:      {results['n_samples']}")

    print_metrics(results["metrics"], "Metrics")

    # Summary
    rmse = results["metrics"].get("rmse", 0)
    r2 = results["metrics"].get("r2", 0)

    print("\n" + "=" * 60)
    print(f"\n✅ Evaluation complete!")
    print(f"   RMSE: {rmse:.3f}°C")
    print(f"   R²:   {r2:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
