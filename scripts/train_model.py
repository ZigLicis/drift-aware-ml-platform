#!/usr/bin/env python3
"""CLI tool for training weather prediction models.

Usage:
    # Train baseline Ridge model
    python scripts/train_model.py --model ridge

    # Train with custom parameters
    python scripts/train_model.py --model ridge --alpha 0.5

    # Train Random Forest
    python scripts/train_model.py --model random_forest --n-estimators 100

    # Train and promote to staging
    python scripts/train_model.py --model ridge --promote staging

    # Specify date range
    python scripts/train_model.py --model ridge --start 2024-12-01 --end 2025-01-15

    # Compare with production before registering
    python scripts/train_model.py --model ridge --compare-production

    # Dry run (no model registration)
    python scripts/train_model.py --model ridge --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.training.trainer import ModelTrainer, TrainerError

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
    # Reduce noise from third-party libraries
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
    print("""
╔══════════════════════════════════════════════════════════════╗
║       DOMAIN-SHIFT ML PLATFORM - Model Training CLI          ║
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


def print_results(results: dict) -> None:
    """Print training results summary."""
    print("\n" + "=" * 60)
    print(" TRAINING RESULTS")
    print("=" * 60)

    # Run info
    print(f"\n  MLflow Run ID:     {results['run_id'][:8]}...")
    if results.get("model_version"):
        print(f"  Model Version:     {results['model_version']}")
        print(f"  Status:            Registered to Model Registry")
    else:
        print(f"  Status:            Not registered (dry run or below threshold)")

    # Metrics
    print_metrics(results["train_metrics"], "Training Metrics")
    print_metrics(results["val_metrics"], "Validation Metrics")
    print_metrics(results["test_metrics"], "Test Metrics")

    # Feature importance (top 10)
    if results.get("feature_importance"):
        print("\n  Top 10 Features:")
        print("  " + "-" * 40)
        sorted_features = sorted(
            results["feature_importance"].items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        for name, importance in sorted_features:
            bar = "█" * int(abs(importance) * 20 / max(abs(v) for _, v in sorted_features))
            print(f"    {name:25s}: {importance:>8.4f} {bar}")

    # Split info
    if results.get("split_info"):
        print("\n  Data Split:")
        print("  " + "-" * 40)
        info = results["split_info"]
        print(f"    Train:  {info['train_start']} to {info['train_end']} ({info['train_samples']} samples)")
        print(f"    Val:    {info['val_start']} to {info['val_end']} ({info['val_samples']} samples)")
        print(f"    Test:   {info['test_start']} to {info['test_end']} ({info['test_samples']} samples)")

    print("\n" + "=" * 60)


def print_mlflow_instructions(tracking_uri: str, run_id: str) -> None:
    """Print instructions for viewing results in MLflow."""
    print("\n📊 View results in MLflow UI:")
    print(f"   URL: {tracking_uri}")
    print(f"   Run: {tracking_uri}/#/experiments/1/runs/{run_id}")
    print("\n   Or open in browser:")
    print(f"   open {tracking_uri}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train weather prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["ridge", "random_forest", "gradient_boosting"],
        default="ridge",
        help="Model type to train (default: ridge)",
    )

    # Model parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Regularization strength for Ridge (default: 1.0)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of trees for Random Forest/Gradient Boosting",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max tree depth for tree-based models",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for Gradient Boosting",
    )

    # Data selection
    parser.add_argument(
        "--start",
        type=validate_date,
        default=None,
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=validate_date,
        default=None,
        help="End date for training data (YYYY-MM-DD)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file",
    )

    # Model registration
    parser.add_argument(
        "--promote",
        type=str,
        choices=["staging", "production"],
        default=None,
        help="Promote model to specified stage after training",
    )
    parser.add_argument(
        "--compare-production",
        action="store_true",
        help="Only register if better than current production model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate without registering model",
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

    # Get connection strings from environment
    db_url = os.getenv("DATABASE_URL")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        print("   Set it in .env file or export it:")
        print("   export DATABASE_URL='postgresql://user:pass@localhost:5432/dsml'")
        return 1

    # Build model parameters
    model_params = {}
    if args.model == "ridge":
        if args.alpha is not None:
            model_params["alpha"] = args.alpha
    elif args.model == "random_forest":
        if args.n_estimators is not None:
            model_params["n_estimators"] = args.n_estimators
        if args.max_depth is not None:
            model_params["max_depth"] = args.max_depth
    elif args.model == "gradient_boosting":
        if args.n_estimators is not None:
            model_params["n_estimators"] = args.n_estimators
        if args.max_depth is not None:
            model_params["max_depth"] = args.max_depth
        if args.learning_rate is not None:
            model_params["learning_rate"] = args.learning_rate

    print(f"📦 Model:            {args.model}")
    print(f"📁 Config:           {args.config}")
    print(f"🗄️  Database:         {db_url.split('@')[-1] if '@' in db_url else db_url}")
    print(f"📊 MLflow:           {mlflow_uri}")
    if model_params:
        print(f"⚙️  Parameters:       {model_params}")
    if args.start or args.end:
        print(f"📅 Date range:       {args.start or 'auto'} to {args.end or 'auto'}")
    print(f"📝 Register model:   {'No (dry run)' if args.dry_run else 'Yes'}")

    # Initialize trainer
    try:
        print("\n🔄 Initializing trainer...")
        trainer = ModelTrainer(
            config_path=args.config,
            db_connection_string=db_url,
            mlflow_tracking_uri=mlflow_uri,
        )
    except TrainerError as e:
        print(f"❌ Failed to initialize trainer: {e}")
        return 1

    # Run training
    try:
        print("🚀 Starting training pipeline...\n")
        results = trainer.run_training_pipeline(
            model_type=args.model,
            model_params=model_params if model_params else None,
            register_model=not args.dry_run,
            start_date=args.start,
            end_date=args.end,
        )
    except TrainerError as e:
        print(f"\n❌ Training failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        trainer.close()

    # Print results
    print_results(results)

    # Promote if requested
    if args.promote and results.get("model_version"):
        stage = "Staging" if args.promote == "staging" else "Production"
        try:
            trainer.tracker.promote_model(
                trainer.model_registry_name,
                results["model_version"],
                stage,
                description=f"Promoted via CLI on {datetime.now().isoformat()}",
            )
            print(f"\n✅ Model promoted to {stage}")
        except Exception as e:
            print(f"\n⚠️  Failed to promote model: {e}")

    # Print MLflow instructions
    print_mlflow_instructions(mlflow_uri, results["run_id"])

    # Summary
    test_rmse = results["test_metrics"].get("rmse", 0)
    test_r2 = results["test_metrics"].get("r2", 0)

    print(f"\n✅ Training complete!")
    print(f"   Test RMSE: {test_rmse:.3f}°C")
    print(f"   Test R²:   {test_r2:.3f}")

    if test_r2 < 0.5:
        print("\n⚠️  Warning: R² is low. Consider:")
        print("   - Adding more training data")
        print("   - Engineering additional features")
        print("   - Trying a different model type")

    return 0


if __name__ == "__main__":
    sys.exit(main())
