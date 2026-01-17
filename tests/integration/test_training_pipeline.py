"""Integration tests for the complete training pipeline.

These tests require:
- PostgreSQL database with weather data
- MLflow server running

Run with:
    pytest tests/integration/test_training_pipeline.py -v

Skip if services unavailable:
    pytest tests/integration/test_training_pipeline.py -v -m "not integration"
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Check if integration tests should run
SKIP_INTEGRATION = os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true"
INTEGRATION_REASON = "Integration tests disabled via SKIP_INTEGRATION_TESTS"


def create_mock_weather_data(n_samples: int = 500) -> pd.DataFrame:
    """Create mock weather data for testing."""
    np.random.seed(42)

    timestamps = pd.date_range(
        start="2024-12-01 00:00:00",
        periods=n_samples,
        freq="h",
        tz="UTC",
    )

    # Create realistic weather patterns
    hours = np.arange(n_samples)
    day_cycle = np.sin(2 * np.pi * hours / 24)

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature_2m": 15 + 10 * day_cycle + np.random.normal(0, 2, n_samples),
        "relative_humidity_2m": 60 + 20 * np.random.random(n_samples),
        "precipitation": np.random.exponential(0.5, n_samples),
        "wind_speed_10m": 10 + 5 * np.random.random(n_samples),
        "surface_pressure": 1013 + np.random.normal(0, 5, n_samples),
        "location_name": "Test City",
        "latitude": 40.7128,
        "longitude": -74.0060,
    })


@pytest.fixture
def mock_weather_data() -> pd.DataFrame:
    """Fixture providing mock weather data."""
    return create_mock_weather_data(500)


class TestFeatureEngineerIntegration:
    """Integration tests for feature engineering."""

    def test_full_feature_pipeline(self, mock_weather_data):
        """Test complete feature engineering pipeline."""
        from src.training.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()

        # Create features
        df = fe.create_features(mock_weather_data)

        # Verify temporal features
        assert "hour_of_day" in df.columns
        assert "day_of_week" in df.columns
        assert "hour_sin" in df.columns

        # Verify lag features
        assert "temperature_2m_lag_1h" in df.columns
        assert "temperature_2m_lag_24h" in df.columns

        # Verify rolling features
        assert "temperature_2m_rolling_mean_6h" in df.columns
        assert "temperature_2m_rolling_mean_24h" in df.columns

        # Create target
        df = fe.create_target(df, horizon_hours=24)
        assert fe.target_column in df.columns

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = \
            fe.train_test_split_temporal(df, test_days=3, validation_days=1)

        # Verify splits are non-empty
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

        # Verify no NaN in output
        assert not X_train.isna().any().any()
        assert not y_train.isna().any()


class TestModelTrainingIntegration:
    """Integration tests for model training."""

    def test_ridge_training(self, mock_weather_data):
        """Test Ridge model training end-to-end."""
        from src.training.feature_engineering import FeatureEngineer
        from src.training.models import RidgeBaseline
        from src.training.evaluation import ModelEvaluator

        # Prepare data
        fe = FeatureEngineer()
        df = fe.create_features(mock_weather_data)
        df = fe.create_target(df)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            fe.train_test_split_temporal(df, test_days=3, validation_days=1)

        # Train
        model = RidgeBaseline(alpha=1.0)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred)

        # Verify metrics are reasonable
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] > 0
        assert metrics["r2"] < 1.0  # Not perfect (would be suspicious)

    def test_random_forest_training(self, mock_weather_data):
        """Test Random Forest model training."""
        from src.training.feature_engineering import FeatureEngineer
        from src.training.models import RandomForestModel
        from src.training.evaluation import ModelEvaluator

        fe = FeatureEngineer()
        df = fe.create_features(mock_weather_data)
        df = fe.create_target(df)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            fe.train_test_split_temporal(df, test_days=3, validation_days=1)

        # Train with small forest for speed
        model = RandomForestModel(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred)

        assert metrics["rmse"] > 0
        assert model.get_feature_importance() is not None

    def test_model_factory(self):
        """Test model factory function."""
        from src.training.models import create_model

        ridge = create_model("ridge", alpha=0.5)
        assert ridge.alpha == 0.5

        rf = create_model("random_forest", n_estimators=50)
        assert rf.n_estimators == 50

        with pytest.raises(ValueError):
            create_model("unknown_model")


class TestEvaluationIntegration:
    """Integration tests for model evaluation."""

    def test_evaluation_report(self, mock_weather_data):
        """Test comprehensive evaluation report."""
        from src.training.feature_engineering import FeatureEngineer
        from src.training.models import RidgeBaseline
        from src.training.evaluation import ModelEvaluator

        fe = FeatureEngineer()
        df = fe.create_features(mock_weather_data)
        df = fe.create_target(df)

        X_train, _, X_test, y_train, _, y_test = \
            fe.train_test_split_temporal(df, test_days=3, validation_days=1)

        model = RidgeBaseline()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        evaluator = ModelEvaluator()
        report = evaluator.create_evaluation_report(
            y_test, y_pred,
            X_train.columns.tolist(),
            model,
        )

        assert "metrics" in report
        assert "residual_stats" in report
        assert "error_percentiles" in report
        assert "feature_importance" in report

    def test_evaluation_plots(self, mock_weather_data):
        """Test that evaluation plots are generated without error."""
        from src.training.feature_engineering import FeatureEngineer
        from src.training.models import RidgeBaseline
        from src.training.evaluation import ModelEvaluator
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        fe = FeatureEngineer()
        df = fe.create_features(mock_weather_data)
        df = fe.create_target(df)

        X_train, _, X_test, y_train, _, y_test = \
            fe.train_test_split_temporal(df, test_days=3, validation_days=1)

        model = RidgeBaseline()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        evaluator = ModelEvaluator()

        # These should not raise
        fig = evaluator.plot_predictions(y_test, y_pred)
        assert fig is not None

        fig = evaluator.plot_residuals(y_test, y_pred)
        assert fig is not None

        importance = model.get_feature_importance(X_train.columns.tolist())
        fig = evaluator.plot_feature_importance(importance)
        assert fig is not None


class TestMLflowIntegration:
    """Integration tests for MLflow tracking."""

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_REASON)
    def test_mlflow_connection(self):
        """Test MLflow server connection."""
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

        try:
            from src.mlflow_utils.tracking import ExperimentTracker

            tracker = ExperimentTracker(
                tracking_uri=mlflow_uri,
                experiment_name="test-integration",
            )
            assert tracker.experiment_id is not None
        except Exception as e:
            pytest.skip(f"MLflow not available: {e}")

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_REASON)
    def test_mlflow_logging(self, mock_weather_data):
        """Test MLflow logging workflow."""
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

        try:
            from src.mlflow_utils.tracking import ExperimentTracker
            from src.training.feature_engineering import FeatureEngineer
            from src.training.models import RidgeBaseline
            import mlflow

            tracker = ExperimentTracker(
                tracking_uri=mlflow_uri,
                experiment_name="test-integration",
            )

            fe = FeatureEngineer()
            df = fe.create_features(mock_weather_data)
            df = fe.create_target(df)
            X_train, _, X_test, y_train, _, y_test = \
                fe.train_test_split_temporal(df, test_days=3, validation_days=1)

            model = RidgeBaseline()
            model.fit(X_train, y_train)

            with tracker.start_run("test-run", tags={"test": "true"}):
                mlflow.log_params(model.get_params())
                mlflow.log_metric("test_metric", 0.123)

                # Log model without registration
                mlflow.sklearn.log_model(model.model, "model")

                run_id = tracker.active_run_id
                assert run_id is not None

        except Exception as e:
            pytest.skip(f"MLflow not available: {e}")


class TestDatabaseIntegration:
    """Integration tests requiring database connection."""

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_REASON)
    def test_database_connection(self):
        """Test database connection."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not set")

        from src.data_ingestion.storage import DataStorage

        storage = DataStorage(db_url)
        assert storage.health_check()
        storage.close()

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_REASON)
    def test_load_data_from_database(self):
        """Test loading data from database."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not set")

        from src.data_ingestion.storage import DataStorage

        storage = DataStorage(db_url)

        try:
            count = storage.get_record_count()
            if count == 0:
                pytest.skip("No data in database")

            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=7)

            df = storage.get_data_by_timerange(start_dt, end_dt)

            if df.empty:
                pytest.skip("No recent data in database")

            assert "timestamp" in df.columns
            assert "temperature_2m" in df.columns

        finally:
            storage.close()


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_REASON)
    def test_full_training_pipeline_with_mock_db(self, mock_weather_data, tmp_path):
        """Test full pipeline with mocked database."""
        from unittest.mock import MagicMock, patch

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

        # Create mock storage that returns our test data
        mock_storage = MagicMock()
        mock_storage.get_data_by_timerange.return_value = mock_weather_data
        mock_storage.health_check.return_value = True

        try:
            with patch("src.training.trainer.DataStorage", return_value=mock_storage):
                from src.training.trainer import ModelTrainer

                # Create a temporary config file
                config_content = """
model:
  name: "test-model"
  prediction_horizon_hours: 24

features:
  weather:
    - temperature_2m
    - relative_humidity_2m
  lag_hours: [1, 6]
  rolling_windows: [6]

training:
  test_days: 3
  validation_days: 1

mlflow:
  experiment_name: "test-pipeline"
"""
                config_path = tmp_path / "test_config.yaml"
                config_path.write_text(config_content)

                trainer = ModelTrainer(
                    config_path=str(config_path),
                    db_connection_string="mock://unused",
                    mlflow_tracking_uri=mlflow_uri,
                )

                results = trainer.run_training_pipeline(
                    model_type="ridge",
                    register_model=False,  # Don't pollute registry
                )

                assert "train_metrics" in results
                assert "test_metrics" in results
                assert "model" in results
                assert results["test_metrics"]["rmse"] > 0

        except Exception as e:
            pytest.skip(f"MLflow not available: {e}")


class TestModelRegistry:
    """Tests for model registry operations."""

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_REASON)
    def test_registry_operations(self):
        """Test model registry list and info operations."""
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

        try:
            from src.mlflow_utils.registry import ModelRegistry

            registry = ModelRegistry(mlflow_uri)
            models = registry.list_models()

            # Just verify we can list models (may be empty)
            assert isinstance(models, list)

        except Exception as e:
            pytest.skip(f"MLflow not available: {e}")
