"""
Tests for the DriftMLflowLogger MLflow integration.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.detector import (
    DriftReport,
    DriftSeverity,
    FeatureDriftResult,
)
from src.drift_detection.mlflow_logger import (
    DriftMLflowLogger,
    DriftMLflowLoggerError,
    MLflowConnectionError,
)


@pytest.fixture
def sample_feature_results():
    """Create sample feature drift results for testing."""
    return {
        "temperature_2m": FeatureDriftResult(
            feature_name="temperature_2m",
            psi=0.25,
            ks_statistic=0.15,
            ks_p_value=0.02,
            js_divergence=0.12,
            wasserstein=2.5,
            severity=DriftSeverity.SIGNIFICANT,
            is_drifted=True,
            reference_mean=20.0,
            reference_std=5.0,
            current_mean=25.0,
            current_std=6.0,
            mean_shift=5.0,
            message="temperature_2m significant drift: PSI=0.250, mean +5.00",
        ),
        "humidity": FeatureDriftResult(
            feature_name="humidity",
            psi=0.08,
            ks_statistic=0.05,
            ks_p_value=0.45,
            js_divergence=0.03,
            wasserstein=1.2,
            severity=DriftSeverity.NONE,
            is_drifted=False,
            reference_mean=60.0,
            reference_std=15.0,
            current_mean=62.0,
            current_std=14.0,
            mean_shift=2.0,
            message="humidity stable (PSI=0.0800)",
        ),
        "precipitation": FeatureDriftResult(
            feature_name="precipitation",
            psi=0.12,
            ks_statistic=0.10,
            ks_p_value=0.08,
            js_divergence=0.06,
            wasserstein=0.8,
            severity=DriftSeverity.LOW,
            is_drifted=False,
            reference_mean=2.0,
            reference_std=3.0,
            current_mean=2.5,
            current_std=3.5,
            mean_shift=0.5,
            message="precipitation low drift: PSI=0.120, mean +0.50",
        ),
    }


@pytest.fixture
def sample_drift_report(sample_feature_results):
    """Create a sample DriftReport for testing."""
    now = datetime.now(timezone.utc)
    return DriftReport(
        report_id="test1234",
        timestamp=now,
        reference_name="baseline_v1",
        reference_samples=1000,
        current_samples=500,
        current_window_start=now - timedelta(hours=24),
        current_window_end=now,
        feature_results=sample_feature_results,
        overall_drift_score=0.45,
        overall_severity=DriftSeverity.SIGNIFICANT,
        n_features_drifted=1,
        n_features_total=3,
        drift_detected=True,
        recommendations=["Retrain model: 1/3 features show significant drift"],
        warnings=[],
    )


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing without actual server."""
    with patch("src.drift_detection.mlflow_logger.mlflow") as mock_mlflow:
        # Setup experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Setup run context
        mock_run = MagicMock()
        mock_run.info.run_id = "run_abc123def456"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Setup search experiments (for connection verification)
        mock_mlflow.search_experiments.return_value = []

        yield mock_mlflow


@pytest.fixture
def mock_mlflow_client():
    """Mock MlflowClient for testing."""
    with patch("src.drift_detection.mlflow_logger.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


class TestDriftMLflowLoggerInit:
    """Tests for DriftMLflowLogger initialization."""

    def test_init_success(self, mock_mlflow, mock_mlflow_client):
        """Successful initialization with valid MLflow connection."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test-drift"
        )

        assert logger.tracking_uri == "http://localhost:5001"
        assert logger.experiment_name == "test-drift"
        assert logger.experiment_id == "exp123"
        assert logger.is_connected is True

    def test_init_creates_new_experiment(self, mock_mlflow, mock_mlflow_client):
        """Creates new experiment when it doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new_exp_456"

        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="new-experiment"
        )

        assert logger.experiment_id == "new_exp_456"
        mock_mlflow.create_experiment.assert_called_once()

    def test_init_connection_failure(self, mock_mlflow, mock_mlflow_client):
        """Raises MLflowConnectionError when server is unreachable."""
        from mlflow.exceptions import MlflowException
        mock_mlflow.search_experiments.side_effect = MlflowException("Connection refused")

        with pytest.raises(MLflowConnectionError) as exc_info:
            DriftMLflowLogger(
                tracking_uri="http://invalid:9999",
                experiment_name="test"
            )

        assert "Connection refused" in str(exc_info.value)


class TestLogDriftReport:
    """Tests for logging drift reports to MLflow."""

    def test_log_drift_report_creates_run(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Logging creates an MLflow run with correct data."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test-drift"
        )

        run_id = logger.log_drift_report(sample_drift_report)

        assert run_id == "run_abc123def456"
        mock_mlflow.start_run.assert_called_once()

    def test_log_drift_report_logs_parameters(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Verifies correct parameters are logged."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test-drift"
        )

        logger.log_drift_report(
            sample_drift_report,
            model_name="weather-model",
            model_version="3"
        )

        # Check log_params was called
        mock_mlflow.log_params.assert_called()
        call_args = mock_mlflow.log_params.call_args[0][0]

        assert call_args["reference_name"] == "baseline_v1"
        assert call_args["n_features_monitored"] == 3
        assert call_args["model_name"] == "weather-model"
        assert call_args["model_version"] == "3"

    def test_log_drift_report_logs_metrics(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Verifies correct metrics are logged."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test-drift"
        )

        logger.log_drift_report(sample_drift_report)

        # log_metrics should be called multiple times (overall + per-feature)
        assert mock_mlflow.log_metrics.call_count >= 2

    def test_log_drift_report_with_custom_tags(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Custom tags are included in the run."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test-drift"
        )

        logger.log_drift_report(
            sample_drift_report,
            tags={"environment": "production", "region": "us-east-1"}
        )

        # Verify start_run was called with tags
        call_kwargs = mock_mlflow.start_run.call_args[1]
        assert "drift_detected" in call_kwargs["tags"]
        assert call_kwargs["tags"]["drift_detected"] == "true"

    def test_log_drift_report_logs_artifacts(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Artifacts (JSON, summary, heatmap) are logged."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test-drift"
        )

        logger.log_drift_report(sample_drift_report)

        # Should log multiple artifacts
        assert mock_mlflow.log_artifact.call_count >= 2


class TestMetricNaming:
    """Tests for consistent metric naming conventions."""

    def test_sanitize_metric_name(self, mock_mlflow, mock_mlflow_client):
        """Feature names are sanitized correctly for metrics."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        # Test various input formats
        assert logger._sanitize_metric_name("temperature_2m") == "temperature_2m"
        assert logger._sanitize_metric_name("wind-speed") == "wind_speed"
        assert logger._sanitize_metric_name("humidity.avg") == "humidity_avg"
        assert logger._sanitize_metric_name("feature with spaces") == "feature_with_spaces"

    def test_sanitize_truncates_long_names(self, mock_mlflow, mock_mlflow_client):
        """Long feature names are truncated."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        long_name = "a" * 100
        sanitized = logger._sanitize_metric_name(long_name)

        assert len(sanitized) == 50

    def test_per_feature_metrics_naming(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Per-feature metrics follow {feature}_{metric} convention."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        logger.log_drift_report(sample_drift_report)

        # Get all logged metric calls
        logged_metrics = {}
        for call in mock_mlflow.log_metrics.call_args_list:
            logged_metrics.update(call[0][0])

        # Check naming convention
        expected_patterns = [
            "temperature_2m_psi",
            "temperature_2m_ks_statistic",
            "humidity_psi",
            "precipitation_js_divergence",
        ]

        for pattern in expected_patterns:
            assert pattern in logged_metrics, f"Expected metric {pattern} not found"


class TestDriftHeatmap:
    """Tests for drift heatmap creation."""

    def test_create_drift_heatmap_returns_path(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Heatmap creation returns a valid file path."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = logger.create_drift_heatmap(sample_drift_report, output_dir=tmpdir)

            assert Path(path).exists()
            assert path.endswith(".png")

    def test_create_drift_heatmap_file_is_valid_image(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Heatmap file is a valid PNG image."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = logger.create_drift_heatmap(sample_drift_report, output_dir=tmpdir)

            # Check file has content and starts with PNG magic bytes
            with open(path, "rb") as f:
                header = f.read(8)

            png_magic = b"\x89PNG\r\n\x1a\n"
            assert header == png_magic, "File is not a valid PNG"

    def test_create_drift_heatmap_uses_tempdir_by_default(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Heatmap is created in temp directory when output_dir not specified."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        path = logger.create_drift_heatmap(sample_drift_report)

        assert Path(path).exists()
        # Clean up
        Path(path).unlink()


class TestDriftTimeline:
    """Tests for drift timeline creation."""

    def test_create_drift_timeline_with_history(self, mock_mlflow, mock_mlflow_client):
        """Timeline is created when drift history is available."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        # Mock drift history
        mock_runs = []
        for i in range(5):
            mock_run = MagicMock()
            mock_run.info.run_id = f"run_{i}"
            mock_run.info.start_time = (datetime.now(timezone.utc) - timedelta(days=i)).timestamp() * 1000
            mock_run.data.tags = {"drift_detected": "false", "overall_severity": "none"}
            mock_run.data.metrics = {
                "temperature_2m_psi": 0.05 + i * 0.02,
                "overall_drift_score": 0.1 + i * 0.05,
            }
            mock_runs.append(mock_run)

        mock_mlflow_client.search_runs.return_value = mock_runs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = logger.create_drift_timeline("temperature_2m", n_recent_runs=5, output_dir=tmpdir)

            assert Path(path).exists()
            assert "temperature_2m" in path

    def test_create_drift_timeline_no_history_raises_error(self, mock_mlflow, mock_mlflow_client):
        """Error raised when no drift history available."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        mock_mlflow_client.search_runs.return_value = []

        with pytest.raises(DriftMLflowLoggerError) as exc_info:
            logger.create_drift_timeline("temperature_2m")

        assert "No drift history available" in str(exc_info.value)


class TestDriftHistory:
    """Tests for retrieving drift history."""

    def test_get_drift_history_returns_dataframe(self, mock_mlflow, mock_mlflow_client):
        """History retrieval returns a DataFrame."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_run.info.start_time = datetime.now(timezone.utc).timestamp() * 1000
        mock_run.data.tags = {"drift_detected": "true", "overall_severity": "moderate"}
        mock_run.data.metrics = {"overall_drift_score": 0.35, "n_features_drifted": 2}

        mock_mlflow_client.search_runs.return_value = [mock_run]

        df = logger.get_drift_history(n_runs=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "run_id" in df.columns
        assert "drift_detected" in df.columns
        assert "overall_drift_score" in df.columns

    def test_get_drift_history_filters_by_model(self, mock_mlflow, mock_mlflow_client):
        """History can be filtered by model name."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        mock_mlflow_client.search_runs.return_value = []

        logger.get_drift_history(n_runs=10, model_name="weather-model")

        # Verify filter was applied
        call_kwargs = mock_mlflow_client.search_runs.call_args[1]
        assert "model_name" in call_kwargs["filter_string"]

    def test_get_drift_history_empty_returns_empty_dataframe(self, mock_mlflow, mock_mlflow_client):
        """Empty history returns empty DataFrame (not error)."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        mock_mlflow_client.search_runs.return_value = []

        df = logger.get_drift_history()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestGracefulErrorHandling:
    """Tests for graceful error handling when MLflow is unavailable."""

    def test_log_drift_report_safe_returns_none_on_failure(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Safe logging returns None instead of raising on error."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        # Make start_run raise an error
        from mlflow.exceptions import MlflowException
        mock_mlflow.start_run.side_effect = MlflowException("Server error")

        result = logger.log_drift_report_safe(sample_drift_report)

        assert result is None

    def test_log_drift_report_safe_returns_run_id_on_success(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Safe logging returns run_id when successful."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        result = logger.log_drift_report_safe(sample_drift_report)

        assert result == "run_abc123def456"

    def test_get_drift_history_handles_mlflow_error(self, mock_mlflow, mock_mlflow_client):
        """History retrieval returns empty DataFrame on error (not exception)."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="test"
        )

        from mlflow.exceptions import MlflowException
        mock_mlflow_client.search_runs.side_effect = MlflowException("Query error")

        df = logger.get_drift_history()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestIntegration:
    """Integration-style tests with multiple components."""

    def test_full_logging_workflow(self, mock_mlflow, mock_mlflow_client, sample_drift_report):
        """Complete workflow: log report, check artifacts, verify metrics."""
        logger = DriftMLflowLogger(
            tracking_uri="http://localhost:5001",
            experiment_name="drift-monitoring"
        )

        # Log report
        run_id = logger.log_drift_report(
            sample_drift_report,
            model_name="weather-predictor",
            model_version="2",
            tags={"environment": "staging"}
        )

        # Verify run was created
        assert run_id is not None

        # Verify correct experiment was used
        call_kwargs = mock_mlflow.start_run.call_args[1]
        assert call_kwargs["experiment_id"] == "exp123"

        # Verify run name follows convention
        assert "drift-test1234" in call_kwargs["run_name"]

    def test_report_to_dict_serializable(self, sample_drift_report):
        """Report can be serialized to JSON (for artifact logging)."""
        report_dict = sample_drift_report.to_dict()

        import json
        json_str = json.dumps(report_dict, default=str)

        assert "test1234" in json_str
        assert "baseline_v1" in json_str
        assert "temperature_2m" in json_str
