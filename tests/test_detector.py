"""
Tests for the DriftDetector orchestration class.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.detector import (
    DriftDetector,
    DriftSeverity,
    DriftDetectorError,
)
from src.drift_detection.reference_manager import (
    ReferenceManager,
    ReferenceNotFoundError,
)


@pytest.fixture
def temp_storage_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def reference_manager(temp_storage_path):
    return ReferenceManager(storage_path=temp_storage_path)


@pytest.fixture
def detector_with_reference(reference_manager):
    """Create detector with saved reference profile."""
    np.random.seed(42)
    n = 1000
    ref_df = pd.DataFrame({
        "temperature_2m": np.random.normal(20, 5, n),
        "humidity": np.random.normal(60, 15, n),
        "precipitation": np.random.exponential(2, n),
    })

    features = ["temperature_2m", "humidity", "precipitation"]
    profiles = reference_manager.create_reference_from_dataframe(
        df=ref_df, feature_columns=features,
        reference_name="baseline", store_raw_values=True
    )
    reference_manager.save_reference(profiles, "baseline")

    return DriftDetector(reference_manager=reference_manager), features


class TestDriftDetection:
    """Core drift detection tests."""

    def test_no_drift_identical_distribution(self, detector_with_reference):
        """Identical distributions show minimal drift."""
        detector, _ = detector_with_reference
        np.random.seed(123)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(20, 5, 500),
            "humidity": np.random.normal(60, 15, 500),
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)

        # Most features should be stable (exponential can vary)
        stable_count = sum(
            1 for r in report.feature_results.values()
            if r.severity in (DriftSeverity.NONE, DriftSeverity.LOW)
        )
        assert stable_count >= 2
        assert report.overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MODERATE)

    def test_detect_shifted_distribution(self, detector_with_reference):
        """Shifted distributions are detected."""
        detector, _ = detector_with_reference
        np.random.seed(456)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(35, 5, 500),  # +15 shift
            "humidity": np.random.normal(85, 15, 500),       # +25 shift
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)

        assert report.drift_detected is True
        assert report.feature_results["temperature_2m"].is_drifted is True
        assert report.feature_results["temperature_2m"].mean_shift > 10

    def test_severe_drift_classification(self, detector_with_reference):
        """Very different distributions classified as severe."""
        detector, _ = detector_with_reference
        np.random.seed(789)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(50, 10, 500),
            "humidity": np.random.normal(20, 5, 500),
            "precipitation": np.random.exponential(20, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)

        assert report.overall_severity in (DriftSeverity.SIGNIFICANT, DriftSeverity.SEVERE)
        assert report.overall_drift_score > 0.5


class TestEdgeCases:
    """Edge case handling."""

    def test_insufficient_samples_warning(self, detector_with_reference):
        """Warning when samples below minimum."""
        detector, _ = detector_with_reference
        current = pd.DataFrame({
            "temperature_2m": [20, 21, 22],
            "humidity": [60, 61, 62],
            "precipitation": [1, 2, 3],
        })

        report = detector.detect_drift("baseline", current_data=current)
        assert len(report.warnings) > 0

    def test_missing_reference_error(self, reference_manager):
        """Missing reference raises error."""
        detector = DriftDetector(reference_manager=reference_manager)
        current = pd.DataFrame({"temperature_2m": [20, 21]})

        with pytest.raises(ReferenceNotFoundError):
            detector.detect_drift("nonexistent", current_data=current)

    def test_missing_columns_error(self, detector_with_reference):
        """Missing columns in current data raises error."""
        detector, _ = detector_with_reference
        current = pd.DataFrame({"temperature_2m": [20, 21, 22]})

        with pytest.raises(DriftDetectorError):
            detector.detect_drift("baseline", current_data=current)

    def test_nan_values_handled(self, detector_with_reference):
        """NaN values handled gracefully."""
        detector, _ = detector_with_reference
        np.random.seed(42)
        data = np.random.normal(20, 5, 500)
        data[::10] = np.nan

        current = pd.DataFrame({
            "temperature_2m": data,
            "humidity": np.random.normal(60, 15, 500),
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)
        assert "temperature_2m" in report.feature_results

    def test_partial_feature_selection(self, detector_with_reference):
        """Detect drift on subset of features."""
        detector, _ = detector_with_reference
        np.random.seed(42)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(20, 5, 500),
            "humidity": np.random.normal(60, 15, 500),
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift(
            "baseline", current_data=current,
            feature_columns=["temperature_2m"]
        )

        assert report.n_features_total == 1
        assert "humidity" not in report.feature_results


class TestDriftReport:
    """DriftReport tests."""

    def test_summary_contains_sections(self, detector_with_reference):
        """Summary has expected sections."""
        detector, _ = detector_with_reference
        np.random.seed(42)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(25, 5, 500),
            "humidity": np.random.normal(60, 15, 500),
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)
        summary = report.summary()

        assert "DRIFT DETECTION REPORT" in summary
        assert "Score:" in summary
        assert "Severity:" in summary

    def test_to_dict_serializable(self, detector_with_reference):
        """Report serializes to JSON."""
        import json
        detector, _ = detector_with_reference
        np.random.seed(42)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(20, 5, 500),
            "humidity": np.random.normal(60, 15, 500),
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)
        data = report.to_dict()

        json.dumps(data)  # Should not raise
        assert "overall_drift_score" in data
        assert "feature_results" in data


class TestRecommendations:
    """Recommendation generation tests."""

    def test_stable_data_no_action(self, detector_with_reference):
        """Stable data generates no-action recommendation."""
        detector, _ = detector_with_reference
        np.random.seed(42)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(20, 5, 500),
            "humidity": np.random.normal(60, 15, 500),
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)
        rec_text = " ".join(report.recommendations).lower()
        assert any(w in rec_text for w in ["no action", "stable", "monitoring"])

    def test_drift_retrain_recommendation(self, detector_with_reference):
        """Drifted data recommends retraining."""
        detector, _ = detector_with_reference
        np.random.seed(42)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(45, 5, 500),
            "humidity": np.random.normal(90, 15, 500),
            "precipitation": np.random.exponential(10, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)
        rec_text = " ".join(report.recommendations).lower()
        assert any(w in rec_text for w in ["retrain", "investigate", "urgent"])


class TestConfiguration:
    """Configuration tests."""

    def test_custom_config_merged(self, reference_manager):
        """Custom config merges with defaults."""
        detector = DriftDetector(
            reference_manager=reference_manager,
            config={"detection": {"min_samples": 100}}
        )

        assert detector.config["detection"]["min_samples"] == 100
        assert detector.config["detection"]["method"] == "psi"  # Default preserved

    def test_ks_detection_method(self, detector_with_reference):
        """KS method works for detection."""
        detector, _ = detector_with_reference
        detector.config["detection"]["method"] = "ks"

        np.random.seed(42)
        current = pd.DataFrame({
            "temperature_2m": np.random.normal(35, 5, 500),
            "humidity": np.random.normal(60, 15, 500),
            "precipitation": np.random.exponential(2, 500),
        })

        report = detector.detect_drift("baseline", current_data=current)
        assert report.feature_results["temperature_2m"].ks_p_value < 0.05
