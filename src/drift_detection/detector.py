"""Drift detection orchestrator for domain shift monitoring."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.drift_detection.reference_manager import ReferenceManager, ReferenceProfile
from src.drift_detection.statistical_tests import StatisticalTests

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Drift severity levels indicating urgency of action."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"


# Severity ordering for comparisons
SEVERITY_ORDER = list(DriftSeverity)
SEVERITY_SCORES = {s: i * 0.25 for i, s in enumerate(DriftSeverity)}


def _to_serializable(obj: Any) -> Any:
    """Convert dataclass to dict, handling enums and datetimes."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    return obj


@dataclass
class FeatureDriftResult:
    """Drift detection result for a single feature."""
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    js_divergence: float
    wasserstein: float
    severity: DriftSeverity
    is_drifted: bool
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    mean_shift: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


@dataclass
class DriftReport:
    """Complete drift detection report."""
    report_id: str
    timestamp: datetime
    reference_name: str
    reference_samples: int
    current_samples: int
    current_window_start: datetime
    current_window_end: datetime
    feature_results: Dict[str, FeatureDriftResult]
    overall_drift_score: float
    overall_severity: DriftSeverity
    n_features_drifted: int
    n_features_total: int
    drift_detected: bool
    recommendations: List[str]
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 50,
            "DRIFT DETECTION REPORT",
            f"ID: {self.report_id} | Reference: {self.reference_name}",
            f"Samples: {self.current_samples:,} current vs {self.reference_samples:,} reference",
            "-" * 50,
            f"Score: {self.overall_drift_score:.3f} | Severity: {self.overall_severity.value.upper()}",
            f"Drifted: {self.n_features_drifted}/{self.n_features_total} | Action: {'YES' if self.drift_detected else 'NO'}",
        ]

        if self.warnings:
            lines.append(f"Warnings: {'; '.join(self.warnings)}")

        lines.append("-" * 50)
        for r in sorted(self.feature_results.values(),
                       key=lambda x: SEVERITY_ORDER.index(x.severity), reverse=True):
            status = "DRIFT" if r.is_drifted else "OK"
            lines.append(f"[{status:5}] {r.feature_name}: PSI={r.psi:.4f} ({r.severity.value})")

        if self.recommendations:
            lines.append("-" * 50)
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


class DriftDetectorError(Exception):
    """Base exception for DriftDetector errors."""
    pass


class InsufficientDataError(DriftDetectorError):
    """Raised when there's not enough data for reliable drift detection."""
    pass


class DriftDetector:
    """Main drift detection orchestrator."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "thresholds": {
            "psi": {"low": 0.1, "moderate": 0.15, "significant": 0.2, "severe": 0.5},
            "ks_p_value": 0.05,
            "js_divergence": {"low": 0.05, "moderate": 0.1, "significant": 0.2, "severe": 0.4},
        },
        "detection": {"method": "psi", "min_samples": 50, "features_drifted_threshold": 0.3},
        "monitoring": {"window_size_hours": 168},
    }

    def __init__(
        self,
        reference_manager: ReferenceManager,
        db_connection_string: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reference_manager = reference_manager
        self.db_connection_string = db_connection_string
        self.config = self._merge_config(self.DEFAULT_CONFIG, config or {})
        self._storage = None
        logger.info(f"DriftDetector initialized: method={self.config['detection']['method']}")

    def _merge_config(self, default: Dict, override: Dict) -> Dict:
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    @property
    def storage(self):
        if self._storage is None and self.db_connection_string:
            from src.data_ingestion.storage import DataStorage
            self._storage = DataStorage(self.db_connection_string)
        return self._storage

    def detect_drift(
        self,
        reference_name: str,
        current_data: Optional[pd.DataFrame] = None,
        current_window_hours: Optional[int] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> DriftReport:
        """Run drift detection comparing current data to reference."""
        report_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc)
        warnings: List[str] = []

        logger.info(f"Starting drift detection (report={report_id}) against '{reference_name}'")

        # Load reference
        reference_profiles = self.reference_manager.load_reference(reference_name)

        # Determine features to check
        if feature_columns is None:
            feature_columns = list(reference_profiles.keys())
        else:
            available = set(reference_profiles.keys())
            missing = set(feature_columns) - available
            if missing:
                warnings.append(f"Features not in reference: {list(missing)}")
                feature_columns = [f for f in feature_columns if f in available]

        if not feature_columns:
            raise DriftDetectorError("No valid features to analyze")

        # Get current data
        current_data, window_start, window_end = self._get_current_data(
            current_data, current_window_hours, feature_columns, timestamp
        )

        current_samples = len(current_data)
        min_samples = self.config["detection"]["min_samples"]
        if current_samples < min_samples:
            warnings.append(f"Low samples ({current_samples} < {min_samples}): results may be unreliable")

        # Run tests per feature
        feature_results: Dict[str, FeatureDriftResult] = {}
        reference_samples = 0

        for feature_name in feature_columns:
            profile = reference_profiles[feature_name]
            reference_samples = max(reference_samples, profile.n_samples)

            if feature_name not in current_data.columns:
                warnings.append(f"Feature '{feature_name}' missing from current data")
                continue

            values = current_data[feature_name].dropna().values
            if len(values) == 0:
                warnings.append(f"Feature '{feature_name}' has no valid values")
                continue

            feature_results[feature_name] = self._analyze_feature(feature_name, profile, values)

        if not feature_results:
            raise DriftDetectorError("No features could be analyzed")

        # Calculate overall metrics
        n_drifted = sum(1 for r in feature_results.values() if r.is_drifted)
        n_total = len(feature_results)
        overall_score = sum(SEVERITY_SCORES[r.severity] for r in feature_results.values()) / n_total
        overall_severity = max(feature_results.values(), key=lambda r: SEVERITY_ORDER.index(r.severity)).severity

        drift_ratio = n_drifted / n_total
        threshold = self.config["detection"]["features_drifted_threshold"]
        drift_detected = drift_ratio >= threshold or overall_severity in (DriftSeverity.SIGNIFICANT, DriftSeverity.SEVERE)

        recommendations = self._get_recommendations(overall_severity, drift_detected, n_drifted, n_total, feature_results)

        report = DriftReport(
            report_id=report_id,
            timestamp=timestamp,
            reference_name=reference_name,
            reference_samples=reference_samples,
            current_samples=current_samples,
            current_window_start=window_start,
            current_window_end=window_end,
            feature_results=feature_results,
            overall_drift_score=overall_score,
            overall_severity=overall_severity,
            n_features_drifted=n_drifted,
            n_features_total=n_total,
            drift_detected=drift_detected,
            recommendations=recommendations,
            warnings=warnings,
        )

        logger.info(f"Drift detection complete: score={overall_score:.3f}, severity={overall_severity.value}, action={drift_detected}")
        return report

    def _get_current_data(
        self,
        current_data: Optional[pd.DataFrame],
        window_hours: Optional[int],
        feature_columns: List[str],
        timestamp: datetime,
    ) -> tuple[pd.DataFrame, datetime, datetime]:
        """Get current data from DataFrame or database."""
        if current_data is not None:
            missing = set(feature_columns) - set(current_data.columns)
            if missing:
                raise DriftDetectorError(f"Current data missing columns: {missing}")

            if "timestamp" in current_data.columns:
                start = current_data["timestamp"].min()
                end = current_data["timestamp"].max()
            else:
                hours = window_hours or self.config["monitoring"]["window_size_hours"]
                start = timestamp - timedelta(hours=hours)
                end = timestamp
        else:
            if self.storage is None:
                raise DriftDetectorError("No current_data provided and no database configured")

            hours = window_hours or self.config["monitoring"]["window_size_hours"]
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=hours)

            current_data = self.storage.get_data_by_timerange(start=start, end=end)
            if current_data.empty:
                raise InsufficientDataError(f"No data in database for {start} to {end}")

        # Normalize timestamps
        if isinstance(start, pd.Timestamp):
            start = start.to_pydatetime()
        if isinstance(end, pd.Timestamp):
            end = end.to_pydatetime()
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        return current_data, start, end

    def _analyze_feature(
        self, name: str, profile: ReferenceProfile, current: np.ndarray
    ) -> FeatureDriftResult:
        """Analyze drift for a single feature."""
        # Get reference values
        ref = profile.values if profile.values is not None else self._sample_from_profile(profile, len(current))

        # Run tests
        psi = StatisticalTests.calculate_psi(ref, current)
        ks_stat, ks_pval = StatisticalTests.ks_test(ref, current)
        js = StatisticalTests.jensen_shannon_divergence(ref, current)
        wass = StatisticalTests.wasserstein_distance(ref, current)

        # Stats
        cur_mean = float(np.mean(current))
        cur_std = float(np.std(current, ddof=1)) if len(current) > 1 else 0.0
        mean_shift = abs(cur_mean - profile.mean)

        # Determine severity from PSI (primary method)
        severity = self._get_severity(psi, js)
        is_drifted = self._check_drifted(psi, js, ks_pval)

        # Message
        if severity == DriftSeverity.NONE:
            msg = f"{name} stable (PSI={psi:.4f})"
        else:
            direction = "+" if cur_mean > profile.mean else "-"
            msg = f"{name} {severity.value} drift: PSI={psi:.3f}, mean {direction}{mean_shift:.2f}"

        return FeatureDriftResult(
            feature_name=name, psi=psi, ks_statistic=ks_stat, ks_p_value=ks_pval,
            js_divergence=js, wasserstein=wass, severity=severity, is_drifted=is_drifted,
            reference_mean=profile.mean, reference_std=profile.std,
            current_mean=cur_mean, current_std=cur_std, mean_shift=mean_shift, message=msg,
        )

    def _sample_from_profile(self, profile: ReferenceProfile, n: int) -> np.ndarray:
        """Sample synthetic data from histogram bins."""
        total = profile.bin_counts.sum()
        if total == 0:
            return np.random.uniform(profile.min, profile.max, n)

        probs = profile.bin_counts / total
        bins = np.random.choice(len(profile.bin_counts), size=n, p=probs)
        return np.array([np.random.uniform(profile.bin_edges[i], profile.bin_edges[i+1]) for i in bins])

    def _get_severity(self, psi: float, js: float) -> DriftSeverity:
        """Determine severity from test results."""
        psi_t = self.config["thresholds"]["psi"]
        js_t = self.config["thresholds"]["js_divergence"]

        # PSI-based (primary)
        if psi >= psi_t["severe"] or js >= js_t["severe"]:
            return DriftSeverity.SEVERE
        if psi >= psi_t["significant"] or js >= js_t["significant"]:
            return DriftSeverity.SIGNIFICANT
        if psi >= psi_t["moderate"] or js >= js_t["moderate"]:
            return DriftSeverity.MODERATE
        if psi >= psi_t["low"] or js >= js_t["low"]:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def _check_drifted(self, psi: float, js: float, ks_pval: float) -> bool:
        """Check if feature exceeds drift threshold."""
        method = self.config["detection"]["method"].lower()
        if method == "psi":
            return psi >= self.config["thresholds"]["psi"]["moderate"]
        elif method == "ks":
            return ks_pval < self.config["thresholds"]["ks_p_value"]
        elif method in ("js", "jensen_shannon"):
            return js >= self.config["thresholds"]["js_divergence"]["moderate"]
        return psi >= self.config["thresholds"]["psi"]["moderate"]

    def _get_recommendations(
        self, severity: DriftSeverity, detected: bool, n_drifted: int, n_total: int,
        results: Dict[str, FeatureDriftResult]
    ) -> List[str]:
        """Generate recommendations."""
        if severity == DriftSeverity.NONE:
            return ["No action needed: all features stable"]

        recs = []
        if severity == DriftSeverity.SEVERE:
            recs.append("URGENT: Retrain model immediately - severe drift detected")
        elif detected:
            recs.append(f"Retrain model: {n_drifted}/{n_total} features show {severity.value} drift")

        # List drifted features by severity
        severe = [r.feature_name for r in results.values() if r.severity == DriftSeverity.SEVERE]
        if severe:
            recs.append(f"Investigate: {', '.join(severe)}")

        if not detected and severity == DriftSeverity.LOW:
            recs.append("Continue monitoring: minor drift within acceptable range")

        return recs or ["Monitor closely"]
