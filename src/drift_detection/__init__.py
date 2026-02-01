"""
Drift Detection Module for Domain-Shift ML Platform.

This module provides statistical tests and utilities for detecting
domain shift (data drift) between reference and current data distributions.

Key Components:
- DriftDetector: Main orchestrator for drift detection workflows
- StatisticalTests: Collection of statistical tests (PSI, KS, JS divergence, etc.)
- ReferenceManager: Manages reference data profiles for drift comparison
- ReferenceProfile: Data class representing a feature's reference distribution
- DriftReport: Complete drift analysis report with recommendations
- DriftSeverity: Enumeration of drift severity levels
"""

from src.drift_detection.statistical_tests import StatisticalTests
from src.drift_detection.reference_manager import (
    ReferenceManager,
    ReferenceProfile,
    ReferenceManagerError,
    ReferenceNotFoundError,
    ReferenceCorruptedError,
)
from src.drift_detection.detector import (
    DriftDetector,
    DriftReport,
    DriftSeverity,
    FeatureDriftResult,
    DriftDetectorError,
    InsufficientDataError,
)

__all__ = [
    # Main detector
    "DriftDetector",
    "DriftReport",
    "DriftSeverity",
    "FeatureDriftResult",
    "DriftDetectorError",
    "InsufficientDataError",
    # Statistical tests
    "StatisticalTests",
    # Reference management
    "ReferenceManager",
    "ReferenceProfile",
    "ReferenceManagerError",
    "ReferenceNotFoundError",
    "ReferenceCorruptedError",
]
