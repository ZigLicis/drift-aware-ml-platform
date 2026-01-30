"""
Drift Detection Module for Domain-Shift ML Platform.

This module provides statistical tests and utilities for detecting
domain shift (data drift) between reference and current data distributions.

Key Components:
- StatisticalTests: Collection of statistical tests (PSI, KS, JS divergence, etc.)
- ReferenceManager: Manages reference data profiles for drift comparison
- ReferenceProfile: Data class representing a feature's reference distribution
"""

from src.drift_detection.statistical_tests import StatisticalTests
from src.drift_detection.reference_manager import (
    ReferenceManager,
    ReferenceProfile,
    ReferenceManagerError,
    ReferenceNotFoundError,
    ReferenceCorruptedError,
)

__all__ = [
    "StatisticalTests",
    "ReferenceManager",
    "ReferenceProfile",
    "ReferenceManagerError",
    "ReferenceNotFoundError",
    "ReferenceCorruptedError",
]
