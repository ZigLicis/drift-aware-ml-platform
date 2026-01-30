"""
Drift Detection Module for Domain-Shift ML Platform.

This module provides statistical tests and utilities for detecting
domain shift (data drift) between reference and current data distributions.

Key Components:
- StatisticalTests: Collection of statistical tests (PSI, KS, JS divergence, etc.)
"""

from src.drift_detection.statistical_tests import StatisticalTests

__all__ = ["StatisticalTests"]
