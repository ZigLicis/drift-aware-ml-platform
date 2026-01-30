"""
Statistical tests for drift detection.

This module provides a collection of statistical tests to detect distribution
shifts between reference (training) and current (production) data.

Interpretation Guidelines:
- PSI: < 0.1 no shift, 0.1-0.2 moderate, > 0.2 significant
- KS test: p-value < 0.05 indicates significant drift
- JS divergence: 0 = identical, 1 = completely different
- Chi-square: p-value < 0.05 indicates significant drift
- Wasserstein: Higher values indicate greater distribution difference
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


class StatisticalTests:
    """Collection of statistical tests for drift detection."""

    @staticmethod
    def _validate_inputs(
        reference: np.ndarray,
        current: np.ndarray,
        allow_empty: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and clean input arrays.

        Args:
            reference: Reference distribution array
            current: Current distribution array
            allow_empty: Whether to allow empty arrays

        Returns:
            Tuple of cleaned (reference, current) arrays with NaN removed

        Raises:
            ValueError: If arrays are empty (when allow_empty=False) or all NaN
        """
        # Convert to numpy arrays if needed
        reference = np.asarray(reference, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)

        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if not allow_empty:
            if len(reference) == 0:
                raise ValueError("Reference array is empty or contains only NaN values")
            if len(current) == 0:
                raise ValueError("Current array is empty or contains only NaN values")

        return reference, current

    @staticmethod
    def _create_histogram(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create aligned histograms for two distributions.

        Uses the combined range of both distributions to ensure
        consistent binning for comparison.

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins

        Returns:
            Tuple of (reference_proportions, current_proportions)
        """
        # Determine bin edges from combined data
        combined = np.concatenate([reference, current])
        min_val, max_val = combined.min(), combined.max()

        # Handle constant values
        if min_val == max_val:
            return np.array([1.0]), np.array([1.0])

        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Calculate histograms
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions, avoiding division by zero
        ref_props = ref_counts / len(reference) if len(reference) > 0 else ref_counts.astype(float)
        cur_props = cur_counts / len(current) if len(current) > 0 else cur_counts.astype(float)

        return ref_props, cur_props

    @staticmethod
    def calculate_psi(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Population Stability Index (PSI).

        PSI measures the shift in distribution between two samples.
        It's commonly used in credit risk modeling and ML monitoring.

        Interpretation:
            PSI < 0.1: No significant shift - distributions are stable
            PSI 0.1-0.2: Moderate shift - some change detected, monitor closely
            PSI > 0.2: Significant shift - action required, consider retraining

        Formula:
            PSI = Σ (P_current - P_reference) * ln(P_current / P_reference)

        Args:
            reference: Reference distribution (training data)
            current: Current distribution (new data)
            bins: Number of bins for discretization (default: 10)

        Returns:
            PSI score (0 = identical distributions, higher = more drift)

        Raises:
            ValueError: If inputs are empty or all NaN

        Example:
            >>> ref = np.random.normal(0, 1, 1000)
            >>> cur = np.random.normal(0, 1, 1000)  # Same distribution
            >>> StatisticalTests.calculate_psi(ref, cur)  # ~0.0
            >>> cur_shifted = np.random.normal(2, 1, 1000)  # Shifted
            >>> StatisticalTests.calculate_psi(ref, cur_shifted)  # > 0.2
        """
        reference, current = StatisticalTests._validate_inputs(reference, current)

        # Check for constant values in both
        if np.std(reference) == 0 and np.std(current) == 0:
            # Both constant - check if same value
            if reference[0] == current[0]:
                return 0.0
            else:
                return float('inf')

        ref_props, cur_props = StatisticalTests._create_histogram(reference, current, bins)

        # Apply small epsilon to avoid division by zero and log(0)
        epsilon = 1e-10
        ref_props = np.clip(ref_props, epsilon, 1.0)
        cur_props = np.clip(cur_props, epsilon, 1.0)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    @staticmethod
    def ks_test(
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution difference.

        The KS test compares the cumulative distribution functions (CDFs)
        of two samples. The KS statistic is the maximum distance between
        the two CDFs.

        Interpretation:
            p-value >= 0.05: No significant evidence of drift
            p-value < 0.05: Significant drift detected
            KS statistic: Ranges from 0 to 1, higher = more different

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            Tuple of (ks_statistic, p_value)
            - ks_statistic: Maximum distance between CDFs (0-1)
            - p_value: Probability that samples come from same distribution

        Raises:
            ValueError: If inputs are empty or all NaN

        Example:
            >>> ref = np.random.normal(0, 1, 1000)
            >>> cur = np.random.normal(0, 1, 1000)
            >>> stat, pval = StatisticalTests.ks_test(ref, cur)
            >>> pval > 0.05  # True - same distribution
        """
        reference, current = StatisticalTests._validate_inputs(reference, current)

        # Handle constant values
        ref_std = np.std(reference)
        cur_std = np.std(current)

        if ref_std == 0 and cur_std == 0:
            # Both constant
            if reference[0] == current[0]:
                return 0.0, 1.0  # Identical
            else:
                return 1.0, 0.0  # Completely different

        statistic, p_value = stats.ks_2samp(reference, current)

        return float(statistic), float(p_value)

    @staticmethod
    def jensen_shannon_divergence(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Jensen-Shannon divergence (symmetric KL divergence).

        JS divergence is a symmetric and bounded version of KL divergence.
        It measures the similarity between two probability distributions.

        Interpretation:
            0.0: Identical distributions
            0.0-0.1: Very similar
            0.1-0.3: Moderate difference
            0.3-0.5: Significant difference
            0.5-1.0: Very different distributions

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization (default: 10)

        Returns:
            JS divergence value between 0 (identical) and 1 (completely different)

        Raises:
            ValueError: If inputs are empty or all NaN

        Note:
            This returns the JS divergence (squared JS distance).
            The JS distance would be sqrt(JS divergence).
        """
        reference, current = StatisticalTests._validate_inputs(reference, current)

        # Handle constant values
        if np.std(reference) == 0 and np.std(current) == 0:
            if reference[0] == current[0]:
                return 0.0
            else:
                return 1.0

        ref_props, cur_props = StatisticalTests._create_histogram(reference, current, bins)

        # Ensure valid probability distributions (sum to 1, no zeros)
        epsilon = 1e-10
        ref_props = ref_props + epsilon
        cur_props = cur_props + epsilon
        ref_props = ref_props / ref_props.sum()
        cur_props = cur_props / cur_props.sum()

        # Calculate JS divergence using scipy (returns JS distance, we square it)
        js_distance = jensenshannon(ref_props, cur_props, base=2)

        # jensenshannon returns the distance (sqrt of divergence)
        # Square it to get divergence, which is bounded [0, 1] with base=2
        js_divergence = js_distance ** 2

        return float(js_divergence)

    @staticmethod
    def chi_square_test(
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Chi-square test for categorical features or binned continuous features.

        For categorical data, pass the raw values. For continuous data,
        the method will bin the values automatically.

        Interpretation:
            p-value >= 0.05: No significant evidence of drift
            p-value < 0.05: Significant drift detected
            Higher chi2 statistic = more different distributions

        Args:
            reference: Reference distribution (categorical values or continuous)
            current: Current distribution
            bins: Number of bins for continuous data. If None, auto-detect
                  based on whether data appears categorical.

        Returns:
            Tuple of (chi2_statistic, p_value)

        Raises:
            ValueError: If inputs are empty or all NaN

        Note:
            For very small expected frequencies (< 5), the chi-square
            approximation may not be reliable. Consider using KS test instead.
        """
        reference, current = StatisticalTests._validate_inputs(reference, current)

        # Determine if data is categorical or continuous
        unique_ref = np.unique(reference)
        unique_cur = np.unique(current)

        # Auto-detect: if few unique values, treat as categorical
        is_categorical = (
            bins is None and
            len(unique_ref) <= 20 and
            len(unique_cur) <= 20
        )

        if is_categorical:
            # Use actual values as categories
            all_categories = np.union1d(unique_ref, unique_cur)

            ref_counts = np.array([np.sum(reference == cat) for cat in all_categories])
            cur_counts = np.array([np.sum(current == cat) for cat in all_categories])
        else:
            # Bin continuous data
            num_bins = bins if bins is not None else 10
            ref_counts, cur_counts = StatisticalTests._get_aligned_counts(
                reference, current, num_bins
            )

        # Handle edge cases
        if np.sum(ref_counts) == 0 or np.sum(cur_counts) == 0:
            return 0.0, 1.0

        # Remove bins with zero counts in both
        mask = (ref_counts > 0) | (cur_counts > 0)
        ref_counts = ref_counts[mask]
        cur_counts = cur_counts[mask]

        if len(ref_counts) < 2:
            # Not enough categories for chi-square
            return 0.0, 1.0

        # Create contingency table and run chi-square test
        contingency = np.array([ref_counts, cur_counts])

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            return float(chi2), float(p_value)
        except ValueError:
            # Edge case: degenerate contingency table
            return 0.0, 1.0

    @staticmethod
    def _get_aligned_counts(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get aligned histogram counts for chi-square test."""
        combined = np.concatenate([reference, current])
        min_val, max_val = combined.min(), combined.max()

        if min_val == max_val:
            return np.array([len(reference)]), np.array([len(current)])

        bin_edges = np.linspace(min_val, max_val, bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        return ref_counts, cur_counts

    @staticmethod
    def wasserstein_distance(
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """
        Earth Mover's Distance (Wasserstein-1 distance) between distributions.

        The Wasserstein distance measures the minimum "work" required to
        transform one distribution into another. It's more interpretable
        for continuous features as it's in the same units as the data.

        Interpretation:
            The distance is in the same units as the input data.
            For normalized data (mean=0, std=1):
                < 0.1: Very similar
                0.1-0.3: Moderate difference
                > 0.3: Significant difference

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            Wasserstein distance (0 = identical, higher = more different)
            Units are the same as the input data.

        Raises:
            ValueError: If inputs are empty or all NaN

        Example:
            >>> ref = np.random.normal(0, 1, 1000)
            >>> cur = np.random.normal(1, 1, 1000)  # Shifted by 1
            >>> StatisticalTests.wasserstein_distance(ref, cur)  # ~1.0
        """
        reference, current = StatisticalTests._validate_inputs(reference, current)

        # Handle constant values
        if np.std(reference) == 0 and np.std(current) == 0:
            return abs(reference[0] - current[0])

        distance = stats.wasserstein_distance(reference, current)

        return float(distance)

    @staticmethod
    def detect_drift(
        reference: np.ndarray,
        current: np.ndarray,
        method: str = "psi",
        threshold: Optional[float] = None,
        **kwargs
    ) -> dict:
        """
        Convenience method to detect drift using a specified method.

        Args:
            reference: Reference distribution
            current: Current distribution
            method: Detection method - "psi", "ks", "js", "chi2", "wasserstein"
            threshold: Custom threshold. If None, uses defaults:
                       - psi: 0.1 (moderate shift)
                       - ks: 0.05 (p-value)
                       - js: 0.1
                       - chi2: 0.05 (p-value)
                       - wasserstein: None (no default threshold)
            **kwargs: Additional arguments passed to the test method

        Returns:
            Dictionary with:
                - method: Test method used
                - statistic: Test statistic value
                - p_value: P-value (if applicable)
                - threshold: Threshold used
                - drift_detected: Boolean indicating if drift was detected
                - interpretation: Human-readable interpretation
        """
        method = method.lower()

        default_thresholds = {
            "psi": 0.1,
            "ks": 0.05,
            "js": 0.1,
            "chi2": 0.05,
            "wasserstein": None
        }

        if threshold is None:
            threshold = default_thresholds.get(method)

        result = {
            "method": method,
            "statistic": None,
            "p_value": None,
            "threshold": threshold,
            "drift_detected": False,
            "interpretation": ""
        }

        if method == "psi":
            psi = StatisticalTests.calculate_psi(reference, current, **kwargs)
            result["statistic"] = psi
            if threshold is not None:
                result["drift_detected"] = psi >= threshold
            if psi < 0.1:
                result["interpretation"] = "No significant shift"
            elif psi < 0.2:
                result["interpretation"] = "Moderate shift - monitor closely"
            else:
                result["interpretation"] = "Significant shift - action required"

        elif method == "ks":
            stat, p_val = StatisticalTests.ks_test(reference, current)
            result["statistic"] = stat
            result["p_value"] = p_val
            if threshold is not None:
                result["drift_detected"] = p_val < threshold
            result["interpretation"] = (
                "Significant drift detected" if p_val < 0.05
                else "No significant drift"
            )

        elif method == "js":
            js = StatisticalTests.jensen_shannon_divergence(reference, current, **kwargs)
            result["statistic"] = js
            if threshold is not None:
                result["drift_detected"] = js >= threshold
            if js < 0.1:
                result["interpretation"] = "Very similar distributions"
            elif js < 0.3:
                result["interpretation"] = "Moderate difference"
            else:
                result["interpretation"] = "Significant difference"

        elif method == "chi2":
            stat, p_val = StatisticalTests.chi_square_test(reference, current, **kwargs)
            result["statistic"] = stat
            result["p_value"] = p_val
            if threshold is not None:
                result["drift_detected"] = p_val < threshold
            result["interpretation"] = (
                "Significant drift detected" if p_val < 0.05
                else "No significant drift"
            )

        elif method == "wasserstein":
            dist = StatisticalTests.wasserstein_distance(reference, current)
            result["statistic"] = dist
            if threshold is not None:
                result["drift_detected"] = dist >= threshold
            result["interpretation"] = f"Distribution distance: {dist:.4f}"

        else:
            raise ValueError(f"Unknown method: {method}. Use: psi, ks, js, chi2, wasserstein")

        return result
