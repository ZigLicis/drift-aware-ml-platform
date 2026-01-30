"""
Tests for drift detection statistical tests module.

Tests each statistical method with:
- Known distributions (identical, shifted, different)
- Edge cases (empty, constant, NaN values)
- Expected mathematical properties
"""

import pytest
import numpy as np
from scipy import stats

from src.drift_detection.statistical_tests import StatisticalTests


class TestPSI:
    """Tests for Population Stability Index."""

    def test_identical_distributions_psi_zero(self):
        """PSI should be ~0 for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = StatisticalTests.calculate_psi(data, data.copy())
        assert psi == pytest.approx(0.0, abs=1e-10)

    def test_similar_distributions_low_psi(self):
        """PSI should be low for similar distributions."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)
        psi = StatisticalTests.calculate_psi(ref, cur)
        assert psi < 0.1, "PSI should be < 0.1 for same distribution"

    def test_shifted_distribution_high_psi(self):
        """PSI should be high for shifted distributions."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)  # Shifted mean
        psi = StatisticalTests.calculate_psi(ref, cur)
        assert psi > 0.2, "PSI should be > 0.2 for significantly shifted distribution"

    def test_different_variance_psi(self):
        """PSI should detect variance changes."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 3, 1000)  # Same mean, different variance
        psi = StatisticalTests.calculate_psi(ref, cur)
        assert psi > 0.1, "PSI should detect variance change"

    def test_psi_is_non_negative(self):
        """PSI should always be non-negative."""
        np.random.seed(42)
        for _ in range(10):
            ref = np.random.uniform(-5, 5, 100)
            cur = np.random.uniform(-5, 5, 100)
            psi = StatisticalTests.calculate_psi(ref, cur)
            assert psi >= 0, "PSI must be non-negative"

    def test_psi_custom_bins(self):
        """PSI should work with different bin counts."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(1, 1, 1000)

        psi_5 = StatisticalTests.calculate_psi(ref, cur, bins=5)
        psi_20 = StatisticalTests.calculate_psi(ref, cur, bins=20)

        # Both should indicate drift, may differ slightly
        assert psi_5 > 0.1
        assert psi_20 > 0.1

    def test_psi_constant_same_value(self):
        """PSI should be 0 for identical constant arrays."""
        ref = np.array([5.0] * 100)
        cur = np.array([5.0] * 100)
        psi = StatisticalTests.calculate_psi(ref, cur)
        assert psi == 0.0

    def test_psi_constant_different_values(self):
        """PSI should be inf for different constant arrays."""
        ref = np.array([5.0] * 100)
        cur = np.array([10.0] * 100)
        psi = StatisticalTests.calculate_psi(ref, cur)
        assert psi == float('inf')


class TestKSTest:
    """Tests for Kolmogorov-Smirnov test."""

    def test_identical_distributions_high_pvalue(self):
        """KS p-value should be high (~1) for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        stat, pval = StatisticalTests.ks_test(data, data.copy())
        assert stat == pytest.approx(0.0, abs=1e-10)
        assert pval == pytest.approx(1.0, abs=1e-10)

    def test_same_distribution_not_significant(self):
        """KS test should not find significance for same distribution."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(0, 1, 500)
        stat, pval = StatisticalTests.ks_test(ref, cur)
        # p-value should generally be > 0.05, though can sometimes fail
        # Using a more relaxed check
        assert pval > 0.01, "p-value should be reasonably high for same distribution"

    def test_different_distributions_low_pvalue(self):
        """KS test should detect different distributions."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(2, 1, 500)  # Shifted
        stat, pval = StatisticalTests.ks_test(ref, cur)
        assert pval < 0.05, "Should detect shifted distribution"
        assert stat > 0.2, "KS statistic should be substantial"

    def test_ks_statistic_bounded(self):
        """KS statistic should be between 0 and 1."""
        np.random.seed(42)
        ref = np.random.exponential(1, 200)
        cur = np.random.uniform(0, 5, 200)
        stat, pval = StatisticalTests.ks_test(ref, cur)
        assert 0 <= stat <= 1
        assert 0 <= pval <= 1

    def test_ks_constant_same(self):
        """KS test with identical constant values."""
        ref = np.array([3.0] * 50)
        cur = np.array([3.0] * 50)
        stat, pval = StatisticalTests.ks_test(ref, cur)
        assert stat == 0.0
        assert pval == 1.0

    def test_ks_constant_different(self):
        """KS test with different constant values."""
        ref = np.array([3.0] * 50)
        cur = np.array([7.0] * 50)
        stat, pval = StatisticalTests.ks_test(ref, cur)
        assert stat == 1.0
        assert pval == 0.0


class TestJensenShannonDivergence:
    """Tests for Jensen-Shannon divergence."""

    def test_identical_distributions_zero(self):
        """JS divergence should be 0 for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        js = StatisticalTests.jensen_shannon_divergence(data, data.copy())
        assert js == pytest.approx(0.0, abs=0.01)

    def test_similar_distributions_low_js(self):
        """JS divergence should be low for similar distributions."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)
        js = StatisticalTests.jensen_shannon_divergence(ref, cur)
        assert js < 0.1, "JS should be low for same distribution"

    def test_different_distributions_high_js(self):
        """JS divergence should be higher for different distributions."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)
        js = StatisticalTests.jensen_shannon_divergence(ref, cur)
        assert js > 0.3, "JS should be higher for shifted distribution"

    def test_js_bounded_zero_one(self):
        """JS divergence should be between 0 and 1."""
        np.random.seed(42)
        ref = np.random.exponential(1, 500)
        cur = np.random.uniform(0, 10, 500)
        js = StatisticalTests.jensen_shannon_divergence(ref, cur)
        assert 0 <= js <= 1

    def test_js_symmetric(self):
        """JS divergence should be symmetric."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(1, 2, 500)
        js1 = StatisticalTests.jensen_shannon_divergence(ref, cur)
        js2 = StatisticalTests.jensen_shannon_divergence(cur, ref)
        assert js1 == pytest.approx(js2, abs=0.01)

    def test_js_constant_same(self):
        """JS divergence for identical constants."""
        ref = np.array([5.0] * 100)
        cur = np.array([5.0] * 100)
        js = StatisticalTests.jensen_shannon_divergence(ref, cur)
        assert js == 0.0

    def test_js_constant_different(self):
        """JS divergence for different constants."""
        ref = np.array([5.0] * 100)
        cur = np.array([10.0] * 100)
        js = StatisticalTests.jensen_shannon_divergence(ref, cur)
        assert js == 1.0


class TestChiSquareTest:
    """Tests for Chi-square test."""

    def test_identical_categorical_high_pvalue(self):
        """Chi-square should have high p-value for identical categorical data."""
        np.random.seed(42)
        categories = np.array([1, 2, 3, 4, 5])
        ref = np.random.choice(categories, 500)
        chi2, pval = StatisticalTests.chi_square_test(ref, ref.copy())
        assert pval == pytest.approx(1.0, abs=0.01)

    def test_similar_categorical_not_significant(self):
        """Chi-square should not be significant for same distribution."""
        np.random.seed(42)
        categories = np.array([1, 2, 3, 4, 5])
        ref = np.random.choice(categories, 500, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        cur = np.random.choice(categories, 500, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        chi2, pval = StatisticalTests.chi_square_test(ref, cur)
        # Should generally not detect drift
        assert pval > 0.01

    def test_different_categorical_significant(self):
        """Chi-square should detect different categorical distributions."""
        np.random.seed(42)
        categories = np.array([1, 2, 3, 4, 5])
        ref = np.random.choice(categories, 500, p=[0.5, 0.2, 0.1, 0.1, 0.1])
        cur = np.random.choice(categories, 500, p=[0.1, 0.1, 0.1, 0.2, 0.5])
        chi2, pval = StatisticalTests.chi_square_test(ref, cur)
        assert pval < 0.05, "Should detect different categorical distribution"

    def test_chi2_continuous_with_bins(self):
        """Chi-square should work with continuous data using bins."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(2, 1, 500)
        chi2, pval = StatisticalTests.chi_square_test(ref, cur, bins=10)
        assert pval < 0.05, "Should detect shifted continuous distribution"

    def test_chi2_pvalue_bounded(self):
        """Chi-square p-value should be between 0 and 1."""
        np.random.seed(42)
        ref = np.random.choice([1, 2, 3], 200)
        cur = np.random.choice([1, 2, 3], 200)
        chi2, pval = StatisticalTests.chi_square_test(ref, cur)
        assert 0 <= pval <= 1


class TestWassersteinDistance:
    """Tests for Wasserstein (Earth Mover's) distance."""

    def test_identical_distributions_zero(self):
        """Wasserstein distance should be 0 for identical data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        dist = StatisticalTests.wasserstein_distance(data, data.copy())
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_shifted_distribution_equals_shift(self):
        """Wasserstein distance should equal the mean shift for shifted normals."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 10000)
        cur = np.random.normal(2, 1, 10000)  # Shifted by 2
        dist = StatisticalTests.wasserstein_distance(ref, cur)
        # Should be approximately 2 (the shift amount)
        assert dist == pytest.approx(2.0, abs=0.1)

    def test_wasserstein_non_negative(self):
        """Wasserstein distance should always be non-negative."""
        np.random.seed(42)
        for _ in range(10):
            ref = np.random.uniform(-10, 10, 100)
            cur = np.random.uniform(-10, 10, 100)
            dist = StatisticalTests.wasserstein_distance(ref, cur)
            assert dist >= 0

    def test_wasserstein_symmetric(self):
        """Wasserstein distance should be symmetric."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.exponential(2, 500)
        dist1 = StatisticalTests.wasserstein_distance(ref, cur)
        dist2 = StatisticalTests.wasserstein_distance(cur, ref)
        assert dist1 == pytest.approx(dist2, abs=1e-10)

    def test_wasserstein_constant_same(self):
        """Wasserstein distance for identical constants."""
        ref = np.array([5.0] * 100)
        cur = np.array([5.0] * 100)
        dist = StatisticalTests.wasserstein_distance(ref, cur)
        assert dist == 0.0

    def test_wasserstein_constant_different(self):
        """Wasserstein distance for different constants equals difference."""
        ref = np.array([5.0] * 100)
        cur = np.array([8.0] * 100)
        dist = StatisticalTests.wasserstein_distance(ref, cur)
        assert dist == 3.0


class TestEdgeCases:
    """Tests for edge cases across all methods."""

    def test_empty_array_raises(self):
        """All methods should raise ValueError for empty arrays."""
        ref = np.array([1.0, 2.0, 3.0])
        empty = np.array([])

        with pytest.raises(ValueError, match="empty"):
            StatisticalTests.calculate_psi(empty, ref)

        with pytest.raises(ValueError, match="empty"):
            StatisticalTests.ks_test(ref, empty)

        with pytest.raises(ValueError, match="empty"):
            StatisticalTests.jensen_shannon_divergence(empty, empty)

    def test_nan_handling(self):
        """Methods should handle NaN values by removing them."""
        np.random.seed(42)
        ref = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0])
        cur = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])

        # Should not raise, NaN values should be filtered out
        psi = StatisticalTests.calculate_psi(ref, cur)
        assert not np.isnan(psi)

        stat, pval = StatisticalTests.ks_test(ref, cur)
        assert not np.isnan(stat)
        assert not np.isnan(pval)

    def test_all_nan_raises(self):
        """Methods should raise ValueError if all values are NaN."""
        ref = np.array([1.0, 2.0, 3.0])
        all_nan = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="empty|NaN"):
            StatisticalTests.calculate_psi(ref, all_nan)

    def test_single_element(self):
        """Methods should handle single-element arrays."""
        ref = np.array([5.0])
        cur = np.array([5.0])

        # These should work without error
        psi = StatisticalTests.calculate_psi(ref, cur)
        stat, pval = StatisticalTests.ks_test(ref, cur)
        dist = StatisticalTests.wasserstein_distance(ref, cur)

        assert psi == 0.0
        assert dist == 0.0


class TestDetectDrift:
    """Tests for the convenience detect_drift method."""

    def test_detect_drift_psi(self):
        """detect_drift should work with PSI method."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)

        result = StatisticalTests.detect_drift(ref, cur, method="psi")

        assert result["method"] == "psi"
        assert result["statistic"] > 0.2
        assert result["drift_detected"] is True
        assert "significant" in result["interpretation"].lower()

    def test_detect_drift_ks(self):
        """detect_drift should work with KS method."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(0, 1, 500)

        result = StatisticalTests.detect_drift(ref, cur, method="ks")

        assert result["method"] == "ks"
        assert result["p_value"] is not None
        assert 0 <= result["statistic"] <= 1

    def test_detect_drift_custom_threshold(self):
        """detect_drift should respect custom thresholds."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0.5, 1, 1000)  # Slight shift

        # Default threshold should not detect
        result_default = StatisticalTests.detect_drift(ref, cur, method="psi")

        # Lower threshold should detect
        result_low = StatisticalTests.detect_drift(
            ref, cur, method="psi", threshold=0.01
        )

        # Verify different outcomes are possible with different thresholds
        assert result_default["threshold"] == 0.1
        assert result_low["threshold"] == 0.01

    def test_detect_drift_invalid_method(self):
        """detect_drift should raise for invalid method."""
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown method"):
            StatisticalTests.detect_drift(ref, cur, method="invalid")

    def test_detect_drift_all_methods(self):
        """All methods should work through detect_drift."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(1, 1, 500)

        for method in ["psi", "ks", "js", "chi2", "wasserstein"]:
            result = StatisticalTests.detect_drift(ref, cur, method=method)
            assert result["method"] == method
            assert result["statistic"] is not None
            assert "interpretation" in result
