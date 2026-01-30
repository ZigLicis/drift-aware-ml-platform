#!/usr/bin/env python3
"""
Day 1 Integration Test: Drift Detection Foundation

Simulates the summer→winter domain shift and validates 
that the statistical tests can detect it.

Usage:
    python scripts/test_drift_day1.py
    python scripts/test_drift_day1.py --keep-files  # Keep test reference files
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drift_detection import StatisticalTests, ReferenceManager


# Test configuration
TEST_REFERENCE_NAME = "test_summer_baseline"
FEATURE_COLUMNS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pressure_msl"]

# Thresholds for interpretation
PSI_THRESHOLD_MODERATE = 0.1
PSI_THRESHOLD_SIGNIFICANT = 0.2
KS_PVALUE_THRESHOLD = 0.05
JS_THRESHOLD_MODERATE = 0.1
JS_THRESHOLD_HIGH = 0.3


def create_summer_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic summer-like reference data."""
    np.random.seed(42)
    return pd.DataFrame({
        "temperature_2m": np.random.normal(22, 8, n_samples),       # Mean 22°C, warm
        "relative_humidity_2m": np.random.normal(55, 15, n_samples),
        "wind_speed_10m": np.random.normal(12, 5, n_samples),
        "pressure_msl": np.random.normal(1013, 10, n_samples),
    })


def create_winter_data(n_samples: int = 500) -> pd.DataFrame:
    """Create synthetic winter-like drifted data."""
    np.random.seed(123)
    return pd.DataFrame({
        "temperature_2m": np.random.normal(3, 6, n_samples),        # Mean 3°C, cold!
        "relative_humidity_2m": np.random.normal(70, 12, n_samples), # Higher humidity
        "wind_speed_10m": np.random.normal(15, 7, n_samples),        # Windier
        "pressure_msl": np.random.normal(1010, 12, n_samples),       # Similar
    })


def get_psi_indicator(psi: float) -> str:
    """Get colored indicator for PSI value."""
    if psi >= PSI_THRESHOLD_SIGNIFICANT:
        return f"[🔴 SIGNIFICANT DRIFT - threshold {PSI_THRESHOLD_SIGNIFICANT}]"
    elif psi >= PSI_THRESHOLD_MODERATE:
        return f"[🟡 MODERATE DRIFT - threshold {PSI_THRESHOLD_SIGNIFICANT}]"
    else:
        return "[🟢 NO SIGNIFICANT DRIFT]"


def get_ks_indicator(p_value: float) -> str:
    """Get colored indicator for KS test p-value."""
    if p_value < 0.001:
        return "[🔴 DRIFT DETECTED]"
    elif p_value < KS_PVALUE_THRESHOLD:
        return "[🟡 POSSIBLE DRIFT]"
    else:
        return "[🟢 NO DRIFT]"


def get_js_indicator(js: float) -> str:
    """Get colored indicator for JS divergence."""
    if js >= JS_THRESHOLD_HIGH:
        return "[🔴 HIGH]"
    elif js >= JS_THRESHOLD_MODERATE:
        return "[🟡 MODERATE]"
    else:
        return "[🟢 LOW]"


def get_wasserstein_indicator(distance: float, feature: str) -> str:
    """Get colored indicator for Wasserstein distance."""
    # Thresholds depend on feature scale
    if feature == "temperature_2m":
        if distance > 10:
            return "[🔴 LARGE SHIFT]"
        elif distance > 3:
            return "[🟡 MODERATE SHIFT]"
    elif feature == "relative_humidity_2m":
        if distance > 10:
            return "[🔴 LARGE SHIFT]"
        elif distance > 5:
            return "[🟡 MODERATE SHIFT]"
    elif feature == "wind_speed_10m":
        if distance > 3:
            return "[🔴 LARGE SHIFT]"
        elif distance > 1:
            return "[🟡 MODERATE SHIFT]"
    elif feature == "pressure_msl":
        if distance > 5:
            return "[🔴 LARGE SHIFT]"
        elif distance > 2:
            return "[🟡 MODERATE SHIFT]"
    return "[🟢 SMALL]"


def print_header():
    """Print test header."""
    print()
    print("=" * 70)
    print("DAY 1 DRIFT DETECTION TEST")
    print("=" * 70)


def print_data_summary(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Print data summary."""
    print(f"Reference Data: Summer-like (n={len(reference_data)})")
    print(f"Current Data:   Winter-like (n={len(current_data)})")


def print_feature_results(
    feature: str,
    ref_data: np.ndarray,
    cur_data: np.ndarray,
):
    """Print drift metrics for a single feature."""
    print("-" * 70)
    print(f"FEATURE: {feature}")
    print("-" * 70)

    # Basic stats
    ref_mean = np.mean(ref_data)
    ref_std = np.std(ref_data)
    cur_mean = np.mean(cur_data)
    cur_std = np.std(cur_data)

    # Units for display
    unit = ""
    if feature == "temperature_2m":
        unit = "°C"
    elif feature == "relative_humidity_2m":
        unit = "%"
    elif feature == "wind_speed_10m":
        unit = " km/h"
    elif feature == "pressure_msl":
        unit = " hPa"

    print(f"  Reference: mean={ref_mean:.1f}{unit}, std={ref_std:.1f}")
    print(f"  Current:   mean={cur_mean:.1f}{unit}, std={cur_std:.1f}")
    print()
    print("  Drift Metrics:")

    # Calculate all metrics
    psi = StatisticalTests.calculate_psi(ref_data, cur_data)
    ks_stat, ks_pval = StatisticalTests.ks_test(ref_data, cur_data)
    js = StatisticalTests.jensen_shannon_divergence(ref_data, cur_data)
    wasserstein = StatisticalTests.wasserstein_distance(ref_data, cur_data)

    # Print with indicators
    print(f"    PSI:              {psi:.3f}   {get_psi_indicator(psi)}")
    print(f"    KS Statistic:     {ks_stat:.3f}   (p-value: {ks_pval:.4f})  {get_ks_indicator(ks_pval)}")
    print(f"    JS Divergence:    {js:.3f}   {get_js_indicator(js)}")
    print(f"    Wasserstein:      {wasserstein:.1f}    {get_wasserstein_indicator(wasserstein, feature)}")

    return {
        "feature": feature,
        "psi": psi,
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "js": js,
        "wasserstein": wasserstein,
    }


def print_summary(results: list):
    """Print summary of all features."""
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    significant = [(r["feature"], r["psi"]) for r in results if r["psi"] >= PSI_THRESHOLD_SIGNIFICANT]
    moderate = [(r["feature"], r["psi"]) for r in results if PSI_THRESHOLD_MODERATE <= r["psi"] < PSI_THRESHOLD_SIGNIFICANT]
    no_drift = [(r["feature"], r["psi"]) for r in results if r["psi"] < PSI_THRESHOLD_MODERATE]

    if significant:
        print(f"Features with significant drift (PSI > {PSI_THRESHOLD_SIGNIFICANT}):")
        for feature, psi in significant:
            print(f"  🔴 {feature:25s} PSI = {psi:.3f}")

    if moderate:
        print(f"Features with moderate drift ({PSI_THRESHOLD_MODERATE} < PSI < {PSI_THRESHOLD_SIGNIFICANT}):")
        for feature, psi in moderate:
            print(f"  🟡 {feature:25s} PSI = {psi:.3f}")

    if no_drift:
        print(f"Features with no significant drift (PSI < {PSI_THRESHOLD_MODERATE}):")
        for feature, psi in no_drift:
            print(f"  🟢 {feature:25s} PSI = {psi:.3f}")


def validate_no_false_positives(reference_data: pd.DataFrame) -> bool:
    """Validate that identical distributions don't trigger false positives."""
    print()
    print("-" * 70)
    print("VALIDATION: Testing with identical distributions (no drift)")
    print("-" * 70)

    all_passed = True

    # Test comparing reference data to itself
    for feature in ["temperature_2m"]:
        ref_data = reference_data[feature].values

        psi = StatisticalTests.calculate_psi(ref_data, ref_data.copy())
        _, ks_pval = StatisticalTests.ks_test(ref_data, ref_data.copy())

        psi_ok = psi < 0.01
        ks_ok = ks_pval > 0.99

        psi_status = "🟢 CORRECT - No false positive" if psi_ok else "🔴 FALSE POSITIVE"
        ks_status = "🟢 CORRECT - No false positive" if ks_ok else "🔴 FALSE POSITIVE"

        print(f"  Same data vs itself:")
        print(f"    {feature} PSI:      {psi:.3f}   [{psi_status}]")
        print(f"    {feature} KS p-val: {ks_pval:.3f}   [{ks_status}]")

        if not psi_ok or not ks_ok:
            all_passed = False

    return all_passed


def test_reference_manager(reference_data: pd.DataFrame) -> bool:
    """Test ReferenceManager save/load functionality."""
    print()
    print("-" * 70)
    print("VALIDATION: ReferenceManager save/load")
    print("-" * 70)

    try:
        manager = ReferenceManager(storage_path="data/references")

        # Create profiles
        profiles = manager.create_reference_from_dataframe(
            df=reference_data,
            feature_columns=FEATURE_COLUMNS,
            reference_name=TEST_REFERENCE_NAME,
        )
        print(f"  Created {len(profiles)} reference profiles")

        # Save profiles
        save_path = manager.save_reference(
            profiles,
            TEST_REFERENCE_NAME,
            metadata={
                "description": "Day 1 test - summer baseline",
                "data_type": "synthetic",
            }
        )
        print(f"  Saved to: {save_path}")

        # Load profiles back
        loaded_profiles = manager.load_reference(TEST_REFERENCE_NAME)
        print(f"  Loaded {len(loaded_profiles)} profiles")

        # Verify data integrity
        for feature in FEATURE_COLUMNS:
            orig = profiles[feature]
            loaded = loaded_profiles[feature]

            if not np.isclose(orig.mean, loaded.mean):
                print(f"  🔴 MISMATCH: {feature} mean differs after load")
                return False

            if not np.allclose(orig.bin_edges, loaded.bin_edges):
                print(f"  🔴 MISMATCH: {feature} bin_edges differ after load")
                return False

        print("  🟢 All profiles verified - save/load working correctly")
        return True

    except Exception as e:
        print(f"  🔴 ERROR: {e}")
        return False


def cleanup_test_files(keep_files: bool):
    """Clean up test reference files."""
    if keep_files:
        print()
        print(f"  Keeping test files at: data/references/{TEST_REFERENCE_NAME}/")
        return

    try:
        manager = ReferenceManager(storage_path="data/references")
        if manager.reference_exists(TEST_REFERENCE_NAME):
            manager.delete_reference(TEST_REFERENCE_NAME)
            print()
            print(f"  Cleaned up test files: data/references/{TEST_REFERENCE_NAME}/")
    except Exception as e:
        print(f"  Warning: Could not clean up test files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Day 1 Integration Test for Drift Detection"
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep test reference files after completion",
    )
    args = parser.parse_args()

    # Track overall success
    all_validations_passed = True

    # Create test data
    print_header()
    reference_data = create_summer_data()
    current_data = create_winter_data()
    print_data_summary(reference_data, current_data)

    # Run drift detection for each feature
    results = []
    for feature in FEATURE_COLUMNS:
        ref_values = reference_data[feature].values
        cur_values = current_data[feature].values
        result = print_feature_results(feature, ref_values, cur_values)
        results.append(result)

    # Print summary
    print_summary(results)

    # Validate no false positives
    no_false_positives = validate_no_false_positives(reference_data)
    if not no_false_positives:
        all_validations_passed = False

    # Test ReferenceManager
    ref_manager_ok = test_reference_manager(reference_data)
    if not ref_manager_ok:
        all_validations_passed = False

    # Validate that we detected the expected drift
    # Temperature should have significant drift (PSI > 0.2)
    temp_result = next(r for r in results if r["feature"] == "temperature_2m")
    if temp_result["psi"] < PSI_THRESHOLD_SIGNIFICANT:
        print()
        print("  🔴 VALIDATION FAILED: Temperature drift not detected!")
        all_validations_passed = False

    # Clean up
    cleanup_test_files(args.keep_files)

    # Final status
    print()
    print("=" * 70)
    if all_validations_passed:
        print("DAY 1 STATUS: ✅ ALL COMPONENTS WORKING")
    else:
        print("DAY 1 STATUS: ❌ SOME VALIDATIONS FAILED")
    print("=" * 70)
    print()

    return 0 if all_validations_passed else 1


if __name__ == "__main__":
    sys.exit(main())
