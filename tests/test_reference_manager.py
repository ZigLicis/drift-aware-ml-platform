"""
Tests for the ReferenceManager and ReferenceProfile classes.

Tests cover:
- Creating profiles from DataFrames
- Save/load roundtrip integrity
- Listing and deleting references
- Error handling for missing/corrupted data
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.reference_manager import (
    ReferenceManager,
    ReferenceProfile,
    ReferenceManagerError,
    ReferenceNotFoundError,
    ReferenceCorruptedError,
)


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for reference storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000
    return pd.DataFrame({
        "temperature_2m": np.random.normal(20, 5, n_samples),
        "humidity": np.random.uniform(30, 90, n_samples),
        "pressure": np.random.normal(1013, 10, n_samples),
        "wind_speed": np.random.exponential(5, n_samples),
    })


@pytest.fixture
def manager(temp_storage_path):
    """Create a ReferenceManager with temporary storage."""
    return ReferenceManager(storage_path=temp_storage_path)


class TestReferenceProfile:
    """Tests for ReferenceProfile dataclass."""

    def test_to_dict_serialization(self):
        """Test that ReferenceProfile can be serialized to dict."""
        profile = ReferenceProfile(
            feature_name="temperature",
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            n_samples=1000,
            mean=20.0,
            std=5.0,
            min=5.0,
            max=35.0,
            median=20.0,
            quantiles={"5%": 12.0, "25%": 17.0, "75%": 23.0, "95%": 28.0},
            bin_edges=np.array([5.0, 12.0, 17.0, 20.0, 23.0, 28.0, 35.0]),
            bin_counts=np.array([50, 150, 300, 300, 150, 50]),
        )

        data = profile.to_dict()

        assert data["feature_name"] == "temperature"
        assert data["n_samples"] == 1000
        assert data["mean"] == 20.0
        assert isinstance(data["bin_edges"], list)
        assert isinstance(data["bin_counts"], list)
        assert data["values"] is None

    def test_from_dict_deserialization(self):
        """Test that ReferenceProfile can be deserialized from dict."""
        data = {
            "feature_name": "humidity",
            "created_at": "2024-01-15T12:00:00+00:00",
            "n_samples": 500,
            "mean": 60.0,
            "std": 15.0,
            "min": 30.0,
            "max": 90.0,
            "median": 60.0,
            "quantiles": {"5%": 35.0, "25%": 50.0, "75%": 70.0, "95%": 85.0},
            "bin_edges": [30.0, 45.0, 60.0, 75.0, 90.0],
            "bin_counts": [100, 150, 150, 100],
            "values": None,
        }

        profile = ReferenceProfile.from_dict(data)

        assert profile.feature_name == "humidity"
        assert profile.n_samples == 500
        assert profile.mean == 60.0
        assert isinstance(profile.bin_edges, np.ndarray)
        assert len(profile.bin_edges) == 5
        assert profile.values is None

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = ReferenceProfile(
            feature_name="pressure",
            created_at=datetime.now(timezone.utc),
            n_samples=100,
            mean=1013.0,
            std=10.0,
            min=980.0,
            max=1050.0,
            median=1013.0,
            quantiles={"5%": 995.0, "25%": 1005.0, "75%": 1020.0, "95%": 1030.0},
            bin_edges=np.array([980.0, 1000.0, 1013.0, 1030.0, 1050.0]),
            bin_counts=np.array([10, 30, 40, 20]),
            values=np.array([1010.0, 1015.0, 1020.0]),
        )

        data = original.to_dict()
        restored = ReferenceProfile.from_dict(data)

        assert restored.feature_name == original.feature_name
        assert restored.n_samples == original.n_samples
        assert restored.mean == original.mean
        assert np.allclose(restored.bin_edges, original.bin_edges)
        assert np.allclose(restored.bin_counts, original.bin_counts)
        assert np.allclose(restored.values, original.values)


class TestReferenceManagerInit:
    """Tests for ReferenceManager initialization."""

    def test_creates_storage_directory(self, temp_storage_path):
        """Manager should create storage directory if it doesn't exist."""
        storage_path = Path(temp_storage_path) / "nested" / "references"
        manager = ReferenceManager(storage_path=str(storage_path))

        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_default_storage_path(self):
        """Manager should have a default storage path."""
        manager = ReferenceManager()
        assert manager.storage_path == Path("data/references")


class TestCreateReferenceFromDataFrame:
    """Tests for creating reference profiles from DataFrames."""

    def test_creates_profiles_for_all_columns(self, manager, sample_dataframe):
        """Should create profiles for all specified columns."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m", "humidity", "pressure"],
            reference_name="test_ref",
        )

        assert len(profiles) == 3
        assert "temperature_2m" in profiles
        assert "humidity" in profiles
        assert "pressure" in profiles

    def test_profile_statistics_are_correct(self, manager, sample_dataframe):
        """Verify that profile statistics match the data."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
        )

        profile = profiles["temperature_2m"]
        actual_data = sample_dataframe["temperature_2m"]

        assert profile.n_samples == len(actual_data)
        assert profile.mean == pytest.approx(actual_data.mean(), rel=1e-5)
        assert profile.std == pytest.approx(actual_data.std(), rel=1e-5)
        assert profile.min == pytest.approx(actual_data.min(), rel=1e-5)
        assert profile.max == pytest.approx(actual_data.max(), rel=1e-5)
        assert profile.median == pytest.approx(actual_data.median(), rel=1e-5)

    def test_profile_quantiles_are_correct(self, manager, sample_dataframe):
        """Verify that profile quantiles match the data."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["humidity"],
            reference_name="test_ref",
        )

        profile = profiles["humidity"]
        actual_data = sample_dataframe["humidity"]

        assert profile.quantiles["5%"] == pytest.approx(
            actual_data.quantile(0.05), rel=1e-5
        )
        assert profile.quantiles["25%"] == pytest.approx(
            actual_data.quantile(0.25), rel=1e-5
        )
        assert profile.quantiles["75%"] == pytest.approx(
            actual_data.quantile(0.75), rel=1e-5
        )
        assert profile.quantiles["95%"] == pytest.approx(
            actual_data.quantile(0.95), rel=1e-5
        )

    def test_histogram_bins_created(self, manager, sample_dataframe):
        """Verify that histogram bins are created correctly."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
            n_bins=10,
        )

        profile = profiles["temperature_2m"]

        # Bin edges should be n_bins + 1 or less (due to deduplication)
        assert len(profile.bin_edges) >= 2
        assert len(profile.bin_edges) <= 11
        # Bin counts should sum to n_samples
        assert profile.bin_counts.sum() == profile.n_samples

    def test_store_raw_values_option(self, manager, sample_dataframe):
        """Test that raw values can be optionally stored."""
        # Without raw values
        profiles_no_raw = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
            store_raw_values=False,
        )
        assert profiles_no_raw["temperature_2m"].values is None

        # With raw values
        profiles_with_raw = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
            store_raw_values=True,
        )
        assert profiles_with_raw["temperature_2m"].values is not None
        assert len(profiles_with_raw["temperature_2m"].values) == len(sample_dataframe)

    def test_handles_nan_values(self, manager):
        """Should handle NaN values in data."""
        df = pd.DataFrame({
            "feature": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
        })

        profiles = manager.create_reference_from_dataframe(
            df=df,
            feature_columns=["feature"],
            reference_name="test_ref",
        )

        # Should only count non-NaN values
        assert profiles["feature"].n_samples == 5

    def test_raises_on_empty_dataframe(self, manager):
        """Should raise ValueError for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            manager.create_reference_from_dataframe(
                df=df,
                feature_columns=["feature"],
                reference_name="test_ref",
            )

    def test_raises_on_missing_columns(self, manager, sample_dataframe):
        """Should raise ValueError for missing columns."""
        with pytest.raises(ValueError, match="Missing columns"):
            manager.create_reference_from_dataframe(
                df=sample_dataframe,
                feature_columns=["nonexistent_column"],
                reference_name="test_ref",
            )

    def test_raises_on_all_nan_column(self, manager):
        """Should raise ValueError if column is all NaN."""
        df = pd.DataFrame({
            "feature": [np.nan, np.nan, np.nan],
        })

        with pytest.raises(ValueError, match="only NaN"):
            manager.create_reference_from_dataframe(
                df=df,
                feature_columns=["feature"],
                reference_name="test_ref",
            )


class TestSaveAndLoadReference:
    """Tests for saving and loading references."""

    def test_save_creates_directory_structure(self, manager, sample_dataframe):
        """Saving should create proper directory structure."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m", "humidity"],
            reference_name="test_ref",
        )

        path = manager.save_reference(profiles, "test_ref")

        assert Path(path).exists()
        assert (Path(path) / "metadata.json").exists()
        assert (Path(path) / "temperature_2m.json").exists()
        assert (Path(path) / "humidity.json").exists()

    def test_save_load_roundtrip(self, manager, sample_dataframe):
        """Data should be identical after save/load roundtrip."""
        original_profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m", "humidity"],
            reference_name="test_ref",
        )

        manager.save_reference(original_profiles, "test_ref")
        loaded_profiles = manager.load_reference("test_ref")

        assert len(loaded_profiles) == len(original_profiles)

        for name in original_profiles:
            orig = original_profiles[name]
            loaded = loaded_profiles[name]

            assert loaded.feature_name == orig.feature_name
            assert loaded.n_samples == orig.n_samples
            assert loaded.mean == pytest.approx(orig.mean, rel=1e-10)
            assert loaded.std == pytest.approx(orig.std, rel=1e-10)
            assert np.allclose(loaded.bin_edges, orig.bin_edges)
            assert np.allclose(loaded.bin_counts, orig.bin_counts)

    def test_save_with_metadata(self, manager, sample_dataframe):
        """Metadata should be saved and retrievable."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
        )

        metadata = {
            "model_version": "1.0.0",
            "date_range": "2024-01-01 to 2024-06-01",
            "description": "Baseline training data",
        }

        manager.save_reference(profiles, "test_ref", metadata=metadata)

        stored_metadata = manager.get_reference_metadata("test_ref")

        assert stored_metadata["model_version"] == "1.0.0"
        assert stored_metadata["date_range"] == "2024-01-01 to 2024-06-01"
        assert stored_metadata["reference_name"] == "test_ref"

    def test_save_overwrites_existing(self, manager, sample_dataframe):
        """Saving should overwrite existing reference."""
        profiles1 = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
        )
        manager.save_reference(profiles1, "test_ref", metadata={"version": 1})

        profiles2 = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["humidity"],
            reference_name="test_ref",
        )
        manager.save_reference(profiles2, "test_ref", metadata={"version": 2})

        loaded = manager.load_reference("test_ref")
        metadata = manager.get_reference_metadata("test_ref")

        assert "humidity" in loaded
        assert "temperature_2m" not in loaded
        assert metadata["version"] == 2

    def test_load_nonexistent_raises(self, manager):
        """Loading nonexistent reference should raise."""
        with pytest.raises(ReferenceNotFoundError):
            manager.load_reference("nonexistent")


class TestListReferences:
    """Tests for listing references."""

    def test_list_empty_storage(self, manager):
        """Should return empty list for empty storage."""
        refs = manager.list_references()
        assert refs == []

    def test_list_multiple_references(self, manager, sample_dataframe):
        """Should list all saved references."""
        for name in ["ref_a", "ref_b", "ref_c"]:
            profiles = manager.create_reference_from_dataframe(
                df=sample_dataframe,
                feature_columns=["temperature_2m"],
                reference_name=name,
            )
            manager.save_reference(profiles, name)

        refs = manager.list_references()

        assert len(refs) == 3
        names = [r["reference_name"] for r in refs]
        assert "ref_a" in names
        assert "ref_b" in names
        assert "ref_c" in names

    def test_list_includes_metadata(self, manager, sample_dataframe):
        """Listed references should include metadata."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m", "humidity"],
            reference_name="test_ref",
        )
        manager.save_reference(profiles, "test_ref")

        refs = manager.list_references()

        assert len(refs) == 1
        ref = refs[0]
        assert ref["reference_name"] == "test_ref"
        assert ref["n_features"] == 2
        assert "temperature_2m" in ref["features"]
        assert "humidity" in ref["features"]


class TestDeleteReference:
    """Tests for deleting references."""

    def test_delete_existing_reference(self, manager, sample_dataframe):
        """Should delete existing reference."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
        )
        manager.save_reference(profiles, "test_ref")

        assert manager.reference_exists("test_ref")

        result = manager.delete_reference("test_ref")

        assert result is True
        assert not manager.reference_exists("test_ref")

    def test_delete_nonexistent_returns_false(self, manager):
        """Deleting nonexistent reference should return False."""
        result = manager.delete_reference("nonexistent")
        assert result is False


class TestReferenceExists:
    """Tests for checking reference existence."""

    def test_exists_returns_true_for_existing(self, manager, sample_dataframe):
        """Should return True for existing reference."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
        )
        manager.save_reference(profiles, "test_ref")

        assert manager.reference_exists("test_ref") is True

    def test_exists_returns_false_for_nonexistent(self, manager):
        """Should return False for nonexistent reference."""
        assert manager.reference_exists("nonexistent") is False


class TestErrorHandling:
    """Tests for error handling."""

    def test_corrupted_metadata_raises(self, manager, sample_dataframe, temp_storage_path):
        """Should raise ReferenceCorruptedError for corrupted metadata."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
        )
        manager.save_reference(profiles, "test_ref")

        # Corrupt the metadata file
        metadata_path = Path(temp_storage_path) / "test_ref" / "metadata.json"
        with open(metadata_path, "w") as f:
            f.write("invalid json {{{")

        with pytest.raises(ReferenceCorruptedError, match="corrupted metadata"):
            manager.load_reference("test_ref")

    def test_corrupted_profile_raises(self, manager, sample_dataframe, temp_storage_path):
        """Should raise ReferenceCorruptedError for corrupted profile."""
        profiles = manager.create_reference_from_dataframe(
            df=sample_dataframe,
            feature_columns=["temperature_2m"],
            reference_name="test_ref",
        )
        manager.save_reference(profiles, "test_ref")

        # Corrupt a profile file
        profile_path = Path(temp_storage_path) / "test_ref" / "temperature_2m.json"
        with open(profile_path, "w") as f:
            f.write('{"incomplete": true}')

        with pytest.raises(ReferenceCorruptedError, match="corrupted"):
            manager.load_reference("test_ref")

    def test_missing_metadata_file_raises(self, manager, temp_storage_path):
        """Should raise ReferenceCorruptedError if metadata.json is missing."""
        # Create directory without metadata
        ref_dir = Path(temp_storage_path) / "bad_ref"
        ref_dir.mkdir()

        with pytest.raises(ReferenceCorruptedError, match="missing metadata"):
            manager.load_reference("bad_ref")
