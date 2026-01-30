"""
Reference data manager for drift detection.

This module handles storing, loading, and managing reference distributions
used to detect data drift. Reference profiles are created from training data
and compared against new data to detect distribution shifts.

Example:
    >>> from src.drift_detection import ReferenceManager
    >>> manager = ReferenceManager(storage_path="data/references")
    >>> profiles = manager.create_reference_from_dataframe(
    ...     df=training_df,
    ...     feature_columns=["temperature_2m", "humidity"],
    ...     reference_name="baseline_v1"
    ... )
    >>> manager.save_reference(profiles, "baseline_v1")
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReferenceProfile:
    """Profile of reference data for drift comparison.

    Contains statistical summaries and histogram data needed for
    drift detection without storing the full raw dataset.

    Attributes:
        feature_name: Name of the feature this profile represents.
        created_at: When this profile was created.
        n_samples: Number of samples used to create the profile.
        mean: Mean value of the feature.
        std: Standard deviation of the feature.
        min: Minimum value.
        max: Maximum value.
        median: Median value.
        quantiles: Dictionary of quantile values (5%, 25%, 75%, 95%).
        bin_edges: Histogram bin edges for PSI calculation.
        bin_counts: Histogram bin counts.
        values: Optional raw values for KS-test (memory intensive).
    """

    feature_name: str
    created_at: datetime
    n_samples: int

    # Distribution statistics
    mean: float
    std: float
    min: float
    max: float
    median: float
    quantiles: Dict[str, float]

    # For PSI calculation
    bin_edges: np.ndarray
    bin_counts: np.ndarray

    # Raw values (optional, for KS-test)
    values: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to JSON-serializable dictionary."""
        return {
            "feature_name": self.feature_name,
            "created_at": self.created_at.isoformat(),
            "n_samples": self.n_samples,
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min),
            "max": float(self.max),
            "median": float(self.median),
            "quantiles": {k: float(v) for k, v in self.quantiles.items()},
            "bin_edges": self.bin_edges.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "values": self.values.tolist() if self.values is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReferenceProfile:
        """Create ReferenceProfile from dictionary."""
        return cls(
            feature_name=data["feature_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            n_samples=data["n_samples"],
            mean=data["mean"],
            std=data["std"],
            min=data["min"],
            max=data["max"],
            median=data["median"],
            quantiles=data["quantiles"],
            bin_edges=np.array(data["bin_edges"]),
            bin_counts=np.array(data["bin_counts"]),
            values=np.array(data["values"]) if data.get("values") is not None else None,
        )


class ReferenceManagerError(Exception):
    """Base exception for ReferenceManager errors."""

    pass


class ReferenceNotFoundError(ReferenceManagerError):
    """Raised when a reference is not found."""

    pass


class ReferenceCorruptedError(ReferenceManagerError):
    """Raised when reference data is corrupted or invalid."""

    pass


class ReferenceManager:
    """Manages reference data profiles for drift detection.

    Handles creating, storing, loading, and managing reference profiles
    used to compare against new data for drift detection.

    Attributes:
        storage_path: Path to the directory where references are stored.

    Example:
        >>> manager = ReferenceManager(storage_path="data/references")
        >>> profiles = manager.create_reference_from_dataframe(
        ...     df=training_df,
        ...     feature_columns=["temperature_2m", "humidity"],
        ...     reference_name="baseline_v1"
        ... )
        >>> manager.save_reference(profiles, "baseline_v1")
        >>> loaded = manager.load_reference("baseline_v1")
    """

    DEFAULT_STORAGE_PATH = "data/references"
    METADATA_FILENAME = "metadata.json"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        db_connection_string: Optional[str] = None,
    ) -> None:
        """
        Initialize reference manager.

        Uses file-based storage with JSON format for transparency.
        Database storage is reserved for future implementation.

        Args:
            storage_path: Path to store reference files.
                         Defaults to 'data/references/'.
            db_connection_string: Optional database connection (future use).
        """
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self.db_connection_string = db_connection_string

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ReferenceManager with storage at {self.storage_path}")

        if db_connection_string:
            logger.warning("Database storage not yet implemented, using file storage")

    def create_reference_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        reference_name: str,
        n_bins: int = 10,
        store_raw_values: bool = False,
    ) -> Dict[str, ReferenceProfile]:
        """
        Create reference profiles from a DataFrame.

        Typically called after training with the training data to establish
        the baseline distribution for drift detection.

        Args:
            df: Training/reference DataFrame.
            feature_columns: Columns to create profiles for.
            reference_name: Identifier for this reference set.
            n_bins: Number of bins for PSI calculation (default: 10).
            store_raw_values: Whether to store raw values for KS-test.
                             This is memory intensive for large datasets.

        Returns:
            Dict mapping feature names to ReferenceProfile objects.

        Raises:
            ValueError: If DataFrame is empty or columns are missing.

        Example:
            >>> profiles = manager.create_reference_from_dataframe(
            ...     df=train_df,
            ...     feature_columns=["temperature_2m", "humidity"],
            ...     reference_name="baseline_v1"
            ... )
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Validate columns exist
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        profiles: Dict[str, ReferenceProfile] = {}
        created_at = datetime.now(timezone.utc)

        logger.info(
            f"Creating reference '{reference_name}' for {len(feature_columns)} features "
            f"from {len(df)} samples"
        )

        for col in feature_columns:
            profile = self._create_single_profile(
                data=df[col],
                feature_name=col,
                created_at=created_at,
                n_bins=n_bins,
                store_raw_values=store_raw_values,
            )
            profiles[col] = profile
            logger.debug(
                f"Created profile for '{col}': "
                f"mean={profile.mean:.3f}, std={profile.std:.3f}, n={profile.n_samples}"
            )

        logger.info(f"Created {len(profiles)} reference profiles")
        return profiles

    def _create_single_profile(
        self,
        data: pd.Series,
        feature_name: str,
        created_at: datetime,
        n_bins: int,
        store_raw_values: bool,
    ) -> ReferenceProfile:
        """Create a reference profile for a single feature."""
        # Remove NaN values
        clean_data = data.dropna().values.astype(np.float64)

        if len(clean_data) == 0:
            raise ValueError(f"Feature '{feature_name}' contains only NaN values")

        # Calculate basic statistics
        # Use ddof=1 for sample std to match pandas behavior
        mean = float(np.mean(clean_data))
        std = float(np.std(clean_data, ddof=1))
        min_val = float(np.min(clean_data))
        max_val = float(np.max(clean_data))
        median = float(np.median(clean_data))

        # Calculate quantiles
        quantiles = {
            "5%": float(np.percentile(clean_data, 5)),
            "25%": float(np.percentile(clean_data, 25)),
            "75%": float(np.percentile(clean_data, 75)),
            "95%": float(np.percentile(clean_data, 95)),
        }

        # Create histogram for PSI
        # Use quantile-based bins for better stability
        if std > 0:
            # Create bins that capture the distribution shape
            bin_edges = np.percentile(
                clean_data,
                np.linspace(0, 100, n_bins + 1)
            )
            # Ensure unique bin edges (can happen with discrete data)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([min_val, max_val])
        else:
            # Constant value - single bin
            bin_edges = np.array([min_val, max_val + 1e-10])

        bin_counts, _ = np.histogram(clean_data, bins=bin_edges)

        # Optionally store raw values
        values = clean_data if store_raw_values else None

        return ReferenceProfile(
            feature_name=feature_name,
            created_at=created_at,
            n_samples=len(clean_data),
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            median=median,
            quantiles=quantiles,
            bin_edges=bin_edges,
            bin_counts=bin_counts,
            values=values,
        )

    def save_reference(
        self,
        profiles: Dict[str, ReferenceProfile],
        reference_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save reference profiles to storage.

        Creates a directory for the reference with individual JSON files
        for each feature profile and a metadata file.

        Args:
            profiles: Feature profiles to save.
            reference_name: Identifier for this reference.
            metadata: Additional metadata (model version, date range, etc.).

        Returns:
            Path where reference was saved.

        Raises:
            ReferenceManagerError: If saving fails.

        Example:
            >>> path = manager.save_reference(
            ...     profiles,
            ...     "baseline_v1",
            ...     metadata={"model_version": "1.0", "date_range": "2024-01-01 to 2024-06-01"}
            ... )
        """
        reference_dir = self.storage_path / reference_name

        # Create directory (remove existing if present)
        if reference_dir.exists():
            logger.warning(f"Overwriting existing reference '{reference_name}'")
            shutil.rmtree(reference_dir)

        reference_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build metadata
            full_metadata = {
                "reference_name": reference_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "n_features": len(profiles),
                "features": list(profiles.keys()),
                "total_samples": (
                    profiles[list(profiles.keys())[0]].n_samples if profiles else 0
                ),
                **(metadata or {}),
            }

            # Save metadata
            metadata_path = reference_dir / self.METADATA_FILENAME
            with open(metadata_path, "w") as f:
                json.dump(full_metadata, f, indent=2)
            logger.debug(f"Saved metadata to {metadata_path}")

            # Save each profile
            for feature_name, profile in profiles.items():
                # Sanitize filename (replace special chars)
                safe_name = feature_name.replace("/", "_").replace("\\", "_")
                profile_path = reference_dir / f"{safe_name}.json"

                with open(profile_path, "w") as f:
                    json.dump(profile.to_dict(), f, indent=2)
                logger.debug(f"Saved profile for '{feature_name}' to {profile_path}")

            logger.info(
                f"Saved reference '{reference_name}' with {len(profiles)} profiles "
                f"to {reference_dir}"
            )
            return str(reference_dir)

        except Exception as e:
            # Clean up on failure
            if reference_dir.exists():
                shutil.rmtree(reference_dir)
            raise ReferenceManagerError(f"Failed to save reference: {e}") from e

    def load_reference(
        self,
        reference_name: str,
    ) -> Dict[str, ReferenceProfile]:
        """
        Load reference profiles from storage.

        Args:
            reference_name: Identifier for the reference.

        Returns:
            Dict mapping feature names to ReferenceProfile objects.

        Raises:
            ReferenceNotFoundError: If reference doesn't exist.
            ReferenceCorruptedError: If reference data is corrupted.

        Example:
            >>> profiles = manager.load_reference("baseline_v1")
            >>> temp_profile = profiles["temperature_2m"]
        """
        reference_dir = self.storage_path / reference_name

        if not reference_dir.exists():
            raise ReferenceNotFoundError(f"Reference '{reference_name}' not found")

        if not reference_dir.is_dir():
            raise ReferenceCorruptedError(
                f"Reference '{reference_name}' is not a valid directory"
            )

        # Load metadata to get feature list
        metadata_path = reference_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            raise ReferenceCorruptedError(
                f"Reference '{reference_name}' is missing metadata.json"
            )

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ReferenceCorruptedError(
                f"Reference '{reference_name}' has corrupted metadata: {e}"
            ) from e

        # Load each profile
        profiles: Dict[str, ReferenceProfile] = {}
        features = metadata.get("features", [])

        for feature_name in features:
            safe_name = feature_name.replace("/", "_").replace("\\", "_")
            profile_path = reference_dir / f"{safe_name}.json"

            if not profile_path.exists():
                logger.warning(
                    f"Profile for '{feature_name}' not found, skipping"
                )
                continue

            try:
                with open(profile_path) as f:
                    profile_data = json.load(f)
                profiles[feature_name] = ReferenceProfile.from_dict(profile_data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ReferenceCorruptedError(
                    f"Profile for '{feature_name}' is corrupted: {e}"
                ) from e

        logger.info(
            f"Loaded reference '{reference_name}' with {len(profiles)} profiles"
        )
        return profiles

    def list_references(self) -> List[Dict[str, Any]]:
        """
        List all available references.

        Returns:
            List of dicts with reference metadata including:
            - reference_name
            - created_at
            - n_features
            - features
            - total_samples

        Example:
            >>> refs = manager.list_references()
            >>> for ref in refs:
            ...     print(f"{ref['reference_name']}: {ref['n_features']} features")
        """
        references = []

        if not self.storage_path.exists():
            return references

        for item in self.storage_path.iterdir():
            if not item.is_dir():
                continue

            metadata_path = item / self.METADATA_FILENAME
            if not metadata_path.exists():
                logger.debug(f"Skipping {item.name}: no metadata.json")
                continue

            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                references.append(metadata)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read metadata for {item.name}: {e}")
                continue

        # Sort by created_at (newest first)
        references.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        logger.debug(f"Found {len(references)} references")
        return references

    def delete_reference(self, reference_name: str) -> bool:
        """
        Delete a reference profile set.

        Args:
            reference_name: Identifier for the reference to delete.

        Returns:
            True if deleted successfully, False if reference didn't exist.

        Raises:
            ReferenceManagerError: If deletion fails.
        """
        reference_dir = self.storage_path / reference_name

        if not reference_dir.exists():
            logger.warning(f"Reference '{reference_name}' does not exist")
            return False

        try:
            shutil.rmtree(reference_dir)
            logger.info(f"Deleted reference '{reference_name}'")
            return True
        except Exception as e:
            raise ReferenceManagerError(
                f"Failed to delete reference '{reference_name}': {e}"
            ) from e

    def get_reference_metadata(self, reference_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific reference.

        Args:
            reference_name: Identifier for the reference.

        Returns:
            Dict with reference metadata.

        Raises:
            ReferenceNotFoundError: If reference doesn't exist.
        """
        reference_dir = self.storage_path / reference_name
        metadata_path = reference_dir / self.METADATA_FILENAME

        if not metadata_path.exists():
            raise ReferenceNotFoundError(f"Reference '{reference_name}' not found")

        with open(metadata_path) as f:
            return json.load(f)

    def reference_exists(self, reference_name: str) -> bool:
        """Check if a reference exists."""
        reference_dir = self.storage_path / reference_name
        metadata_path = reference_dir / self.METADATA_FILENAME
        return metadata_path.exists()
