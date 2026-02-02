"""MLflow integration for drift detection logging.

This module provides the DriftMLflowLogger class for logging drift detection
results to MLflow, enabling tracking, visualization, and trend analysis.

Example:
    >>> from src.drift_detection import DriftMLflowLogger, DriftDetector
    >>> logger = DriftMLflowLogger(
    ...     tracking_uri="http://localhost:5001",
    ...     experiment_name="drift-monitoring"
    ... )
    >>> report = detector.detect_drift("baseline", current_data=df)
    >>> run_id = logger.log_drift_report(report, model_name="weather-model")
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.drift_detection.detector import DriftReport, DriftSeverity

logger = logging.getLogger(__name__)


class DriftMLflowLoggerError(Exception):
    """Base exception for DriftMLflowLogger errors."""
    pass


class MLflowConnectionError(DriftMLflowLoggerError):
    """Raised when MLflow server connection fails."""
    pass


class DriftMLflowLogger:
    """Logs drift detection results to MLflow for tracking and visualization.

    This class provides methods to:
    - Log complete drift reports with metrics, parameters, and artifacts
    - Create visualizations (heatmaps, timelines)
    - Query drift history for trend analysis

    All logging operations handle MLflow connection errors gracefully,
    ensuring drift detection continues even if logging fails.

    Attributes:
        tracking_uri: MLflow server URI.
        experiment_name: Name of the MLflow experiment.
        experiment_id: MLflow experiment ID.
        client: MLflow tracking client.

    Example:
        >>> logger = DriftMLflowLogger(
        ...     tracking_uri="http://localhost:5001",
        ...     experiment_name="drift-monitoring"
        ... )
        >>> run_id = logger.log_drift_report(report, model_name="weather-model")
    """

    DEFAULT_EXPERIMENT_NAME = "drift-monitoring"
    DEFAULT_TAGS = {
        "project": "domain-shift-ml-platform",
        "component": "drift-detection",
    }

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "drift-monitoring",
    ) -> None:
        """Initialize the drift MLflow logger.

        Args:
            tracking_uri: MLflow server URI (e.g., "http://localhost:5001").
            experiment_name: MLflow experiment for drift runs.

        Raises:
            MLflowConnectionError: If connection to MLflow server fails.
            ImportError: If mlflow package is not installed.
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow package is required for DriftMLflowLogger. "
                "Install with: pip install mlflow"
            )

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._connected = False

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Verify connection
        self._verify_connection()

        # Create MLflow client
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment()

        self._connected = True
        logger.info(
            f"DriftMLflowLogger initialized: uri={tracking_uri}, "
            f"experiment={experiment_name}, experiment_id={self.experiment_id}"
        )

    def _verify_connection(self) -> None:
        """Verify connection to MLflow server."""
        try:
            mlflow.search_experiments(max_results=1)
            logger.debug(f"Connected to MLflow at {self.tracking_uri}")
        except MlflowException as e:
            raise MLflowConnectionError(
                f"Failed to connect to MLflow server at {self.tracking_uri}: {e}"
            ) from e
        except Exception as e:
            raise MLflowConnectionError(
                f"Unexpected error connecting to MLflow at {self.tracking_uri}: {e}"
            ) from e

    def _get_or_create_experiment(self) -> str:
        """Get existing experiment or create new one."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                tags={
                    **self.DEFAULT_TAGS,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.debug(f"Using existing experiment: {self.experiment_name}")

        return experiment_id

    def log_drift_report(
        self,
        report: DriftReport,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Log a complete drift report to MLflow.

        Creates a new MLflow run with parameters, metrics, artifacts, and tags
        capturing all drift detection results.

        Parameters logged:
            - reference_name
            - current_window_hours (calculated from window)
            - detection_method (from config)
            - n_features_monitored
            - model_name, model_version (if provided)

        Metrics logged:
            - overall_drift_score
            - n_features_drifted
            - Per feature: {feature}_psi, {feature}_ks_statistic, etc.

        Artifacts logged:
            - drift_report.json (complete report)
            - drift_summary.txt (human-readable summary)
            - drift_heatmap.png (visual of drift across features)

        Tags:
            - drift_detected: "true"/"false"
            - overall_severity: severity level
            - model_name, model_version (if provided)

        Args:
            report: DriftReport to log.
            model_name: Associated model name in registry.
            model_version: Associated model version.
            tags: Additional tags.

        Returns:
            MLflow run_id.

        Raises:
            DriftMLflowLoggerError: If logging fails (non-connection errors).
        """
        run_id = None

        try:
            # Calculate window hours
            window_hours = int(
                (report.current_window_end - report.current_window_start).total_seconds() / 3600
            )

            # Build run tags
            run_tags = {
                **self.DEFAULT_TAGS,
                "drift_detected": str(report.drift_detected).lower(),
                "overall_severity": report.overall_severity.value,
                "report_id": report.report_id,
            }
            if model_name:
                run_tags["model_name"] = model_name
            if model_version:
                run_tags["model_version"] = str(model_version)
            if tags:
                run_tags.update(tags)

            # Start run
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"drift-{report.report_id}",
                tags=run_tags,
            ) as run:
                run_id = run.info.run_id

                # Log parameters
                params = {
                    "reference_name": report.reference_name,
                    "current_window_hours": window_hours,
                    "n_features_monitored": report.n_features_total,
                    "reference_samples": report.reference_samples,
                    "current_samples": report.current_samples,
                }
                if model_name:
                    params["model_name"] = model_name
                if model_version:
                    params["model_version"] = str(model_version)

                mlflow.log_params(params)

                # Log overall metrics
                mlflow.log_metrics({
                    "overall_drift_score": report.overall_drift_score,
                    "n_features_drifted": report.n_features_drifted,
                    "drift_ratio": report.n_features_drifted / report.n_features_total,
                })

                # Log per-feature metrics
                self._log_feature_metrics(report)

                # Log artifacts
                self._log_artifacts(report)

                logger.info(
                    f"Logged drift report {report.report_id} to MLflow run {run_id[:8]}"
                )

            return run_id

        except MlflowException as e:
            logger.error(f"MLflow error logging drift report: {e}")
            raise DriftMLflowLoggerError(f"Failed to log drift report: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error logging drift report: {e}")
            raise DriftMLflowLoggerError(f"Failed to log drift report: {e}") from e

    def _log_feature_metrics(self, report: DriftReport) -> None:
        """Log per-feature metrics to active MLflow run."""
        for feature_name, result in report.feature_results.items():
            # Sanitize feature name for metric naming
            safe_name = self._sanitize_metric_name(feature_name)

            metrics = {
                f"{safe_name}_psi": result.psi,
                f"{safe_name}_ks_statistic": result.ks_statistic,
                f"{safe_name}_ks_p_value": result.ks_p_value,
                f"{safe_name}_js_divergence": result.js_divergence,
                f"{safe_name}_wasserstein": result.wasserstein,
                f"{safe_name}_mean_shift": result.mean_shift,
            }

            mlflow.log_metrics(metrics)

    def _sanitize_metric_name(self, name: str) -> str:
        """Sanitize feature name for use as MLflow metric name."""
        # Replace problematic characters
        safe = name.replace(" ", "_").replace("-", "_").replace(".", "_")
        # Truncate if too long
        return safe[:50]

    def _log_artifacts(self, report: DriftReport) -> None:
        """Log artifacts (JSON report, summary, heatmap) to active MLflow run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Complete JSON report
            report_path = tmpdir / "drift_report.json"
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            mlflow.log_artifact(str(report_path))

            # 2. Human-readable summary
            summary_path = tmpdir / "drift_summary.txt"
            with open(summary_path, "w") as f:
                f.write(report.summary())
            mlflow.log_artifact(str(summary_path))

            # 3. Drift heatmap
            try:
                heatmap_path = self.create_drift_heatmap(report, output_dir=str(tmpdir))
                mlflow.log_artifact(heatmap_path, artifact_path="plots")
            except Exception as e:
                logger.warning(f"Failed to create drift heatmap: {e}")

    def create_drift_heatmap(
        self,
        report: DriftReport,
        output_dir: Optional[str] = None,
    ) -> str:
        """Create a heatmap visualization of drift across features.

        Creates a heatmap where:
        - Rows: features
        - Columns: PSI, KS-stat, JS-div (normalized 0-1)
        - Color: green (no drift) -> yellow (moderate) -> red (severe)

        Args:
            report: DriftReport to visualize.
            output_dir: Directory to save plot. Uses tempdir if not provided.

        Returns:
            Path to saved plot file.
        """
        # Prepare data
        features = list(report.feature_results.keys())
        n_features = len(features)

        # Collect metrics (normalize to 0-1 range for visualization)
        psi_values = []
        ks_values = []
        js_values = []

        for feature in features:
            result = report.feature_results[feature]
            # Normalize: PSI typically 0-1+, cap at 1 for display
            psi_values.append(min(result.psi, 1.0))
            # KS statistic is already 0-1
            ks_values.append(result.ks_statistic)
            # JS divergence 0-1
            js_values.append(min(result.js_divergence, 1.0))

        # Create matrix
        data = np.array([psi_values, ks_values, js_values]).T
        metrics = ["PSI", "KS-stat", "JS-div"]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, max(4, n_features * 0.4)))

        # Custom colormap: green -> yellow -> red
        from matplotlib.colors import LinearSegmentedColormap
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]  # green, yellow, red
        cmap = LinearSegmentedColormap.from_list("drift", colors)

        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(features)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Add text annotations
        for i in range(n_features):
            for j in range(len(metrics)):
                value = data[i, j]
                # Choose text color based on background
                text_color = "white" if value > 0.5 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                       color=text_color, fontsize=9)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Drift Score (normalized)", rotation=270, labelpad=15)

        # Add severity indicator on right
        severity_colors = {
            DriftSeverity.NONE: "#2ecc71",
            DriftSeverity.LOW: "#27ae60",
            DriftSeverity.MODERATE: "#f1c40f",
            DriftSeverity.SIGNIFICANT: "#e67e22",
            DriftSeverity.SEVERE: "#e74c3c",
        }

        for i, feature in enumerate(features):
            result = report.feature_results[feature]
            color = severity_colors.get(result.severity, "#95a5a6")
            ax.add_patch(plt.Rectangle((len(metrics) - 0.5 + 0.1, i - 0.4),
                                       0.3, 0.8, color=color, clip_on=False))

        # Title
        status = "DRIFT DETECTED" if report.drift_detected else "NO DRIFT"
        ax.set_title(
            f"Drift Detection Heatmap - {status}\n"
            f"Reference: {report.reference_name} | Score: {report.overall_drift_score:.3f}",
            fontsize=11, fontweight="bold"
        )

        plt.tight_layout()

        # Save
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        output_path = Path(output_dir) / "drift_heatmap.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Created drift heatmap at {output_path}")
        return str(output_path)

    def create_drift_timeline(
        self,
        feature_name: str,
        n_recent_runs: int = 30,
        output_dir: Optional[str] = None,
    ) -> str:
        """Create timeline plot of drift metrics for a feature over recent runs.

        Queries recent MLflow runs in the drift-monitoring experiment and plots
        PSI over time with threshold lines.

        Args:
            feature_name: Feature to plot timeline for.
            n_recent_runs: Number of recent runs to include.
            output_dir: Directory to save plot. Uses tempdir if not provided.

        Returns:
            Path to saved plot file.
        """
        # Get drift history
        history_df = self.get_drift_history(n_runs=n_recent_runs)

        if history_df.empty:
            raise DriftMLflowLoggerError("No drift history available for timeline")

        safe_name = self._sanitize_metric_name(feature_name)
        psi_col = f"{safe_name}_psi"

        if psi_col not in history_df.columns:
            raise DriftMLflowLoggerError(
                f"Feature '{feature_name}' not found in drift history"
            )

        # Sort by timestamp
        history_df = history_df.sort_values("timestamp")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot PSI over time
        timestamps = pd.to_datetime(history_df["timestamp"])
        psi_values = history_df[psi_col]

        ax.plot(timestamps, psi_values, marker="o", linewidth=2,
                markersize=6, color="#3498db", label=f"{feature_name} PSI")

        # Add threshold lines
        ax.axhline(y=0.1, color="#27ae60", linestyle="--",
                   linewidth=1.5, alpha=0.8, label="Low threshold (0.1)")
        ax.axhline(y=0.15, color="#f1c40f", linestyle="--",
                   linewidth=1.5, alpha=0.8, label="Moderate threshold (0.15)")
        ax.axhline(y=0.2, color="#e67e22", linestyle="--",
                   linewidth=1.5, alpha=0.8, label="Significant threshold (0.2)")
        ax.axhline(y=0.5, color="#e74c3c", linestyle="--",
                   linewidth=1.5, alpha=0.8, label="Severe threshold (0.5)")

        # Fill regions
        ax.fill_between(timestamps, 0, 0.1, alpha=0.1, color="#27ae60")
        ax.fill_between(timestamps, 0.1, 0.15, alpha=0.1, color="#f1c40f")
        ax.fill_between(timestamps, 0.15, 0.2, alpha=0.1, color="#e67e22")
        ax.fill_between(timestamps, 0.2, 0.5, alpha=0.1, color="#e74c3c")

        # Formatting
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("PSI", fontsize=11)
        ax.set_title(f"Drift Timeline: {feature_name}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        safe_filename = feature_name.replace(" ", "_").replace("/", "_")
        output_path = Path(output_dir) / f"drift_timeline_{safe_filename}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Created drift timeline at {output_path}")
        return str(output_path)

    def get_drift_history(
        self,
        n_runs: int = 30,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve drift detection history from MLflow.

        Queries recent runs from the drift-monitoring experiment and returns
        a DataFrame with drift metrics for trend analysis.

        Args:
            n_runs: Maximum number of runs to retrieve.
            model_name: Filter by model name tag (optional).

        Returns:
            DataFrame with columns:
            - run_id, timestamp, overall_drift_score, n_features_drifted
            - Per-feature PSI values ({feature}_psi)
            - drift_detected flag
        """
        try:
            # Build filter string
            filter_string = ""
            if model_name:
                filter_string = f"tags.model_name = '{model_name}'"

            # Query runs
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"],
                max_results=n_runs,
            )

            if not runs:
                logger.warning("No drift runs found in history")
                return pd.DataFrame()

            # Extract data
            records = []
            for run in runs:
                record = {
                    "run_id": run.info.run_id,
                    "timestamp": datetime.fromtimestamp(
                        run.info.start_time / 1000, tz=timezone.utc
                    ),
                    "drift_detected": run.data.tags.get("drift_detected", "false") == "true",
                    "overall_severity": run.data.tags.get("overall_severity", "unknown"),
                }

                # Add metrics
                for metric_name, metric_value in run.data.metrics.items():
                    record[metric_name] = metric_value

                records.append(record)

            df = pd.DataFrame(records)
            logger.debug(f"Retrieved {len(df)} drift runs from history")
            return df

        except MlflowException as e:
            logger.error(f"MLflow error retrieving drift history: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error retrieving drift history: {e}")
            return pd.DataFrame()

    def log_drift_report_safe(
        self,
        report: DriftReport,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Log drift report with graceful error handling.

        Same as log_drift_report but catches all errors and returns None
        instead of raising. Use this when drift detection should continue
        even if logging fails.

        Args:
            report: DriftReport to log.
            model_name: Associated model name.
            model_version: Associated model version.
            tags: Additional tags.

        Returns:
            MLflow run_id if successful, None if logging failed.
        """
        try:
            return self.log_drift_report(report, model_name, model_version, tags)
        except Exception as e:
            logger.error(f"Failed to log drift report (non-fatal): {e}")
            return None

    @property
    def is_connected(self) -> bool:
        """Check if logger is connected to MLflow."""
        return self._connected
