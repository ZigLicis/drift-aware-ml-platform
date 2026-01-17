"""Model evaluation utilities for weather prediction.

This module provides comprehensive evaluation metrics, visualizations,
and reporting for regression models.

Example:
    >>> from src.training.evaluation import ModelEvaluator
    >>> evaluator = ModelEvaluator()
    >>> metrics = evaluator.evaluate(y_true, y_pred)
    >>> report = evaluator.create_evaluation_report(y_true, y_pred, features, model)
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate regression model performance with multiple metrics.

    Provides methods for calculating metrics, creating visualizations,
    and generating comprehensive evaluation reports.

    Attributes:
        metrics: List of metric names to calculate.

    Example:
        >>> evaluator = ModelEvaluator(metrics=["rmse", "mae", "r2"])
        >>> results = evaluator.evaluate(y_true, y_pred)
        >>> print(f"RMSE: {results['rmse']:.2f}")
    """

    # Available metrics and their functions
    METRIC_FUNCTIONS = {
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "r2": r2_score,
        "mape": mean_absolute_percentage_error,
    }

    DEFAULT_METRICS = ["rmse", "mae", "r2", "mape"]

    def __init__(self, metrics: list[str] | None = None) -> None:
        """Initialize the evaluator.

        Args:
            metrics: List of metric names to calculate.
                    Options: rmse, mse, mae, r2, mape.
                    Defaults to ["rmse", "mae", "r2", "mape"].
        """
        self.metrics = metrics or self.DEFAULT_METRICS

        # Validate metrics
        invalid = set(self.metrics) - set(self.METRIC_FUNCTIONS.keys())
        if invalid:
            raise ValueError(
                f"Unknown metrics: {invalid}. "
                f"Available: {list(self.METRIC_FUNCTIONS.keys())}"
            )

        logger.info(f"ModelEvaluator initialized with metrics: {self.metrics}")

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """Calculate all configured metrics.

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            Dictionary mapping metric names to values.

        Example:
            >>> metrics = evaluator.evaluate(y_test, predictions)
            >>> print(f"RMSE: {metrics['rmse']:.3f}")
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        results = {}
        for metric_name in self.metrics:
            try:
                value = self.METRIC_FUNCTIONS[metric_name](y_true, y_pred)
                results[metric_name] = float(value)
            except Exception as e:
                logger.warning(f"Could not calculate {metric_name}: {e}")
                results[metric_name] = np.nan

        return results

    def evaluate_by_time(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        timestamps: pd.Series,
        freq: str = "D",
    ) -> pd.DataFrame:
        """Evaluate metrics grouped by time period.

        Useful for detecting temporal patterns in model errors.

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            timestamps: Timestamps corresponding to each prediction.
            freq: Grouping frequency ("H" for hourly, "D" for daily, "W" for weekly).

        Returns:
            DataFrame with metrics for each time period.

        Example:
            >>> time_metrics = evaluator.evaluate_by_time(y_true, y_pred, timestamps, freq="D")
            >>> print(time_metrics.head())
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps),
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": np.asarray(y_true) - np.asarray(y_pred),
        })

        # Group by time period
        df["period"] = df["timestamp"].dt.to_period(freq)

        results = []
        for period, group in df.groupby("period"):
            if len(group) < 2:
                continue

            period_metrics = {
                "period": str(period),
                "n_samples": len(group),
                "mean_residual": group["residual"].mean(),
                "std_residual": group["residual"].std(),
            }

            # Add configured metrics
            for metric_name in self.metrics:
                try:
                    value = self.METRIC_FUNCTIONS[metric_name](
                        group["y_true"], group["y_pred"]
                    )
                    period_metrics[metric_name] = float(value)
                except Exception:
                    period_metrics[metric_name] = np.nan

            results.append(period_metrics)

        return pd.DataFrame(results)

    def create_evaluation_report(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
        model: Any = None,
    ) -> dict[str, Any]:
        """Create comprehensive evaluation report.

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            feature_names: List of feature names (for importance).
            model: Trained model object (for extracting importance).

        Returns:
            Dictionary containing:
            - metrics: All calculated metrics
            - residual_stats: Residual distribution statistics
            - feature_importance: Feature importance (if available)
            - error_percentiles: Error distribution percentiles

        Example:
            >>> report = evaluator.create_evaluation_report(y_true, y_pred, features, model)
            >>> print(f"RMSE: {report['metrics']['rmse']:.3f}")
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_true - y_pred

        # Calculate metrics
        metrics = self.evaluate(y_true, y_pred)

        # Residual statistics
        residual_stats = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
            "skewness": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
        }

        # Error percentiles
        abs_errors = np.abs(residuals)
        error_percentiles = {
            "p50": float(np.percentile(abs_errors, 50)),
            "p90": float(np.percentile(abs_errors, 90)),
            "p95": float(np.percentile(abs_errors, 95)),
            "p99": float(np.percentile(abs_errors, 99)),
        }

        # Feature importance (if model provided)
        feature_importance = None
        if model is not None and hasattr(model, "get_feature_importance"):
            feature_importance = model.get_feature_importance(feature_names)

        report = {
            "metrics": metrics,
            "residual_stats": residual_stats,
            "error_percentiles": error_percentiles,
            "n_samples": len(y_true),
        }

        if feature_importance:
            report["feature_importance"] = feature_importance

        return report

    def plot_predictions(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        timestamps: pd.Series | None = None,
        title: str = "Predictions vs Actual",
    ) -> plt.Figure:
        """Plot predicted vs actual values.

        Creates either a time series plot (if timestamps provided)
        or a scatter plot.

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            timestamps: Optional timestamps for time series plot.
            title: Plot title.

        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Left: Scatter plot
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", s=20)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate R2 for annotation
        r2 = r2_score(y_true, y_pred)
        ax.annotate(
            f"R² = {r2:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Right: Time series or histogram
        ax = axes[1]
        if timestamps is not None:
            timestamps = pd.to_datetime(timestamps)
            ax.plot(timestamps, y_true, label="Actual", alpha=0.7)
            ax.plot(timestamps, y_pred, label="Predicted", alpha=0.7)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title("Predictions Over Time")
            ax.legend()
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            # Histogram of errors
            residuals = y_true - y_pred
            ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(x=0, color="r", linestyle="--", lw=2)
            ax.set_xlabel("Residual (Actual - Predicted)")
            ax.set_ylabel("Frequency")
            ax.set_title("Residual Distribution")

        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> plt.Figure:
        """Create residual analysis plots.

        Creates:
        - Residuals vs predicted values
        - Residual distribution histogram
        - Q-Q plot for normality check

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            Matplotlib figure.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Residuals vs Predicted
        ax = axes[0]
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none", s=20)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Predicted")
        ax.grid(True, alpha=0.3)

        # 2. Residual histogram
        ax = axes[1]
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7, density=True)
        ax.axvline(x=0, color="r", linestyle="--", lw=2)

        # Overlay normal distribution
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), "orange", lw=2, label="Normal fit")

        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual Distribution\n(μ={mu:.2f}, σ={std:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Q-Q plot
        ax = axes[2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Normality Check)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        importance_dict: dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
    ) -> plt.Figure:
        """Create feature importance bar chart.

        Args:
            importance_dict: Dictionary mapping feature names to importance.
            top_n: Number of top features to display.
            title: Plot title.

        Returns:
            Matplotlib figure.
        """
        # Sort by importance
        sorted_items = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))

        y_pos = np.arange(len(names))
        colors = ["green" if v >= 0 else "red" for v in values]

        ax.barh(y_pos, values, align="center", color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def plot_error_by_time(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        timestamps: pd.Series,
    ) -> plt.Figure:
        """Plot error metrics over time.

        Useful for identifying when model performance degrades.

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            timestamps: Timestamps for each prediction.

        Returns:
            Matplotlib figure.
        """
        time_metrics = self.evaluate_by_time(y_true, y_pred, timestamps, freq="D")

        if len(time_metrics) < 2:
            logger.warning("Not enough data points for temporal error plot")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        periods = pd.to_datetime(time_metrics["period"].astype(str))

        # RMSE over time
        ax = axes[0]
        if "rmse" in time_metrics.columns:
            ax.plot(periods, time_metrics["rmse"], marker="o", label="RMSE")
        if "mae" in time_metrics.columns:
            ax.plot(periods, time_metrics["mae"], marker="s", label="MAE")
        ax.set_ylabel("Error")
        ax.set_title("Error Metrics Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mean residual (bias) over time
        ax = axes[1]
        ax.bar(periods, time_metrics["mean_residual"], alpha=0.7, width=0.8)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Residual (Bias)")
        ax.set_title("Model Bias Over Time")
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        return fig
