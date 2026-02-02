# Domain-Shift ML Platform

**An end-to-end weather prediction system with automated domain shift detection and model retraining.**

![Development Status](https://img.shields.io/badge/status-active%20development-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MLflow](https://img.shields.io/badge/mlflow-2.10.0-blue)
![PostgreSQL](https://img.shields.io/badge/postgresql-15-blue)

---

## Overview

Machine learning models degrade in production when the data distribution shifts from what they were trained on. This platform addresses that problem by building a complete ML pipeline that:

1. **Ingests weather data** from the Open-Meteo API with validation and quality scoring
2. **Engineers features** using temporal encoding, lag features, and rolling statistics
3. **Trains and evaluates models** with temporal train/test splits that prevent data leakage
4. **Tracks experiments** and manages model versions through MLflow
5. **Detects domain shift** and triggers automated retraining (planned)


### Why This Exists

During model development, I observed that a model trained on 6 months of summer/fall data achieved 0.95 R² on training data but **-62 R²** on winter test data, a failure caused by a seasonal domain shift. This project exists to detect and respond to such distribution drifts automatically.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                 Domain-Shift ML Platform                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Open-Meteo │───▶│   Weather   │───▶│    Data     │     │
│  │     API     │    │   Client    │    │  Validator  │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                               │            │
│                                               ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  PostgreSQL │◀───│    Data     │◀───│    Data     │     │
│  │   Database  │    │   Storage   │    │ Transformer │     │
│  └──────┬──────┘    └─────────────┘    └─────────────┘     │
│         │                                                  │
│         ▼                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Feature   │───▶│   Model     │───▶│   Model     │     │
│  │  Engineer   │    │  Trainer    │    │  Evaluator  │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                               │
│                            ▼                               │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │   MLflow    │◀───│  Experiment │                        │
│  │   Server    │    │   Tracker   │                        │
│  └─────────────┘    └─────────────┘                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Data Flow:**
```
Open-Meteo API → WeatherAPIClient → DataValidator → DataTransformer → PostgreSQL
                                                                          ↓
                 MLflow ← ModelTrainer ← FeatureEngineer ← Query historical data
```

---

## Key Features

### Implemented

- **Data Ingestion Pipeline**
  - Open-Meteo API integration with automatic archive/forecast API selection
  - Exponential backoff retry logic with rate limit handling
  - Data validation with quality scoring (completeness, range checks, anomaly detection)
  - Upsert storage to PostgreSQL with connection pooling

- **Feature Engineering**
  - 30 engineered features from 5 raw weather variables
  - Temporal encoding with cyclical (sin/cos) transformations
  - Lag features (1h, 6h, 24h) for temperature and humidity
  - Rolling statistics (6h/24h windows) for mean, std, min, max

- **Model Training**
  - Ridge regression baseline, Random Forest, and Gradient Boosting
  - Temporal train/validation/test splits (prevents data leakage)
  - Automatic model registration when metrics exceed thresholds

- **MLflow Integration**
  - Experiment tracking with parameters, metrics, and artifacts
  - Model registry with staging/production workflows
  - Version comparison and promotion utilities

- **CLI Tools**
  - `run_ingestion.py` - Historical, incremental, and backfill ingestion modes
  - `train_model.py` - Train models with configurable hyperparameters
  - `evaluate_model.py` - Evaluate and compare model versions
  - `verify_ingestion.py` - Health checks for API, database, and MLflow

- **Infrastructure**
  - Docker Compose orchestration (PostgreSQL, MLflow, App)
  - Structured logging with `structlog`
  - Configuration via YAML files and environment variables

- **Drift Detection**
  - Statistical tests: PSI, KS test, Jensen-Shannon divergence, Chi-square, Wasserstein distance
  - Reference data manager for storing/loading baseline distributions
  - **Drift Detector** with severity classification and recommendations
  - Configurable thresholds via `config/drift_config.yaml`
  - `DriftReport` with human-readable summaries and JSON serialization
  - Edge case handling (NaN, empty arrays, constant values, insufficient samples)
  - Integration test validating summer→winter drift detection

- **Drift MLflow Integration**
  - `DriftMLflowLogger` for logging drift reports to MLflow
  - Per-feature metrics tracking (`{feature}_psi`, `{feature}_ks_statistic`, etc.)
  - Drift heatmap visualization (green→yellow→red severity scale)
  - Drift timeline plots with threshold lines
  - History retrieval for trend analysis and dashboards
  - Graceful error handling (drift detection continues if logging fails)

### TODO:

- Automated retraining triggers
- FastAPI REST endpoints for predictions
- Model performance monitoring dashboard

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Database | PostgreSQL 15 |
| ML Tracking | MLflow 2.10.0 |
| ML Framework | scikit-learn 1.4 |
| Data Processing | pandas 2.1, NumPy 1.26, SciPy 1.12 |
| HTTP Client | requests + tenacity (retry) |
| ORM | SQLAlchemy 2.0 |
| Validation | Pydantic 2.5 |
| Containerization | Docker Compose |
| Logging | structlog |

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)

### Quick Start

1. **Clone and configure .env variables**
   ```bash
   git clone https://github.com/ZigLicis/drift-aware-ml-platform.git
   cd domain-shift-ml-platform
   cp .env.example .env
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

3. **Verify services are running**
   ```bash
   docker-compose ps
   ```

4. **Ingest historical data**
   ```bash
   python scripts/run_ingestion.py --mode historical --start 2024-01-01 --end 2025-01-15
   ```

5. **Train a model**
   ```bash
   python scripts/train_model.py --model ridge
   ```

6. **View experiments in MLflow**

   Open http://localhost:5001

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 5432 | localhost:5432 |
| MLflow UI | 5001 | http://localhost:5001 |
| Application | 8000 | http://localhost:8000 |

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Database user | `dsml_user` |
| `POSTGRES_PASSWORD` | Database password | `dsml_password_change_me` |
| `POSTGRES_DB` | Database name | `dsml_db` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5001` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DRIFT_THRESHOLD` | Domain shift detection threshold | `0.1` |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/settings.yaml` | Application settings, locations, model params |
| `config/data_config.yaml` | Ingestion settings, API config, quality thresholds |
| `config/model_config.yaml` | Model training configuration, feature definitions |

---

## Usage

### Data Ingestion

```bash
# Ingest historical data for a date range
python scripts/run_ingestion.py --mode historical --start 2024-01-01 --end 2024-12-31

# Ingest last 24 hours (incremental)
python scripts/run_ingestion.py --mode incremental

# Backfill large date ranges in batches
python scripts/run_ingestion.py --mode backfill --start 2024-01-01 --end 2024-12-31 --batch-days 7

# Health check
python scripts/run_ingestion.py --mode health
```

### Model Training

```bash
# Train Ridge regression (baseline)
python scripts/train_model.py --model ridge

# Train Random Forest with custom parameters
python scripts/train_model.py --model random_forest --n-estimators 100 --max-depth 10

# Train and auto-promote to staging
python scripts/train_model.py --model ridge --promote staging

# Train on specific date range
python scripts/train_model.py --model ridge --start 2024-01-01 --end 2025-01-15
```

### Model Evaluation

```bash
# Evaluate production model
python scripts/evaluate_model.py --model weather-forecaster --stage Production

# Evaluate specific version
python scripts/evaluate_model.py --model weather-forecaster --version 1

# Compare two versions
python scripts/evaluate_model.py --model weather-forecaster --compare 1 2

# View model registry info
python scripts/evaluate_model.py --model weather-forecaster --info
```

### Verification

```bash
# Run all health checks
python scripts/verify_ingestion.py --all

# Check individual components
python scripts/verify_ingestion.py --check-api
python scripts/verify_ingestion.py --check-db
python scripts/verify_ingestion.py --check-mlflow
```

### Drift Detection

```python
from src.drift_detection import DriftDetector, ReferenceManager
import pandas as pd

# Initialize components
manager = ReferenceManager(storage_path="data/references")
detector = DriftDetector(reference_manager=manager)

# Create reference from training data
profiles = manager.create_reference_from_dataframe(
    df=training_df,
    feature_columns=["temperature_2m", "humidity", "precipitation"],
    reference_name="baseline_v1",
    store_raw_values=True
)
manager.save_reference(profiles, "baseline_v1")

# Detect drift against new data
report = detector.detect_drift(
    reference_name="baseline_v1",
    current_data=production_df
)

# Check results
print(report.summary())           # Human-readable report
print(report.drift_detected)      # True if action needed
print(report.overall_severity)    # NONE, LOW, MODERATE, SIGNIFICANT, SEVERE
print(report.recommendations)     # Actionable recommendations

# Serialize for logging
report_dict = report.to_dict()
```

### Drift MLflow Logging

```python
from src.drift_detection import DriftDetector, DriftMLflowLogger, ReferenceManager

# Initialize
manager = ReferenceManager(storage_path="data/references")
detector = DriftDetector(reference_manager=manager)
logger = DriftMLflowLogger(
    tracking_uri="http://localhost:5001",
    experiment_name="drift-monitoring"
)

# Detect drift
report = detector.detect_drift("baseline_v1", current_data=production_df)

# Log to MLflow (creates run with metrics, artifacts, heatmap)
run_id = logger.log_drift_report(
    report,
    model_name="weather-predictor",
    model_version="2"
)

# Safe logging (won't fail if MLflow is down)
run_id = logger.log_drift_report_safe(report)

# Get drift history for dashboards
history_df = logger.get_drift_history(n_runs=30)

# Create timeline visualization
timeline_path = logger.create_drift_timeline("temperature_2m", n_recent_runs=30)
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Skip network-dependent tests
pytest tests/ -v -m "not network"

# Skip integration tests (require running services)
pytest tests/ -v -m "not integration"

# Run with coverage
pytest --cov=src tests/
```

---

## Project Status

### Current State

The foundation is complete with a working end-to-end pipeline:

| Metric | Value |
|--------|-------|
| Data Records | 9,121 hourly observations |
| Date Range | January 2024 - January 2025 |
| Engineered Features | 30 |
| Baseline Model RMSE | 3.38°C |
| Baseline Model R² | 0.30 |

### Roadmap

**Drift Detection & Automation** (In Progress)
- [x] Statistical tests module (PSI, KS, JS divergence, Chi-square, Wasserstein)
- [x] Reference data manager for baseline distributions
- [x] Integration test validating drift detection
- [x] DriftDetector orchestrator with severity classification
- [x] DriftReport with summaries and recommendations
- [x] MLflow logging for drift reports (DriftMLflowLogger)
- [ ] Automated drift alerts and retraining triggers
- [ ] REST API for predictions (FastAPI)

**Future**
- Monitoring dashboard
- Scheduled ingestion jobs
- A/B testing infrastructure

---

## Repository Structure

```
domain-shift-ml-platform/
├── data/
│   └── references/             # Reference profiles for drift detection
├── config/
│   ├── settings.yaml           # Application settings
│   ├── data_config.yaml        # Ingestion configuration
│   ├── drift_config.yaml       # Drift thresholds configuration
│   └── model_config.yaml       # Model training configuration
├── docker/
│   ├── app/Dockerfile          # Application container
│   ├── mlflow/Dockerfile       # MLflow server container
│   └── postgres/init.sql       # Database schema initialization
├── scripts/
│   ├── run_ingestion.py        # Data ingestion CLI
│   ├── verify_ingestion.py     # Verification tool
│   ├── train_model.py          # Model training CLI
│   ├── evaluate_model.py       # Model evaluation CLI
│   └── test_drift_day1.py      # Drift detection integration test
├── src/
│   ├── data_ingestion/
│   │   ├── weather_client.py   # Open-Meteo API client
│   │   ├── validator.py        # Data quality validation
│   │   ├── transformer.py      # Data transformation
│   │   ├── storage.py          # PostgreSQL persistence
│   │   └── pipeline.py         # Pipeline orchestration
│   ├── drift_detection/
│   │   ├── statistical_tests.py  # PSI, KS, JS divergence, Chi-square, Wasserstein
│   │   ├── reference_manager.py  # Reference data storage and management
│   │   ├── detector.py           # DriftDetector orchestrator
│   │   └── mlflow_logger.py      # MLflow integration for drift tracking
│   ├── mlflow_utils/
│   │   ├── tracking.py         # Experiment tracking
│   │   └── registry.py         # Model registry management
│   ├── training/
│   │   ├── feature_engineering.py  # Feature creation
│   │   ├── models.py           # Model implementations
│   │   ├── evaluation.py       # Metrics and evaluation
│   │   └── trainer.py          # Training orchestration
│   └── main.py                 # Application entry point
├── tests/
│   ├── integration/            # End-to-end tests
│   ├── test_weather_client.py
│   ├── test_validator.py
│   ├── test_feature_engineering.py
│   ├── test_statistical_tests.py      # Drift detection tests
│   ├── test_reference_manager.py      # Reference manager tests
│   ├── test_detector.py               # DriftDetector tests
│   └── test_drift_mlflow_logger.py    # MLflow logger tests
├── .gitignore                  # .gitignore
├── .dockerignore               # .dockerignore
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies
└── .env.example                # Environment template
```

---

## License

This project is for demonstration and educational purposes.
