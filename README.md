# Domain-Shift ML Platform

**An end-to-end machine learning platform that detects data distribution shifts in production and enables proactive model maintenance.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MLflow](https://img.shields.io/badge/mlflow-2.10-orange)
![PostgreSQL](https://img.shields.io/badge/postgresql-15-blue)

## Overview

Machine learning models silently degrade in production when the data they encounter differs from their training data. This phenomenon, known as **domain shift** or **data drift**, is one of the leading causes of ML system failures—yet it often goes undetected until significant business impact occurs.

This platform provides a complete solution for weather prediction that includes automated drift detection. It ingests weather data from the Open-Meteo API, trains regression models to predict next-day maximum temperature, and continuously monitors for distribution shifts using statistical tests like Population Stability Index (PSI) and Kolmogorov-Smirnov tests.

When drift is detected, the system provides severity classifications (Low → Moderate → Significant → Severe) along with actionable recommendations. All metrics are logged to MLflow for visualization and historical analysis, creating a complete audit trail of data quality over time.

## Why This Matters

### The Domain Shift Problem

Models trained on historical data assume the future will look like the past. When this assumption breaks, model performance degrades—often catastrophically.

Consider these real-world scenarios:
- A fraud model trained on pre-pandemic data fails when consumer behavior shifts
- A demand forecasting model breaks during supply chain disruptions
- A weather model trained on summer data produces nonsense predictions in winter

### What I Found

During development of this platform, I trained a temperature prediction model on 6 months of summer and fall data (June–November). The model achieved impressive metrics:

| Metric | Training Performance |
|--------|---------------------|
| R² Score | 0.95 |
| RMSE | 1.2°C |

Then I tested it on winter data (December–January):

| Metric | Winter Performance |
|--------|-------------------|
| R² Score | **-62** |
| RMSE | 28°C |

The model wasn't just wrong—it was worse than predicting the mean. This catastrophic failure occurred because the temperature distribution shifted dramatically between seasons, and the model had no mechanism to detect this.

### Why Automated Detection is Valuable

Manual monitoring doesn't scale. With dozens of features and models, drift can happen in subtle ways. This platform:

1. **Detects drift automatically** using proven statistical methods
2. **Quantifies severity** so you can prioritize responses
3. **Provides recommendations** based on drift patterns
4. **Logs everything** for audit trails and trend analysis
5. **Scales** to any number of features and models

## Data Flow

1. **Ingestion**: Weather Client fetches hourly data from Open-Meteo API (forecast for recent data, archive for historical)
2. **Validation**: Data Validator scores quality (completeness, range checks, anomaly detection)
3. **Transformation**: Data Transformer adds metadata, maps schema, engineers basic features
4. **Storage**: Data Storage upserts to PostgreSQL with conflict resolution
5. **Training**: Feature Engineer creates 30 features, Model Trainer fits and evaluates
6. **Drift Detection**: Reference Manager stores baselines, Drift Detector compares distributions
7. **Tracking**: All experiments, metrics, and artifacts logged to MLflow

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/domain-shift-ml-platform.git
cd domain-shift-ml-platform
cp .env.example .env

# Start services
docker-compose up -d

# Verify services are running and healthy
docker-compose ps

# Run pipeline (in container)
docker-compose exec app python scripts/run_ingestion.py \
    --mode historical --start 2024-01-01 --end 2024-12-31

docker-compose exec app python scripts/train_model.py --model ridge

docker-compose exec app python scripts/run_drift_check.py create-reference \
    --name baseline_v1 --start 2024-01-01 --end 2024-06-30

docker-compose exec app python scripts/run_drift_check.py check \
    --reference baseline_v1 --window-hours 168
```

### Note: One-off Container (for scripts that exit)

```bash
# Use 'run' instead of 'exec' if the app container isn't finished/running
docker-compose run --rm app python scripts/run_ingestion.py \
    --mode historical --start 2024-01-01 --end 2024-12-31
```

**Service Endpoints:**
| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 5432 | `localhost:5432` |
| MLflow UI | 5001 | http://localhost:5001 |

## Architecture

```
Open-Meteo API → Weather Client → Validator → Transformer → PostgreSQL
                                                                 ↓
MLflow ← Drift Detector ← Reference Manager ← Feature Engineer ← Query
```

### Tech Stack

| Component | Library/Framework | Purpose |
|-----------|------------|---------|
| Language | Python 3.10+ | Core development |
| Database | PostgreSQL 15 | Data storage |
| ML Tracking | MLflow 2.10 | Experiment tracking, model registry |
| ML Framework | scikit-learn 1.4 | Model training |
| Data Processing | pandas 2.1, NumPy 1.26 | Data manipulation |
| Statistics | SciPy 1.12 | Statistical tests |
| HTTP Client | requests + tenacity | API calls with retry |
| ORM | SQLAlchemy 2.0 | Database abstraction |
| Validation | Pydantic 2.5 | Data validation |
| Container | Docker Compose | Service orchestration |
| Logging | structlog | Structured logging |

---

## CLI Commands

### Data Ingestion

```bash
# Historical data
python scripts/run_ingestion.py --mode historical --start 2024-01-01 --end 2024-12-31

# Last 24 hours
python scripts/run_ingestion.py --mode incremental

# Backfill in batches
python scripts/run_ingestion.py --mode backfill --start 2020-01-01 --end 2024-12-31 --batch-days 30
```

### Model Training

```bash
# Train Ridge baseline
python scripts/train_model.py --model ridge

# Train with date range
python scripts/train_model.py --model ridge --start 2024-01-01 --end 2024-12-31

# Train and promote
python scripts/train_model.py --model ridge --promote staging
```

### Drift Detection

```bash
# Create reference baseline
python scripts/run_drift_check.py create-reference \
    --name baseline_v1 --start 2024-01-01 --end 2024-06-30

# Check last 7 days
python scripts/run_drift_check.py check --reference baseline_v1 --window-hours 168

# Check date range
python scripts/run_drift_check.py check \
    --reference baseline_v1 --start 2024-11-01 --end 2024-12-31

# View history
python scripts/run_drift_check.py history --runs 10

# List references
python scripts/run_drift_check.py list-references
```


### Running in Docker

```bash
# Interactive shell
docker-compose exec app bash

# Single command
docker-compose exec app python scripts/run_drift_check.py check \
    --reference baseline_v1 --window-hours 168

# Run tests
docker-compose exec app pytest tests/ -v
```

## Drift Metrics

| Metric | What It Measures | Thresholds |
|--------|------------------|------------|
| **PSI** | Distribution shift magnitude | < 0.1 stable, 0.1-0.2 moderate, > 0.2 action needed |
| **KS-test** | Max CDF difference | p-value < 0.05 = significant |
| **JS Divergence** | Symmetric distribution similarity | 0-1 scale, > 0.2 = significant |
| **Wasserstein** | "Earth mover" distance | Scale-dependent |

### Severity Levels

| Level | PSI Range | Action |
|-------|-----------|--------|
| None/Low | < 0.1 | Continue monitoring |
| Moderate | 0.1 - 0.15 | Investigate |
| Significant | 0.15 - 0.2 | Plan remediation |
| Severe | > 0.5 | Immediate retraining |

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Database username | `dsml_user` |
| `POSTGRES_PASSWORD` | Database password | `dsml_password_change_me` |
| `POSTGRES_DB` | Database name | `dsml_db` |
| `POSTGRES_PORT` | Database port | `5432` |
| `DATABASE_URL` | Full connection string | `postgresql://dsml_user:...@localhost:5432/dsml_db` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5001` |
| `MLFLOW_PORT` | MLflow UI port | `5001` |
| `APP_PORT` | Application port | `8000` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `DRIFT_THRESHOLD` | Default drift threshold | `0.1` |

### Config Files

| File | Purpose |
|------|---------|
| `config/data_config.yaml` | API settings, location, features |
| `config/model_config.yaml` | Model params, training settings |
| `config/drift_config.yaml` | Drift thresholds, monitoring |


## Project Structure

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
│   ├── run_drift_check.py      # Drift detection CLI
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

## Key Components

### Data Ingestion (`src/data_ingestion/`)
- `weather_client.py` — Open-Meteo API with retry logic
- `validator.py` — Quality scoring, range validation
- `transformer.py` — Feature engineering
- `storage.py` — PostgreSQL with upsert

### Training (`src/training/`)
- `feature_engineering.py` — 30 features (temporal, lag, rolling)
- `models.py` — Ridge, Random Forest, Gradient Boosting
- `trainer.py` — Training with MLflow tracking

### Drift Detection (`src/drift_detection/`)
- `statistical_tests.py` — PSI, KS, JS divergence, Wasserstein
- `reference_manager.py` — Store/load reference profiles
- `detector.py` — Orchestrator with severity classification
- `mlflow_logger.py` — Log drift reports to MLflow

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `DATABASE_URL not set` | Check `.env` file exists and has `DATABASE_URL` |
| `Connection refused :5432` | Run `docker-compose up -d postgres` |
| `MLflow connection failed` | Run `docker-compose up -d mlflow` |
| `No data found` | Run ingestion first |
| `Reference not found` | Create reference with `create-reference` command |

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

## Development

```bash
# Tests
pytest tests/ -v
pytest tests/ --cov=src

# Code quality
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
mypy src/
```

## Why This Exists

A model trained on summer data (R² = 0.95) and tested on winter data (R² = **-62**) failed catastrophically due to seasonal domain shift. This platform detects such shifts before they cause problems.

### Key Takeaways

1. **Representative training data is critical** — Models only know what they've seen
2. **Temporal splits prevent leakage** — Random splits for time series data are wrong
3. **Monitoring is not optional** — Production models need continuous validation
4. **Automated detection scales** — Manual review doesn't work for many features/models
5. **Severity matters** — Not all drift requires immediate action
