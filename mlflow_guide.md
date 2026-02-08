## MLflow Guide

### Accessing the UI

After starting services with `docker-compose up -d`, open:

**http://localhost:5001**

### Experiments Overview

The platform uses two MLflow experiments:

| Experiment | Purpose | Key Metrics |
|------------|---------|-------------|
| `weather-prediction` | Model training runs | RMSE, MAE, R², MAPE |
| `drift-monitoring` | Drift detection runs | PSI, KS-stat, drift_score |

### Viewing Model Training Runs

1. Navigate to **Experiments** → **weather-prediction**
2. Compare runs by selecting checkboxes and clicking **Compare**
3. View **Parameters**: model type, hyperparameters, feature count
4. View **Metrics**: training/validation/test performance
5. View **Artifacts**: saved model files, feature importance plots

### Viewing Drift Monitoring Runs

1. Navigate to **Experiments** → **drift-monitoring**
2. Each run represents one drift check
3. View **Metrics**:
   - `overall_drift_score`: 0-1 severity score
   - `{feature}_psi`: Per-feature PSI values
   - `{feature}_ks_statistic`: Per-feature KS statistics
4. View **Artifacts**:
   - `drift_report.json`: Full report data
   - `drift_heatmap.png`: Visual severity map
   - `recommendations.txt`: Actionable recommendations

### Model Registry

1. Navigate to **Models** in the top menu
2. View registered model: `weather-forecaster`
3. See version history with metrics
4. **Stage transitions**: None → Staging → Production → Archived
5. Click version to see linked training run

### What to Look For

**Healthy Training:**
- Test metrics similar to validation (no overfitting)
- Consistent performance across runs
- R² > 0 (model is learning)

**Drift Alerts:**
- `overall_drift_score` trending upward
- Specific features with high PSI values
- Recommendations indicating action needed