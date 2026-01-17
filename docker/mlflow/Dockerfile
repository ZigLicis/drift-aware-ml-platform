FROM python:3.10-slim

LABEL maintainer="Domain-Shift ML Platform"
LABEL description="MLflow tracking server with PostgreSQL backend"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create mlflow user for security
RUN useradd -m -s /bin/bash mlflow

# Set working directory
WORKDIR /mlflow

# Install MLflow and PostgreSQL driver
RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow

# Switch to non-root user
USER mlflow

# Expose MLflow port
EXPOSE 5000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start MLflow server (using shell form for env var expansion)
CMD mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000
