-- Initialize databases for Domain-Shift ML Platform

-- Create MLflow database
CREATE DATABASE mlflow_db;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO dsml_user;

-- Connect to main database and create extensions
\c dsml_db

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schema for domain shift platform
CREATE SCHEMA IF NOT EXISTS dsml;

-- Table for storing weather data
CREATE TABLE IF NOT EXISTS dsml.weather_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location_name VARCHAR(255) NOT NULL,
    latitude DECIMAL(9, 6) NOT NULL,
    longitude DECIMAL(9, 6) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    temperature_2m DECIMAL(5, 2),
    relative_humidity_2m DECIMAL(5, 2),
    precipitation DECIMAL(8, 2),
    wind_speed_10m DECIMAL(6, 2),
    wind_direction_10m DECIMAL(5, 2),
    surface_pressure DECIMAL(7, 2),
    cloud_cover DECIMAL(5, 2),
    raw_data JSONB,
    batch_id VARCHAR(36),
    data_source VARCHAR(100),
    ingestion_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(latitude, longitude, timestamp)
);

-- Table for tracking domain shift metrics
CREATE TABLE IF NOT EXISTS dsml.domain_shift_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 6) NOT NULL,
    threshold_value DECIMAL(10, 6),
    is_shift_detected BOOLEAN DEFAULT FALSE,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for model retraining history
CREATE TABLE IF NOT EXISTS dsml.retraining_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    previous_version VARCHAR(50),
    new_version VARCHAR(50) NOT NULL,
    trigger_reason VARCHAR(255) NOT NULL,
    mlflow_run_id VARCHAR(255),
    training_started_at TIMESTAMPTZ NOT NULL,
    training_completed_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'running',
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX idx_weather_data_location ON dsml.weather_data(latitude, longitude);
CREATE INDEX idx_weather_data_timestamp ON dsml.weather_data(timestamp DESC);
CREATE INDEX idx_weather_data_location_time ON dsml.weather_data(location_name, timestamp DESC);

CREATE INDEX idx_domain_shift_model ON dsml.domain_shift_metrics(model_name, model_version);
CREATE INDEX idx_domain_shift_time ON dsml.domain_shift_metrics(window_end DESC);
CREATE INDEX idx_domain_shift_detected ON dsml.domain_shift_metrics(is_shift_detected) WHERE is_shift_detected = TRUE;

CREATE INDEX idx_retraining_model ON dsml.retraining_history(model_name);
CREATE INDEX idx_retraining_status ON dsml.retraining_history(status);

-- Grant schema permissions
GRANT USAGE ON SCHEMA dsml TO dsml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dsml TO dsml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA dsml TO dsml_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dsml GRANT ALL PRIVILEGES ON TABLES TO dsml_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA dsml GRANT ALL PRIVILEGES ON SEQUENCES TO dsml_user;

-- Connect to MLflow database and set up
\c mlflow_db

-- Grant full access for MLflow
GRANT ALL ON SCHEMA public TO dsml_user;
