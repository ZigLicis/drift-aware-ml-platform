"""Data storage module for weather data persistence.

This module provides database operations for storing and retrieving
weather data using SQLAlchemy.

Example:
    >>> from src.data_ingestion.storage import DataStorage
    >>> storage = DataStorage("postgresql://user:pass@localhost:5432/db")
    >>> count = storage.store_weather_data(df, batch_id="batch-001")
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator
from uuid import uuid4

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class DataStorageError(Exception):
    """Base exception for data storage errors."""

    pass


class DataStorage:
    """Database storage manager for weather data.

    Provides methods for storing, retrieving, and managing weather data
    in PostgreSQL using SQLAlchemy.

    Attributes:
        engine: SQLAlchemy database engine.
        schema: Database schema name.

    Example:
        >>> storage = DataStorage(
        ...     connection_string="postgresql://user:pass@localhost:5432/db"
        ... )
        >>> with storage.get_session() as session:
        ...     count = storage.store_weather_data(df, "batch-001")
        >>> print(f"Stored {count} records")
    """

    def __init__(
        self,
        connection_string: str,
        schema: str = "dsml",
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ) -> None:
        """Initialize database connection.

        Args:
            connection_string: PostgreSQL connection string.
            schema: Database schema name.
            pool_size: Connection pool size.
            max_overflow: Max connections above pool_size.
            echo: Whether to log SQL statements.

        Raises:
            DataStorageError: If connection fails.
        """
        self.connection_string = connection_string
        self.schema = schema

        try:
            self.engine: Engine = create_engine(
                connection_string,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,  # Verify connections before use
                echo=echo,
            )
            self._session_factory = sessionmaker(bind=self.engine)
            self._metadata = MetaData(schema=schema)

            # Define table reference
            self._weather_table = self._define_weather_table()

            logger.info(f"Initialized DataStorage with schema '{schema}'")

        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise DataStorageError(f"Database connection failed: {e}") from e

    def _define_weather_table(self) -> Table:
        """Define weather_data table schema."""
        return Table(
            "weather_data",
            self._metadata,
            Column("id", UUID, primary_key=True, default=uuid4),
            Column("location_name", String(255), nullable=False),
            Column("latitude", Numeric(9, 6), nullable=False),
            Column("longitude", Numeric(9, 6), nullable=False),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("temperature_2m", Numeric(5, 2)),
            Column("relative_humidity_2m", Numeric(5, 2)),
            Column("precipitation", Numeric(8, 2)),
            Column("wind_speed_10m", Numeric(6, 2)),
            Column("wind_direction_10m", Numeric(5, 2)),
            Column("surface_pressure", Numeric(7, 2)),
            Column("cloud_cover", Numeric(5, 2)),
            Column("raw_data", JSONB),
            Column("created_at", DateTime(timezone=True), default=func.now()),
            Column("updated_at", DateTime(timezone=True), default=func.now()),
            # Extended columns for ingestion metadata
            Column("batch_id", String(36)),
            Column("data_source", String(100)),
            Column("ingestion_timestamp", DateTime(timezone=True)),
            extend_existing=True,
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup.

        Yields:
            SQLAlchemy session.

        Example:
            >>> with storage.get_session() as session:
            ...     result = session.execute(query)
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Session error, rolling back: {e}")
            raise
        finally:
            session.close()

    def store_weather_data(
        self,
        df: pd.DataFrame,
        batch_id: str | None = None,
        upsert: bool = True,
    ) -> int:
        """Store weather data in the database.

        Performs bulk insert with optional upsert (update on conflict)
        for handling duplicate timestamps.

        Args:
            df: DataFrame with transformed weather data.
            batch_id: Batch identifier for this ingestion.
            upsert: If True, update existing records on conflict.

        Returns:
            Number of records inserted/updated.

        Raises:
            DataStorageError: If storage operation fails.

        Example:
            >>> count = storage.store_weather_data(df, batch_id="batch-001")
            >>> print(f"Stored {count} records")
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to store")
            return 0

        batch_id = batch_id or str(uuid4())
        records_stored = 0

        # Prepare data for insertion
        df = df.copy()
        if "batch_id" not in df.columns:
            df["batch_id"] = batch_id

        # Generate UUIDs for records without them
        if "id" not in df.columns:
            df["id"] = [str(uuid4()) for _ in range(len(df))]

        # Map columns to database schema
        db_columns = [
            "id",
            "location_name",
            "latitude",
            "longitude",
            "timestamp",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "raw_data",
            "batch_id",
            "data_source",
            "ingestion_timestamp",
            "created_at",
            "updated_at",
        ]

        # Keep only columns that exist in both df and db_columns
        available_columns = [c for c in db_columns if c in df.columns]
        df_to_store = df[available_columns]

        logger.info(f"Storing {len(df_to_store)} records (batch: {batch_id[:8]}...)")

        try:
            if upsert:
                records_stored = self._upsert_records(df_to_store)
            else:
                records_stored = self._insert_records(df_to_store)

            logger.info(f"Successfully stored {records_stored} records")
            return records_stored

        except SQLAlchemyError as e:
            logger.error(f"Failed to store weather data: {e}")
            raise DataStorageError(f"Storage operation failed: {e}") from e

    def _insert_records(self, df: pd.DataFrame) -> int:
        """Insert records without upsert logic."""
        try:
            with self.engine.connect() as conn:
                df.to_sql(
                    "weather_data",
                    conn,
                    schema=self.schema,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=1000,
                )
                conn.commit()
            return len(df)
        except IntegrityError as e:
            logger.warning(f"Duplicate records detected: {e}")
            raise

    def _upsert_records(self, df: pd.DataFrame) -> int:
        """Upsert records using PostgreSQL ON CONFLICT."""
        records = df.to_dict(orient="records")
        upserted_count = 0

        # Process in batches
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            with self.get_session() as session:
                for record in batch:
                    # Convert raw_data to proper JSON if it's a dict
                    if "raw_data" in record and isinstance(record["raw_data"], dict):
                        pass  # Keep as dict for JSONB

                    # Build upsert query
                    upsert_sql = text(f"""
                        INSERT INTO {self.schema}.weather_data (
                            id, location_name, latitude, longitude, timestamp,
                            temperature_2m, relative_humidity_2m, precipitation,
                            wind_speed_10m, surface_pressure, raw_data,
                            batch_id, data_source, ingestion_timestamp,
                            created_at, updated_at
                        ) VALUES (
                            :id, :location_name, :latitude, :longitude, :timestamp,
                            :temperature_2m, :relative_humidity_2m, :precipitation,
                            :wind_speed_10m, :surface_pressure, :raw_data,
                            :batch_id, :data_source, :ingestion_timestamp,
                            :created_at, :updated_at
                        )
                        ON CONFLICT (latitude, longitude, timestamp)
                        DO UPDATE SET
                            temperature_2m = EXCLUDED.temperature_2m,
                            relative_humidity_2m = EXCLUDED.relative_humidity_2m,
                            precipitation = EXCLUDED.precipitation,
                            wind_speed_10m = EXCLUDED.wind_speed_10m,
                            surface_pressure = EXCLUDED.surface_pressure,
                            raw_data = EXCLUDED.raw_data,
                            updated_at = EXCLUDED.updated_at
                    """)

                    # Prepare parameters with defaults for missing columns
                    params = {
                        "id": record.get("id", str(uuid4())),
                        "location_name": record.get("location_name", "Unknown"),
                        "latitude": record.get("latitude"),
                        "longitude": record.get("longitude"),
                        "timestamp": record.get("timestamp"),
                        "temperature_2m": record.get("temperature_2m"),
                        "relative_humidity_2m": record.get("relative_humidity_2m"),
                        "precipitation": record.get("precipitation"),
                        "wind_speed_10m": record.get("wind_speed_10m"),
                        "surface_pressure": record.get("surface_pressure"),
                        "raw_data": (
                            json.dumps(record.get("raw_data"))
                            if record.get("raw_data")
                            else None
                        ),
                        "batch_id": record.get("batch_id"),
                        "data_source": record.get("data_source"),
                        "ingestion_timestamp": record.get("ingestion_timestamp"),
                        "created_at": record.get("created_at", datetime.now(timezone.utc)),
                        "updated_at": record.get("updated_at", datetime.now(timezone.utc)),
                    }

                    session.execute(upsert_sql, params)
                    upserted_count += 1

        return upserted_count

    def get_latest_timestamp(self) -> datetime | None:
        """Get the most recent timestamp in the database.

        Used to determine the starting point for incremental ingestion.

        Returns:
            Latest timestamp or None if table is empty.

        Example:
            >>> latest = storage.get_latest_timestamp()
            >>> if latest:
            ...     print(f"Latest data: {latest}")
        """
        query = text(f"""
            SELECT MAX(timestamp) as max_ts
            FROM {self.schema}.weather_data
        """)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
                if result and result[0]:
                    logger.debug(f"Latest timestamp: {result[0]}")
                    return result[0]
                return None

        except SQLAlchemyError as e:
            logger.error(f"Failed to get latest timestamp: {e}")
            raise DataStorageError(f"Query failed: {e}") from e

    def get_data_by_timerange(
        self,
        start: datetime,
        end: datetime,
        location_name: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve weather data for a time range.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            location_name: Optional filter by location.

        Returns:
            DataFrame with weather data.

        Example:
            >>> from datetime import datetime, timedelta
            >>> end = datetime.now()
            >>> start = end - timedelta(days=7)
            >>> df = storage.get_data_by_timerange(start, end)
        """
        query = f"""
            SELECT *
            FROM {self.schema}.weather_data
            WHERE timestamp >= :start_time
              AND timestamp <= :end_time
        """

        params: dict[str, Any] = {
            "start_time": start,
            "end_time": end,
        }

        if location_name:
            query += " AND location_name = :location_name"
            params["location_name"] = location_name

        query += " ORDER BY timestamp ASC"

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)

            logger.info(
                f"Retrieved {len(df)} records for "
                f"{start.isoformat()} to {end.isoformat()}"
            )
            return df

        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve data: {e}")
            raise DataStorageError(f"Query failed: {e}") from e

    def get_record_count(self, location_name: str | None = None) -> int:
        """Get total record count in the database.

        Args:
            location_name: Optional filter by location.

        Returns:
            Number of records.
        """
        query = f"SELECT COUNT(*) FROM {self.schema}.weather_data"
        params: dict[str, Any] = {}

        if location_name:
            query += " WHERE location_name = :location_name"
            params["location_name"] = location_name

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params).fetchone()
                return result[0] if result else 0

        except SQLAlchemyError as e:
            logger.error(f"Failed to get record count: {e}")
            raise DataStorageError(f"Query failed: {e}") from e

    def log_ingestion_metadata(
        self,
        batch_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Log metadata about an ingestion run.

        Stores information about the ingestion batch for monitoring
        and debugging purposes.

        Args:
            batch_id: Unique batch identifier.
            metadata: Dictionary with ingestion details:
                - start_time: When ingestion started
                - end_time: When ingestion completed
                - records_count: Number of records processed
                - quality_score: Data quality score
                - status: success/failed
                - error_message: Error details if failed

        Example:
            >>> storage.log_ingestion_metadata(
            ...     batch_id="batch-001",
            ...     metadata={
            ...         "start_time": start,
            ...         "end_time": end,
            ...         "records_count": 168,
            ...         "quality_score": 0.95,
            ...         "status": "success"
            ...     }
            ... )
        """
        # Create ingestion_log table if it doesn't exist
        create_table_sql = text(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.ingestion_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                batch_id VARCHAR(36) NOT NULL,
                start_time TIMESTAMPTZ,
                end_time TIMESTAMPTZ,
                records_count INTEGER,
                quality_score NUMERIC(5, 4),
                status VARCHAR(50),
                error_message TEXT,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        insert_sql = text(f"""
            INSERT INTO {self.schema}.ingestion_log (
                batch_id, start_time, end_time, records_count,
                quality_score, status, error_message, metadata
            ) VALUES (
                :batch_id, :start_time, :end_time, :records_count,
                :quality_score, :status, :error_message, :metadata
            )
        """)

        try:
            with self.engine.connect() as conn:
                # Ensure table exists
                conn.execute(create_table_sql)
                conn.commit()

                # Insert log entry
                params = {
                    "batch_id": batch_id,
                    "start_time": metadata.get("start_time"),
                    "end_time": metadata.get("end_time"),
                    "records_count": metadata.get("records_count"),
                    "quality_score": metadata.get("quality_score"),
                    "status": metadata.get("status", "unknown"),
                    "error_message": metadata.get("error_message"),
                    "metadata": json.dumps(metadata, default=str),
                }
                conn.execute(insert_sql, params)
                conn.commit()

            logger.info(f"Logged ingestion metadata for batch {batch_id}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to log ingestion metadata: {e}")
            # Don't raise - logging failure shouldn't break ingestion

    def health_check(self) -> bool:
        """Check database connectivity.

        Returns:
            True if database is accessible.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("Database health check passed")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def close(self) -> None:
        """Close database connections."""
        self.engine.dispose()
        logger.info("Closed database connections")
