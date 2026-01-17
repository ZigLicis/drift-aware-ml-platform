"""Main entry point for Domain-Shift ML Platform."""

import logging
import sys

import structlog


def configure_logging() -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", level=logging.INFO)


def main() -> None:
    """Run the Domain-Shift ML Platform."""
    configure_logging()
    logger = structlog.get_logger(__name__)

    logger.info("Starting Domain-Shift ML Platform", version="0.1.0")
    logger.info("Platform initialized successfully")
    logger.info("Ready to ingest data and train models")

    # Placeholder for main application loop
    # This will be expanded in subsequent sprints
    logger.info("Shutting down...")


if __name__ == "__main__":
    main()
