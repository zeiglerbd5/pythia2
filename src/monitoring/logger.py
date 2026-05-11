"""
Structured logging setup for Pythia using loguru.

Provides JSON-formatted logs with rotation and retention policies.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    rotation: str = "1 day",
    retention: str = "30 days",
    structured: bool = True,
    enable_console: bool = True
):
    """
    Setup loguru logging with structured JSON output.

    Args:
        log_file: Path to log file (default: logs/pythia.log)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Log rotation policy (e.g., "1 day", "100 MB")
        retention: Log retention policy (e.g., "30 days")
        structured: If True, use JSON format for file logs
        enable_console: If True, also log to console
    """
    # Remove default handler
    logger.remove()

    # Console handler (human-readable)
    if enable_console:
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )

    # File handler (JSON structured for production)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if structured:
            # JSON format for structured logging
            # Note: backtrace/diagnose disabled to prevent recursive error message explosion
            logger.add(
                log_file,
                level=log_level,
                rotation="100 MB",  # Size-based rotation to prevent bloat
                retention=retention,
                compression="zip",
                serialize=True,  # JSON format
                backtrace=False,
                diagnose=False
            )
        else:
            # Human-readable format
            logger.add(
                log_file,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                backtrace=True,
                diagnose=True
            )

    logger.info(
        f"Logging initialized",
        extra={
            "log_file": log_file,
            "log_level": log_level,
            "structured": structured
        }
    )


def get_logger(name: str):
    """
    Get a logger instance with a specific name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(logger_name=name)


if __name__ == "__main__":
    # Test logging setup
    setup_logging(
        log_file="logs/test_pythia.log",
        log_level="DEBUG",
        structured=True
    )

    logger.info("Test info message")
    logger.debug("Test debug message", extra={"test_key": "test_value"})
    logger.warning("Test warning message")
    logger.error("Test error message")

    try:
        1 / 0
    except Exception as e:
        logger.exception("Test exception logging")

    logger.success("Logging test completed!")
