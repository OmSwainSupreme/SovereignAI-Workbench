"""Logging configuration for SovereignAI Workbench."""
from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
    )
