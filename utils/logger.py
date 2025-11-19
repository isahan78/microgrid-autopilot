"""
Centralized logging configuration for Microgrid Autopilot.

Provides consistent logging across all modules with file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Parameters:
    -----------
    name : str
        Logger name (typically __name__)
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
    --------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # File handler - detailed logs
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - important messages only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create logger."""
    return setup_logger(name)
