"""Logging setup for the concert video editor."""

import logging
import os


def setup_logger(log_file: str = "logs/app.log") -> logging.Logger:
    """
    Create and configure a logger that writes to both file and console.

    Args:
        log_file: Path to the log file. Parent directory is created if needed.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("ConcertVideoEditor")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
