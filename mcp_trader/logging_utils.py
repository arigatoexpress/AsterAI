import logging
import sys


def setup_logger(name: str | None = None, level: str = "INFO") -> logging.Logger:
    """Set up a configured logger with proper formatting and level."""
    logger = logging.getLogger(name if name else __name__)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
