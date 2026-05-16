import logging
import sys


def setup_logging(debug: bool = False):
    """Configure structured logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)

    # Silence noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
