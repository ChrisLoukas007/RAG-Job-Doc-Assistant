# core/logging.py — cross-cutting logging setup
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import LOGS_DIR

def get_logger(name: str = "app"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)

    # Rotating file handler
    log_path = Path(LOGS_DIR) / "app.log"
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    logger.propagate = False
    return logger
# Now you can use get_logger() in other modules to get a configured logger
# Example:
# logger = get_logger(__name__)