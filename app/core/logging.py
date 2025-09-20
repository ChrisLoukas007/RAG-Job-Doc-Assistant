# core/logging.py — cross-cutting logging setup
import logging
from logging.handlers import RotatingFileHandler # for log file rotation
from pathlib import Path
from .config import LOGS_DIR

def get_logger(name: str = "app"):
    # Get a logger with the given name (default is "app")
    logger = logging.getLogger(name)
    logger = logging.getLogger(name)
    
    # If this logger is already set up, just return it
    if logger.handlers:
        return logger # already configured
    
    # Set the minimum level of messages to log (INFO means it logs INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.INFO)
    
    # === CONSOLE HANDLER ===
    # This makes log messages appear in the terminal/console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Only log INFO level and above to console
    # Set the format: shows timestamp, level (like ERROR), logger name, and the message
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)  # Add the console handler to our logger
    
    # === FILE HANDLER ===
    # This saves log messages to a file called "app.log"
    log_path = Path(LOGS_DIR) / "app.log"
    # RotatingFileHandler automatically creates new files when the current one gets too big
    # maxBytes=5_000_000 means when file reaches ~5MB, start a new one
    # backupCount=3 means keep 3 old files (app.log.1, app.log.2, app.log.3)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)  # Only log INFO level and above to file
    # Use the same format as console
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)  # Add the file handler to our logger
    
    # Don't pass messages up to parent loggers (prevents duplicate messages)
    logger.propagate = False
    
    # Return the fully configured logger
    return logger

# Now you can use get_logger() in other modules to get a configured logger
# Example:
# logger = get_logger(__name__)