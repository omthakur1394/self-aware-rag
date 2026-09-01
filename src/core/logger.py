import logging
import sys


def setup_logger(name: str = "self_rag") -> logging.Logger:
    """Configures and returns a structured application logger."""
    logger_instance = logging.getLogger(name)
    if not logger_instance.handlers:
        logger_instance.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
    return logger_instance


logger = setup_logger()
