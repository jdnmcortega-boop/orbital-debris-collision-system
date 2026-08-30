"""
Shared logging setup. Writes to both console and results/pipeline.log.

Usage in any module:
    from modules.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
from pathlib import Path


_CONFIGURED = False


def get_logger(name="pipeline", log_dir=None):
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not _CONFIGURED:
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent.parent / "results"
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_dir / "pipeline.log")
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        _CONFIGURED = True

    return logger