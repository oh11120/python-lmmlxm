from __future__ import annotations

import logging
from pathlib import Path


def get_backend_logger(log_file: str = "system/storage/tasks/backend.log", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("backend")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(str(path), encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
