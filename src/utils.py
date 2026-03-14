"""Logging setup and helpers."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with rich console + rotating file output."""
    logger = logging.getLogger("agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        # Console handler (rich)
        ch = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        # File handler (rotating — max 10MB, keep 5 backups)
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            _LOG_DIR / "agent.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def now_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def now_ts() -> int:
    """Current UTC timestamp in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


log = setup_logging()
