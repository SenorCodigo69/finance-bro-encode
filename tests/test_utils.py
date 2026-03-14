"""Tests for utility functions."""

import re
from datetime import datetime, timezone

from src.utils import now_iso, now_ts, setup_logging


def test_now_iso_format():
    result = now_iso()
    # Should be a valid ISO 8601 string with timezone
    dt = datetime.fromisoformat(result)
    assert dt.tzinfo is not None


def test_now_ts_milliseconds():
    before = int(datetime.now(timezone.utc).timestamp() * 1000)
    result = now_ts()
    after = int(datetime.now(timezone.utc).timestamp() * 1000)
    assert before <= result <= after


def test_setup_logging_returns_logger():
    logger = setup_logging("DEBUG")
    assert logger.name == "agent"
    assert logger.level == 10  # DEBUG


def test_setup_logging_idempotent():
    """Calling setup_logging multiple times should not add duplicate handlers."""
    logger1 = setup_logging()
    n_handlers = len(logger1.handlers)
    logger2 = setup_logging()
    assert len(logger2.handlers) == n_handlers
    assert logger1 is logger2
