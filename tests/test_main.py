"""Tests for main module helpers."""

import pandas as pd
import numpy as np
import pytest

from src.main import _build_market_context, handle_shutdown


def _make_market_data():
    """Create minimal market data dict for testing."""
    n = 30
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 3600000,
        "open": [50000 + i * 10 for i in range(n)],
        "high": [50100 + i * 10 for i in range(n)],
        "low": [49900 + i * 10 for i in range(n)],
        "close": [50050 + i * 10 for i in range(n)],
        "volume": [1000.0] * n,
    }, index=timestamps)
    df.index.name = "datetime"
    return {"BTC/USDC:USDC": {"1h": df}}


def test_build_market_context():
    market_data = _make_market_data()
    prices = {"BTC/USDC:USDC": 50300.0}

    context = _build_market_context(market_data, prices)

    assert "BTC/USDC:USDC" in context
    btc = context["BTC/USDC:USDC"]
    assert "price" in btc
    assert "change_24h_pct" in btc
    assert "high_24h" in btc
    assert "low_24h" in btc
    assert "volume_24h" in btc
    assert btc["price"] == 50300.0


def test_build_market_context_insufficient_data():
    """Should skip pairs with less than 24 candles."""
    n = 10
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 3600000,
        "open": [100] * n, "high": [110] * n, "low": [90] * n,
        "close": [105] * n, "volume": [500] * n,
    }, index=timestamps)
    df.index.name = "datetime"

    context = _build_market_context(
        {"ETH/USDC:USDC": {"1h": df}}, {"ETH/USDC:USDC": 105.0}
    )
    assert "ETH/USDC:USDC" not in context  # Skipped — too few candles


def test_build_market_context_no_price():
    """Should skip pairs without a current price."""
    market_data = _make_market_data()
    context = _build_market_context(market_data, {})  # No prices
    assert "BTC/USDC:USDC" not in context


def test_handle_shutdown():
    import src.main as main_mod
    original = main_mod._shutdown
    try:
        main_mod._shutdown = False
        handle_shutdown(None, None)
        assert main_mod._shutdown is True
    finally:
        main_mod._shutdown = original
