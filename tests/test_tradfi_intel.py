"""Tests for TradFi intelligence layer — earnings, options, FRED, correlations."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.tradfi_intel import (
    CorrelationMatrix,
    EarningsCalendar,
    FREDClient,
    IVContext,
    OptionsIntel,
    TradFiIntel,
)


# ── EarningsCalendar ─────────────────────────────────────────────────────


def test_earnings_no_data_no_blackout():
    ec = EarningsCalendar()
    assert ec.is_earnings_blackout("XYZ-AAPL/USDC:USDC") is False


def test_earnings_blackout_active():
    ec = EarningsCalendar(hours_before=24, hours_after=2)
    # Set earnings to 12 hours from now
    future = datetime.now(timezone.utc) + timedelta(hours=12)
    ec._cache["XYZ-AAPL"] = future
    assert ec.is_earnings_blackout("XYZ-AAPL/USDC:USDC") is True


def test_earnings_blackout_past():
    ec = EarningsCalendar(hours_before=24, hours_after=2)
    # Earnings was 3 hours ago (past the 2h after window)
    past = datetime.now(timezone.utc) - timedelta(hours=3)
    ec._cache["XYZ-AAPL"] = past
    assert ec.is_earnings_blackout("XYZ-AAPL/USDC:USDC") is False


def test_earnings_blackout_just_after():
    ec = EarningsCalendar(hours_before=24, hours_after=2)
    # Earnings was 1 hour ago (within the 2h after window)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    ec._cache["XYZ-AAPL"] = past
    assert ec.is_earnings_blackout("XYZ-AAPL/USDC:USDC") is True


def test_earnings_crypto_never_blocked():
    ec = EarningsCalendar()
    future = datetime.now(timezone.utc) + timedelta(hours=12)
    ec._cache["BTC"] = future  # Shouldn't happen, but just in case
    assert ec.is_earnings_blackout("BTC/USDC:USDC") is False


def test_earnings_invalid_pair():
    ec = EarningsCalendar()
    assert ec.is_earnings_blackout("INVALID") is False


def test_earnings_none_cached():
    ec = EarningsCalendar()
    ec._cache["XYZ-AAPL"] = None
    assert ec.is_earnings_blackout("XYZ-AAPL/USDC:USDC") is False


def test_get_upcoming_earnings():
    ec = EarningsCalendar()
    dt = datetime(2026, 4, 1, tzinfo=timezone.utc)
    ec._cache["XYZ-AAPL"] = dt
    ec._cache["XYZ-TSLA"] = None
    result = ec.get_upcoming_earnings()
    assert "XYZ-AAPL" in result
    assert result["XYZ-AAPL"] is not None
    assert result["XYZ-TSLA"] is None


# ── OptionsIntel ─────────────────────────────────────────────────────────


def test_iv_context_to_dict():
    ctx = IVContext(avg_iv=0.3456, put_call_ratio=1.234, calls_count=50, puts_count=60)
    d = ctx.to_dict()
    assert d["avg_iv"] == 0.3456
    assert d["put_call_ratio"] == 1.23
    assert d["calls_count"] == 50


@pytest.mark.asyncio
async def test_options_crypto_returns_none():
    oi = OptionsIntel()
    result = await oi.get_iv_context("BTC/USDC:USDC")
    assert result is None


@pytest.mark.asyncio
async def test_options_invalid_pair_returns_none():
    oi = OptionsIntel()
    result = await oi.get_iv_context("INVALID")
    assert result is None


# ── FREDClient ───────────────────────────────────────────────────────────


def test_fred_no_key_returns_empty():
    fred = FREDClient(api_key="")
    # Can't do async test here easily, but check it initializes
    assert fred._api_key == ""


@pytest.mark.asyncio
async def test_fred_no_key_snapshot_empty():
    fred = FREDClient(api_key="")
    result = await fred.get_macro_snapshot()
    assert result == {}


@pytest.mark.asyncio
async def test_fred_cached_returns_cache():
    fred = FREDClient(api_key="test")
    fred._cache = {"treasury_10y": {"value": 4.25, "date": "2026-03-01"}}
    import time
    fred._cache_time = time.time()  # Fresh cache

    result = await fred.get_macro_snapshot()
    assert result["treasury_10y"]["value"] == 4.25


def test_fred_yield_spread():
    fred = FREDClient(api_key="test")
    fred._cache = {
        "treasury_10y": {"value": 4.25, "date": "2026-03-01"},
        "treasury_2y": {"value": 3.75, "date": "2026-03-01"},
    }
    spread = fred.get_yield_spread()
    assert spread == 0.5


def test_fred_yield_spread_missing_data():
    fred = FREDClient(api_key="test")
    fred._cache = {}
    assert fred.get_yield_spread() is None


# ── CorrelationMatrix ────────────────────────────────────────────────────


def test_correlation_high():
    """Two assets that move together should have high correlation."""
    cm = CorrelationMatrix(threshold=0.85, window=20)
    prices = list(range(100, 120))
    data = {
        "A/USDC:USDC": pd.DataFrame({"close": prices}),
        "B/USDC:USDC": pd.DataFrame({"close": [p * 1.01 for p in prices]}),  # Nearly identical
    }
    matrix = cm.compute(data)
    assert abs(matrix["A/USDC:USDC"]["B/USDC:USDC"]) > 0.99


def test_correlation_independent():
    """Uncorrelated assets should have low correlation."""
    cm = CorrelationMatrix(threshold=0.85, window=20)
    np.random.seed(42)
    data = {
        "A/USDC:USDC": pd.DataFrame({"close": np.random.randn(20).cumsum() + 100}),
        "B/USDC:USDC": pd.DataFrame({"close": np.random.randn(20).cumsum() + 100}),
    }
    matrix = cm.compute(data)
    corr = abs(matrix["A/USDC:USDC"]["B/USDC:USDC"])
    assert corr < 0.85  # Should not be highly correlated


def test_get_correlated_pairs():
    cm = CorrelationMatrix(threshold=0.85, window=20)
    prices = list(range(100, 120))
    data = {
        "A/USDC:USDC": pd.DataFrame({"close": prices}),
        "B/USDC:USDC": pd.DataFrame({"close": [p * 1.5 for p in prices]}),
        "C/USDC:USDC": pd.DataFrame({"close": np.random.randn(20).cumsum() + 100}),
    }
    np.random.seed(42)
    cm.compute(data)

    correlated = cm.get_correlated_pairs("A/USDC:USDC")
    assert "B/USDC:USDC" in correlated


def test_correlation_insufficient_data():
    cm = CorrelationMatrix(threshold=0.85, window=20)
    data = {
        "A/USDC:USDC": pd.DataFrame({"close": [100, 101, 102]}),  # Only 3 points
    }
    matrix = cm.compute(data)
    assert matrix == {}


def test_correlation_empty_data():
    cm = CorrelationMatrix(threshold=0.85, window=20)
    matrix = cm.compute({})
    assert matrix == {}


def test_correlation_warnings():
    cm = CorrelationMatrix(threshold=0.85, window=20)
    prices = list(range(100, 120))
    data = {
        "A/USDC:USDC": pd.DataFrame({"close": prices}),
        "B/USDC:USDC": pd.DataFrame({"close": [p * 1.5 for p in prices]}),
    }
    cm.compute(data)
    warnings = cm.get_correlation_warnings(["A/USDC:USDC", "B/USDC:USDC"])
    assert len(warnings) == 1
    assert warnings[0]["pair_a"] == "A/USDC:USDC"
    assert warnings[0]["pair_b"] == "B/USDC:USDC"
    assert abs(warnings[0]["correlation"]) > 0.85


def test_correlation_no_warnings_when_uncorrelated():
    cm = CorrelationMatrix(threshold=0.85, window=20)
    cm._last_matrix = {
        "A/USDC:USDC": {"A/USDC:USDC": 1.0, "B/USDC:USDC": 0.3},
        "B/USDC:USDC": {"A/USDC:USDC": 0.3, "B/USDC:USDC": 1.0},
    }
    warnings = cm.get_correlation_warnings(["A/USDC:USDC", "B/USDC:USDC"])
    assert len(warnings) == 0


# ── TradFiIntel aggregator ──────────────────────────────────────────────


def test_tradfi_intel_init():
    ti = TradFiIntel(fred_api_key="", earnings_hours_before=24, earnings_hours_after=2)
    assert ti.earnings.hours_before == 24
    assert ti.earnings.hours_after == 2
    assert isinstance(ti.options, OptionsIntel)
    assert isinstance(ti.correlations, CorrelationMatrix)
