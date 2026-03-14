"""Tests for the data fetcher with DataSourceManager (native HL API)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.config import AgentConfig, DataSourcesConfig, ExchangeConfig
from src.data_fetcher import DataFetcher, _TF_MS
from src.database import Database
from src.exchange import Exchange


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def exchange():
    ex = Exchange(ExchangeConfig(name="hyperliquid"), mode="paper")
    return ex


@pytest.fixture
def config():
    return AgentConfig(pairs=["BTC/USDC:USDC"], timeframes=["1h"])


@pytest.fixture
def fetcher(exchange, db, config):
    return DataFetcher(exchange, db, config)


def _make_candles(n=100, start_price=50000.0):
    """Generate fake OHLCV candles."""
    candles = []
    for i in range(n):
        ts = 1700000000000 + i * 3600000
        p = start_price + i * 10
        candles.append([ts, p, p + 50, p - 50, p + 5, 1000.0 + i])
    return candles


@pytest.mark.asyncio
async def test_fetch_pair_success(fetcher, exchange):
    # Mock the source manager's primary source
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=_make_candles())

    df = await fetcher.fetch_pair("BTC/USDC:USDC", "1h")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "close" in df.columns
    assert "datetime" not in df.columns  # It's the index
    assert df.index.name == "datetime"


@pytest.mark.asyncio
async def test_fetch_pair_empty_raises(fetcher):
    """Empty candles from primary raises."""
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="Primary OHLCV failed"):
        await fetcher.fetch_pair("BTC/USDC:USDC", "1h")


@pytest.mark.asyncio
async def test_fetch_pair_caches_to_db(fetcher, db):
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=_make_candles(10))

    await fetcher.fetch_pair("BTC/USDC:USDC", "1h")

    # Check that candles were cached
    cached = db.get_cached_ohlcv("BTC/USDC:USDC", "1h", 0)
    assert len(cached) == 10


@pytest.mark.asyncio
async def test_fetch_pair_validates_data(fetcher):
    """Should drop invalid candles (negative prices, high < low)."""
    candles = _make_candles(5)
    candles.append([1700100000000, -1.0, 50, 40, 45, 100])  # Negative open
    candles.append([1700200000000, 100, 50, 60, 55, 100])    # High < low
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=candles)

    df = await fetcher.fetch_pair("BTC/USDC:USDC", "1h")
    assert len(df) == 5  # Only the valid candles


@pytest.mark.asyncio
async def test_fetch_all_pairs_updates_prices(fetcher):
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=_make_candles(10))
    fetcher._source_manager.fetch_validated_price = AsyncMock(
        return_value=MagicMock(
            source_prices={"hyperliquid": 50095.0},
            divergence_pct=0.0,
            price_valid=True,
            anomaly_flag=False,
            source_latencies={"hyperliquid": 50.0},
        )
    )

    await fetcher.fetch_all_pairs()

    prices = fetcher.get_latest_prices()
    assert "BTC/USDC:USDC" in prices
    assert prices["BTC/USDC:USDC"] > 0


@pytest.mark.asyncio
async def test_get_single_price(fetcher):
    """get_single_price uses native allMids via primary source."""
    fetcher._source_manager._primary.fetch_latest_price = AsyncMock(return_value=50000.0)

    price = await fetcher.get_single_price("BTC/USDC:USDC")
    assert price == 50000.0


@pytest.mark.asyncio
async def test_get_single_price_failure_returns_cached(fetcher):
    fetcher._latest_prices = {"BTC/USDC:USDC": 49000.0}
    fetcher._source_manager._primary.fetch_latest_price = AsyncMock(
        side_effect=Exception("timeout")
    )

    price = await fetcher.get_single_price("BTC/USDC:USDC")
    assert price == 49000.0


@pytest.mark.asyncio
async def test_all_sources_fail_raises(exchange, db):
    config = AgentConfig(pairs=["BTC/USDC:USDC"], timeframes=["1h"])
    fetcher = DataFetcher(exchange, db, config)
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="Primary OHLCV failed"):
        await fetcher.fetch_pair("BTC/USDC:USDC", "1h")


@pytest.mark.asyncio
async def test_source_health_tracking(fetcher):
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=_make_candles(10))
    fetcher._source_manager.fetch_validated_price = AsyncMock(
        return_value=MagicMock(
            source_prices={"hyperliquid": 50000.0, "defillama": 50050.0},
            divergence_pct=0.1,
            price_valid=True,
            anomaly_flag=False,
            source_latencies={"hyperliquid": 50.0, "defillama": 200.0},
        )
    )

    await fetcher.fetch_all_pairs()

    health = fetcher.get_source_health()
    assert "per_pair" in health
    assert "BTC/USDC:USDC" in health["per_pair"]
    assert health["per_pair"]["BTC/USDC:USDC"]["price_valid"] is True


# ── Candle cache tests ───────────────────────────────────────────────────


def _make_recent_candles(n=60, timeframe="1h"):
    """Generate candles with the last one being the current in-progress candle."""
    tf_ms = _TF_MS.get(timeframe, 3_600_000)
    now_ms = int(time.time() * 1000)
    # Last candle = current period start (in-progress candle)
    last_ts = now_ms - (now_ms % tf_ms)
    start_ts = last_ts - ((n - 1) * tf_ms)
    candles = []
    for i in range(n):
        ts = start_ts + i * tf_ms
        p = 50000.0 + i * 10
        candles.append([ts, p, p + 50, p - 50, p + 5, 1000.0 + i])
    return candles


@pytest.mark.asyncio
async def test_cache_cold_start_does_full_fetch(fetcher):
    """First fetch should be a full fetch (cache miss)."""
    candles = _make_recent_candles(10, "1h")
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=candles)

    df = await fetcher.fetch_pair("BTC/USDC:USDC", "1h")

    assert len(df) == 10
    assert fetcher._cache_stats["misses"] == 1
    assert fetcher._cache_stats["hits"] == 0
    fetcher._source_manager._primary.fetch_ohlcv.assert_called_once()


@pytest.mark.asyncio
async def test_cache_hit_skips_api_call(fetcher):
    """Second fetch within the same timeframe should return cache (no API call)."""
    candles = _make_recent_candles(10, "1h")
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=candles)

    # First fetch — populates cache
    df1 = await fetcher.fetch_pair("BTC/USDC:USDC", "1h")

    # Second fetch — should hit cache
    df2 = await fetcher.fetch_pair("BTC/USDC:USDC", "1h")

    assert len(df2) == 10
    assert fetcher._cache_stats["hits"] == 1
    # Only one API call (the first one)
    assert fetcher._source_manager._primary.fetch_ohlcv.call_count == 1


@pytest.mark.asyncio
async def test_cache_incremental_fetch_when_new_candle(fetcher):
    """When enough time has passed for a new candle, fetch incrementally."""
    tf_ms = _TF_MS["1h"]
    now_ms = int(time.time() * 1000)

    # Create candles where the last one is >1h old (a new candle should exist)
    old_candles = []
    for i in range(10):
        ts = now_ms - (12 - i) * tf_ms  # Last candle ~2h ago
        p = 50000.0 + i * 10
        old_candles.append([ts, p, p + 50, p - 50, p + 5, 1000.0])

    # Pre-populate cache with old candles
    fetcher._candle_cache[("BTC/USDC:USDC", "1h")] = old_candles
    fetcher._df_cache[("BTC/USDC:USDC", "1h")] = DataFetcher._candles_to_df(old_candles)

    # New candles from API (incremental fetch)
    new_candles = []
    for i in range(3):
        ts = now_ms - (2 - i) * tf_ms
        p = 50100.0 + i * 10
        new_candles.append([ts, p, p + 50, p - 50, p + 5, 2000.0])

    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(return_value=new_candles)

    df = await fetcher.fetch_pair("BTC/USDC:USDC", "1h")

    assert fetcher._cache_stats["incremental"] == 1
    assert fetcher._cache_stats["hits"] == 0
    # Should have merged old + new candles
    assert len(df) >= 10


@pytest.mark.asyncio
async def test_cache_fallback_on_incremental_failure(fetcher):
    """If incremental fetch fails, return stale cache instead of raising."""
    tf_ms = _TF_MS["1h"]
    now_ms = int(time.time() * 1000)

    # Create candles where the last one is >1h old
    old_candles = []
    for i in range(10):
        ts = now_ms - (12 - i) * tf_ms
        p = 50000.0 + i * 10
        old_candles.append([ts, p, p + 50, p - 50, p + 5, 1000.0])

    fetcher._candle_cache[("BTC/USDC:USDC", "1h")] = old_candles
    fetcher._df_cache[("BTC/USDC:USDC", "1h")] = DataFetcher._candles_to_df(old_candles)

    # API fails on incremental fetch
    fetcher._source_manager._primary.fetch_ohlcv = AsyncMock(
        side_effect=Exception("rate limited")
    )

    # Should NOT raise — returns stale cache
    df = await fetcher.fetch_pair("BTC/USDC:USDC", "1h")
    assert len(df) == 10
    assert fetcher._cache_stats["hits"] == 1


def test_merge_candles_deduplicates():
    """Merge should deduplicate by timestamp and keep latest data."""
    cached = [[1000, 1, 2, 0.5, 1.5, 100], [2000, 2, 3, 1.5, 2.5, 200]]
    new = [[2000, 2.1, 3.1, 1.6, 2.6, 210], [3000, 3, 4, 2.5, 3.5, 300]]

    merged = DataFetcher._merge_candles(cached, new, max_keep=60)

    assert len(merged) == 3
    # ts=2000 should have new data (overwritten)
    ts_2000 = [c for c in merged if c[0] == 2000][0]
    assert ts_2000[1] == 2.1  # Updated open


def test_merge_candles_trims_to_max():
    """Merge should keep only the last max_keep candles."""
    cached = [[i * 1000, i, i + 1, i - 1, i + 0.5, 100] for i in range(50)]
    new = [[i * 1000, i, i + 1, i - 1, i + 0.5, 100] for i in range(50, 55)]

    merged = DataFetcher._merge_candles(cached, new, max_keep=10)

    assert len(merged) == 10
    assert merged[0][0] == 45000  # Oldest kept
    assert merged[-1][0] == 54000  # Newest


def test_candles_to_df_validates():
    """_candles_to_df should drop invalid candles."""
    candles = [
        [1000, 100, 110, 90, 105, 500],    # Valid
        [2000, -1, 110, 90, 105, 500],      # Negative open
        [3000, 100, 80, 90, 85, 500],       # High < low
        [4000, 100, 110, 90, 105, 500],     # Valid
    ]
    df = DataFetcher._candles_to_df(candles)
    assert len(df) == 2


@pytest.mark.asyncio
async def test_cache_different_timeframes_independent(fetcher):
    """Cache for 1h and 4h should be independent."""
    candles_1h = _make_recent_candles(10, "1h")
    candles_4h = _make_recent_candles(10, "4h")

    call_count = 0

    async def mock_fetch(pair, tf, limit):
        nonlocal call_count
        call_count += 1
        if tf == "1h":
            return candles_1h
        return candles_4h

    fetcher._source_manager.fetch_ohlcv = mock_fetch

    # First fetch for both timeframes
    await fetcher.fetch_pair("BTC/USDC:USDC", "1h")
    await fetcher.fetch_pair("BTC/USDC:USDC", "4h")
    assert call_count == 2

    # Second fetch — both should be cache hits
    await fetcher.fetch_pair("BTC/USDC:USDC", "1h")
    await fetcher.fetch_pair("BTC/USDC:USDC", "4h")
    assert call_count == 2  # No new API calls
    assert fetcher._cache_stats["hits"] == 2
