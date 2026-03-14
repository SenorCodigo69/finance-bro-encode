"""Tests for multi-source data layer — classify_pair, adapters, cross-validation, anomaly detection."""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import DataSourcesConfig
from src.data_sources import (
    AlphaVantageSource,
    CoinGeckoSource,
    DataSourceManager,
    DefiLlamaSource,
    FetchResult,
    HyperliquidNativeClient,
    HyperliquidSource,
    YFinanceSource,
    _pair_to_hl_coin,
    classify_pair,
)


# ── classify_pair ────────────────────────────────────────────────────────


def test_classify_btc():
    assert classify_pair("BTC") == "crypto"


def test_classify_eth():
    assert classify_pair("ETH") == "crypto"


def test_classify_aave():
    assert classify_pair("AAVE") == "crypto"


def test_classify_aapl():
    assert classify_pair("XYZ-AAPL") == "stocks"


def test_classify_tsla():
    assert classify_pair("XYZ-TSLA") == "stocks"


def test_classify_nvda():
    assert classify_pair("XYZ-NVDA") == "stocks"


def test_classify_msft():
    assert classify_pair("XYZ-MSFT") == "stocks"


def test_classify_gold():
    assert classify_pair("XYZ-GOLD") == "commodities"


def test_classify_brent():
    assert classify_pair("XYZ-BRENTOIL") == "commodities"


def test_classify_silver():
    assert classify_pair("XYZ-SILVER") == "commodities"


def test_classify_unknown_defaults_crypto():
    assert classify_pair("DOGE") == "crypto"


def test_classify_unknown_xyz_defaults_synthetics():
    assert classify_pair("XYZ-UNKNOWN") == "synthetics"


# ── _pair_to_hl_coin ────────────────────────────────────────────────────


def test_pair_to_hl_coin_crypto():
    assert _pair_to_hl_coin("BTC/USDC:USDC") == "BTC"
    assert _pair_to_hl_coin("ETH/USDC:USDC") == "ETH"
    assert _pair_to_hl_coin("AAVE/USDC:USDC") == "AAVE"


def test_pair_to_hl_coin_synthetics():
    assert _pair_to_hl_coin("XYZ-NVDA/USDC:USDC") == "xyz:NVDA"
    assert _pair_to_hl_coin("XYZ-AAPL/USDC:USDC") == "xyz:AAPL"
    assert _pair_to_hl_coin("XYZ-GOLD/USDC:USDC") == "xyz:GOLD"
    assert _pair_to_hl_coin("XYZ-BRENTOIL/USDC:USDC") == "xyz:BRENTOIL"
    assert _pair_to_hl_coin("XYZ-XYZ100/USDC:USDC") == "xyz:XYZ100"


# ── supports_pair ────────────────────────────────────────────────────────


def test_hyperliquid_supports_all():
    client = MagicMock()
    src = HyperliquidSource(client)
    assert src.supports_pair("BTC") is True
    assert src.supports_pair("XYZ-AAPL") is True
    assert src.supports_pair("ANYTHING") is True


def test_defillama_supports_crypto():
    src = DefiLlamaSource()
    assert src.supports_pair("BTC") is True
    assert src.supports_pair("ETH") is True
    assert src.supports_pair("XYZ-AAPL") is False


def test_coingecko_supports_crypto():
    src = CoinGeckoSource()
    assert src.supports_pair("BTC") is True
    assert src.supports_pair("AAVE") is True
    assert src.supports_pair("XYZ-GOLD") is False


def test_yfinance_supports_stocks_and_commodities():
    src = YFinanceSource()
    assert src.supports_pair("XYZ-AAPL") is True
    assert src.supports_pair("XYZ-GOLD") is True
    assert src.supports_pair("XYZ-BRENTOIL") is True
    assert src.supports_pair("BTC") is False


def test_alphavantage_supports_stocks_and_commodities():
    src = AlphaVantageSource(api_key="test")
    assert src.supports_pair("XYZ-AAPL") is True
    assert src.supports_pair("XYZ-GOLD") is True
    assert src.supports_pair("BTC") is False


def test_alphavantage_no_key_returns_none():
    src = AlphaVantageSource(api_key="")
    assert src.supports_pair("XYZ-AAPL") is True  # supports_pair doesn't check key


# ── Cross-validation ─────────────────────────────────────────────────────


def test_compute_divergence_agreeing():
    divergence = DataSourceManager._compute_divergence({
        "a": 100.0,
        "b": 100.5,
        "c": 99.5,
    })
    assert divergence < 1.0  # Less than 1% divergence


def test_compute_divergence_divergent():
    divergence = DataSourceManager._compute_divergence({
        "a": 100.0,
        "b": 105.0,
        "c": 95.0,
    })
    assert divergence > 4.0  # Significant divergence


def test_compute_divergence_single_source():
    divergence = DataSourceManager._compute_divergence({"a": 100.0})
    assert divergence == 0.0


def test_compute_divergence_empty():
    divergence = DataSourceManager._compute_divergence({})
    assert divergence == 0.0


def test_compute_divergence_zero_median():
    divergence = DataSourceManager._compute_divergence({"a": 0.0, "b": 0.0})
    assert divergence == 0.0


# ── Anomaly detection ────────────────────────────────────────────────────


def test_anomaly_normal_tick():
    config = DataSourcesConfig(anomaly_zscore_threshold=3.0)
    mgr = DataSourceManager(config=config)
    # Seed with stable prices
    mgr._price_history["TEST"] = deque([100, 101, 99, 100, 101, 100, 99, 100], maxlen=20)
    # Normal tick — should not flag
    assert mgr._check_anomaly("TEST", 101.5) is False


def test_anomaly_spike():
    config = DataSourcesConfig(anomaly_zscore_threshold=3.0)
    mgr = DataSourceManager(config=config)
    # Seed with stable prices around 100
    mgr._price_history["TEST"] = deque([100, 101, 99, 100, 101, 100, 99, 100], maxlen=20)
    # Extreme spike — should flag
    assert mgr._check_anomaly("TEST", 200.0) is True


def test_anomaly_insufficient_history():
    config = DataSourcesConfig(anomaly_zscore_threshold=3.0)
    mgr = DataSourceManager(config=config)
    # Not enough data — should not flag
    assert mgr._check_anomaly("TEST", 100.0) is False
    assert mgr._check_anomaly("TEST", 200.0) is False  # Still < 5 samples


def test_anomaly_zero_std():
    config = DataSourcesConfig(anomaly_zscore_threshold=3.0)
    mgr = DataSourceManager(config=config)
    # All same prices — zero std dev — should not flag
    mgr._price_history["TEST"] = deque([100, 100, 100, 100, 100, 100], maxlen=20)
    assert mgr._check_anomaly("TEST", 100.0) is False


# ── DataSourceManager: fetch_validated_price ─────────────────────────────


@pytest.mark.asyncio
async def test_fetch_validated_price_agreeing_sources():
    config = DataSourcesConfig(max_price_divergence_pct=2.0)
    mgr = DataSourceManager(config=config)

    # Mock primary
    mgr._primary.fetch_latest_price = AsyncMock(return_value=50000.0)
    # Mock secondary sources to agree
    for sources in mgr._secondary.values():
        for src in sources:
            src.fetch_latest_price = AsyncMock(return_value=50100.0)

    result = await mgr.fetch_validated_price("BTC/USDC:USDC")

    assert result.price_valid is True
    assert result.divergence_pct < 2.0
    assert "hyperliquid" in result.source_prices
    assert result.latest_price == 50000.0


@pytest.mark.asyncio
async def test_fetch_validated_price_divergent_sources():
    config = DataSourcesConfig(max_price_divergence_pct=2.0)
    mgr = DataSourceManager(config=config)

    mgr._primary.fetch_latest_price = AsyncMock(return_value=50000.0)
    for sources in mgr._secondary.values():
        for src in sources:
            src.fetch_latest_price = AsyncMock(return_value=55000.0)  # 10% off

    result = await mgr.fetch_validated_price("BTC/USDC:USDC")

    assert result.price_valid is False
    assert result.divergence_pct > 2.0


@pytest.mark.asyncio
async def test_fetch_validated_price_single_source():
    config = DataSourcesConfig(max_price_divergence_pct=2.0)
    mgr = DataSourceManager(config=config)

    mgr._primary.fetch_latest_price = AsyncMock(return_value=50000.0)
    for sources in mgr._secondary.values():
        for src in sources:
            src.fetch_latest_price = AsyncMock(return_value=None)

    result = await mgr.fetch_validated_price("BTC/USDC:USDC")

    assert result.price_valid is True  # Can't cross-validate with 1 source
    assert result.latest_price == 50000.0


@pytest.mark.asyncio
async def test_fetch_validated_price_all_fail():
    config = DataSourcesConfig()
    mgr = DataSourceManager(config=config)

    mgr._primary.fetch_latest_price = AsyncMock(return_value=None)
    for sources in mgr._secondary.values():
        for src in sources:
            src.fetch_latest_price = AsyncMock(return_value=None)

    result = await mgr.fetch_validated_price("BTC/USDC:USDC")

    assert result.latest_price is None
    assert result.price_valid is True  # No data to invalidate
    assert len(result.source_prices) == 0


@pytest.mark.asyncio
async def test_fetch_validated_price_primary_fail_uses_median():
    config = DataSourcesConfig()
    mgr = DataSourceManager(config=config)

    mgr._primary.fetch_latest_price = AsyncMock(return_value=None)
    # Inject known secondary prices for crypto
    mgr._secondary["crypto"][0].fetch_latest_price = AsyncMock(return_value=49000.0)
    mgr._secondary["crypto"][1].fetch_latest_price = AsyncMock(return_value=51000.0)

    result = await mgr.fetch_validated_price("BTC/USDC:USDC")

    assert result.latest_price == 50000.0  # Median of 49000 and 51000


# ── DataSourceManager: graceful degradation ──────────────────────────────


@pytest.mark.asyncio
async def test_source_exception_doesnt_crash():
    config = DataSourcesConfig()
    mgr = DataSourceManager(config=config)

    mgr._primary.fetch_latest_price = AsyncMock(return_value=50000.0)
    # Make one secondary raise
    mgr._secondary["crypto"][0].fetch_latest_price = AsyncMock(side_effect=Exception("boom"))
    mgr._secondary["crypto"][1].fetch_latest_price = AsyncMock(return_value=50100.0)

    result = await mgr.fetch_validated_price("BTC/USDC:USDC")

    assert result.latest_price == 50000.0
    assert len(result.source_prices) >= 2  # Primary + one secondary


# ── DataSourceManager: fetch_ohlcv ──────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_ohlcv_success():
    config = DataSourcesConfig()
    mgr = DataSourceManager(config=config)
    candles = [[1700000000000, 50000, 50100, 49900, 50050, 1000]]
    mgr._primary.fetch_ohlcv = AsyncMock(return_value=candles)

    result = await mgr.fetch_ohlcv("BTC/USDC:USDC", "1h")
    assert result == candles


@pytest.mark.asyncio
async def test_fetch_ohlcv_failure_raises():
    config = DataSourcesConfig()
    mgr = DataSourceManager(config=config)
    mgr._primary.fetch_ohlcv = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="Primary OHLCV failed"):
        await mgr.fetch_ohlcv("BTC/USDC:USDC", "1h")


# ── Latency tracking ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_latency_tracking():
    config = DataSourcesConfig()
    mgr = DataSourceManager(config=config)

    mgr._primary.fetch_latest_price = AsyncMock(return_value=50000.0)
    for sources in mgr._secondary.values():
        for src in sources:
            src.fetch_latest_price = AsyncMock(return_value=50000.0)

    await mgr.fetch_validated_price("BTC/USDC:USDC")

    health = mgr.get_source_health()
    assert "hyperliquid" in health
    assert health["hyperliquid"]["calls"] >= 1


# ── FetchResult dataclass ───────────────────────────────────────────────


def test_fetch_result_defaults():
    r = FetchResult()
    assert r.candles == []
    assert r.latest_price is None
    assert r.price_valid is True
    assert r.anomaly_flag is False
    assert r.divergence_pct == 0.0


# ── Config parsing ───────────────────────────────────────────────────────


def test_data_sources_config_defaults():
    config = DataSourcesConfig()
    assert config.max_price_divergence_pct == 2.0
    assert config.anomaly_zscore_threshold == 3.0


def test_data_sources_config_from_yaml():
    import yaml
    from src.config import load_config

    yaml_content = {
        "data_sources": {
            "max_price_divergence_pct": 3.5,
            "anomaly_zscore_threshold": 4.0,
        }
    }

    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(yaml_content, f)
        f.flush()
        cfg = load_config(config_path=f.name)

    assert cfg.data_sources.max_price_divergence_pct == 3.5
    assert cfg.data_sources.anomaly_zscore_threshold == 4.0


# ── HyperliquidNativeClient ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_native_client_fetch_candles_parses_response():
    """Verify candleSnapshot response parsing to ccxt format."""
    client = HyperliquidNativeClient()
    raw_response = [
        {"t": 1700000000000, "T": 1700003599999, "s": "BTC", "i": "1h",
         "o": "67000.0", "c": "67100.0", "h": "67200.0", "l": "66900.0",
         "v": "123.45", "n": 100},
        {"t": 1700003600000, "T": 1700007199999, "s": "BTC", "i": "1h",
         "o": "67100.0", "c": "67300.0", "h": "67400.0", "l": "67050.0",
         "v": "150.2", "n": 120},
    ]
    client._post = AsyncMock(return_value=raw_response)

    candles = await client.fetch_candles("BTC", "1h", 1700000000000, 1700007200000)

    assert len(candles) == 2
    # [timestamp, open, high, low, close, volume]
    assert candles[0] == [1700000000000, 67000.0, 67200.0, 66900.0, 67100.0, 123.45]
    assert candles[1] == [1700003600000, 67100.0, 67400.0, 67050.0, 67300.0, 150.2]
    await client.close()


@pytest.mark.asyncio
async def test_native_client_fetch_all_mids():
    """Verify allMids response parsing."""
    client = HyperliquidNativeClient()
    client._post = AsyncMock(return_value={"BTC": "67000.5", "ETH": "3400.25"})

    mids = await client.fetch_all_mids()

    assert mids["BTC"] == 67000.5
    assert mids["ETH"] == 3400.25
    await client.close()


@pytest.mark.asyncio
async def test_native_client_handles_empty_response():
    client = HyperliquidNativeClient()
    client._post = AsyncMock(return_value=None)

    assert await client.fetch_all_mids() == {}
    assert await client.fetch_candles("BTC", "1h", 0, 1) == []
    await client.close()


@pytest.mark.asyncio
async def test_native_client_handles_malformed_candle():
    """Malformed candles should be skipped, not crash."""
    client = HyperliquidNativeClient()
    raw = [
        {"t": 1700000000000, "o": "67000.0", "h": "67200.0", "l": "66900.0",
         "c": "67100.0", "v": "123.45"},  # Valid
        {"t": 1700003600000, "o": "bad"},  # Missing fields
        {"broken": True},  # Totally broken
    ]
    client._post = AsyncMock(return_value=raw)

    candles = await client.fetch_candles("BTC", "1h", 0, 1)
    assert len(candles) == 1  # Only the valid one
    await client.close()


# ── HyperliquidSource with native client ─────────────────────────────────


@pytest.mark.asyncio
async def test_hl_source_fetch_latest_price_crypto():
    """fetch_latest_price uses allMids for crypto pairs."""
    client = MagicMock()
    client.fetch_all_mids = AsyncMock(return_value={"BTC": 67000.0, "ETH": 3400.0})
    client.fetch_all_mids_xyz = AsyncMock(return_value={})
    src = HyperliquidSource(client)

    price = await src.fetch_latest_price("BTC/USDC:USDC")
    assert price == 67000.0


@pytest.mark.asyncio
async def test_hl_source_fetch_latest_price_synthetic():
    """fetch_latest_price uses xyz allMids for synthetic pairs."""
    client = MagicMock()
    client.fetch_all_mids = AsyncMock(return_value={"BTC": 67000.0})
    client.fetch_all_mids_xyz = AsyncMock(return_value={"xyz:NVDA": 183.5})
    src = HyperliquidSource(client)

    price = await src.fetch_latest_price("XYZ-NVDA/USDC:USDC")
    assert price == 183.5


@pytest.mark.asyncio
async def test_hl_source_mids_cache():
    """allMids should be cached within the TTL."""
    client = MagicMock()
    client.fetch_all_mids = AsyncMock(return_value={"BTC": 67000.0})
    client.fetch_all_mids_xyz = AsyncMock(return_value={})
    src = HyperliquidSource(client)

    await src.fetch_latest_price("BTC/USDC:USDC")
    await src.fetch_latest_price("BTC/USDC:USDC")

    # allMids called only once (cached)
    assert client.fetch_all_mids.call_count == 1


@pytest.mark.asyncio
async def test_hl_source_fetch_ohlcv():
    """fetch_ohlcv calls native client with correct coin name."""
    client = MagicMock()
    candles = [[1700000000000, 67000.0, 67200.0, 66900.0, 67100.0, 123.45]]
    client.fetch_candles = AsyncMock(return_value=candles)
    src = HyperliquidSource(client)

    result = await src.fetch_ohlcv("XYZ-NVDA/USDC:USDC", "1h", 60)

    assert result == candles
    # Verify coin name mapping
    call_args = client.fetch_candles.call_args
    assert call_args[0][0] == "xyz:NVDA"
    assert call_args[0][1] == "1h"
