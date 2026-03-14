"""Tests for the cross-DEX price scanner."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.config import DexScannerConfig, DexVenueConfig
from src.dex_scanner import (
    DexScanner,
    DexScanResult,
    DexVenueSnapshot,
    DydxAdapter,
    GmxAdapter,
    HyperliquidAdapter,
    _DYDX_PAIR_MAP,
    _GMX_PAIR_MAP,
)


# --- DexVenueSnapshot tests ---

def test_venue_snapshot_defaults():
    snap = DexVenueSnapshot(venue="test", pair="BTC/USDC:USDC")
    assert snap.mark_price is None
    assert snap.taker_fee_bps == 0.0
    assert snap.latency_ms == 0.0


def test_venue_snapshot_with_data():
    snap = DexVenueSnapshot(
        venue="hyperliquid", pair="BTC/USDC:USDC",
        mark_price=50000.0, best_bid=49990.0, best_ask=50010.0,
        spread_bps=4.0, funding_rate=0.0003, taker_fee_bps=4.5,
    )
    assert snap.mark_price == 50000.0
    assert snap.spread_bps == 4.0


# --- DexScanResult tests ---

def test_scan_result_defaults():
    result = DexScanResult(pair="BTC/USDC:USDC")
    assert result.max_price_divergence_pct == 0.0
    assert result.arb_opportunity_bps == 0.0
    assert result.venue_snapshots == {}


# --- Adapter supports_pair tests ---

def test_hl_adapter_supports_all():
    adapter = HyperliquidAdapter()
    assert adapter.supports_pair("BTC/USDC:USDC")
    assert adapter.supports_pair("XYZ-GOLD/USDC:USDC")
    assert adapter.supports_pair("ANY/THING:THING")


def test_dydx_adapter_supports_mapped():
    adapter = DydxAdapter()
    assert adapter.supports_pair("BTC/USDC:USDC")
    assert adapter.supports_pair("ETH/USDC:USDC")
    assert adapter.supports_pair("AAVE/USDC:USDC")
    assert not adapter.supports_pair("XYZ-GOLD/USDC:USDC")
    assert not adapter.supports_pair("XYZ-TSLA/USDC:USDC")


def test_gmx_adapter_supports_mapped():
    adapter = GmxAdapter()
    assert adapter.supports_pair("BTC/USDC:USDC")
    assert adapter.supports_pair("ETH/USDC:USDC")
    assert not adapter.supports_pair("AAVE/USDC:USDC")
    assert not adapter.supports_pair("XYZ-AAPL/USDC:USDC")


def test_dydx_pair_map_has_expected_pairs():
    assert "BTC/USDC:USDC" in _DYDX_PAIR_MAP
    assert "ETH/USDC:USDC" in _DYDX_PAIR_MAP
    assert _DYDX_PAIR_MAP["BTC/USDC:USDC"] == "BTC-USD"


def test_gmx_pair_map_has_expected_pairs():
    assert "BTC/USDC:USDC" in _GMX_PAIR_MAP
    assert _GMX_PAIR_MAP["BTC/USDC:USDC"] == "BTC"


# --- DexScanner initialization tests ---

def test_scanner_init_default_config():
    exchange = MagicMock()
    config = DexScannerConfig()
    scanner = DexScanner(exchange, config)
    # Should have HL + dYdX (dYdX enabled by default, GMX disabled)
    assert len(scanner._adapters) == 2
    venue_names = [a.venue_name for a in scanner._adapters]
    assert "hyperliquid" in venue_names
    assert "dydx" in venue_names


def test_scanner_init_all_disabled():
    exchange = MagicMock()
    config = DexScannerConfig(venues=[
        DexVenueConfig(name="dydx", enabled=False),
        DexVenueConfig(name="gmx", enabled=False),
    ])
    scanner = DexScanner(exchange, config)
    # Only HL adapter (always included)
    assert len(scanner._adapters) == 1
    assert scanner._adapters[0].venue_name == "hyperliquid"


def test_scanner_init_all_enabled():
    exchange = MagicMock()
    config = DexScannerConfig(venues=[
        DexVenueConfig(name="dydx", enabled=True),
        DexVenueConfig(name="gmx", enabled=True),
    ])
    scanner = DexScanner(exchange, config)
    assert len(scanner._adapters) == 3


# --- DexScanner._build_result tests ---

def test_build_result_single_venue():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    snap = DexVenueSnapshot(
        venue="hyperliquid", pair="BTC/USDC:USDC",
        mark_price=50000.0, best_bid=49990.0, best_ask=50010.0,
    )
    result = scanner._build_result("BTC/USDC:USDC", {"hyperliquid": snap})
    assert result.best_bid_venue == "hyperliquid"
    assert result.best_ask_venue == "hyperliquid"
    assert result.max_price_divergence_pct == 0.0


def test_build_result_two_venues_no_divergence():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    snaps = {
        "hyperliquid": DexVenueSnapshot(
            venue="hyperliquid", pair="BTC/USDC:USDC",
            mark_price=50000.0, best_bid=49990.0, best_ask=50010.0,
            taker_fee_bps=4.5,
        ),
        "dydx": DexVenueSnapshot(
            venue="dydx", pair="BTC/USDC:USDC",
            mark_price=50005.0, best_bid=49995.0, best_ask=50015.0,
            taker_fee_bps=5.0,
        ),
    }
    result = scanner._build_result("BTC/USDC:USDC", snaps)
    assert result.max_price_divergence_pct < 0.1  # Tiny divergence


def test_build_result_two_venues_with_divergence():
    exchange = MagicMock()
    config = DexScannerConfig(divergence_alert_pct=1.0)
    scanner = DexScanner(exchange, config)
    snaps = {
        "hyperliquid": DexVenueSnapshot(
            venue="hyperliquid", pair="BTC/USDC:USDC",
            mark_price=50000.0, best_bid=49990.0, best_ask=50010.0,
            taker_fee_bps=4.5,
        ),
        "dydx": DexVenueSnapshot(
            venue="dydx", pair="BTC/USDC:USDC",
            mark_price=50800.0, best_bid=50790.0, best_ask=50810.0,
            taker_fee_bps=5.0,
        ),
    }
    result = scanner._build_result("BTC/USDC:USDC", snaps)
    assert result.max_price_divergence_pct > 1.0


def test_build_result_arb_opportunity():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    # dYdX has lower ask, HL has higher bid — arb exists
    snaps = {
        "hyperliquid": DexVenueSnapshot(
            venue="hyperliquid", pair="BTC/USDC:USDC",
            mark_price=50100.0, best_bid=50100.0, best_ask=50110.0,
            taker_fee_bps=4.5,
        ),
        "dydx": DexVenueSnapshot(
            venue="dydx", pair="BTC/USDC:USDC",
            mark_price=50000.0, best_bid=49990.0, best_ask=50000.0,
            taker_fee_bps=5.0,
        ),
    }
    result = scanner._build_result("BTC/USDC:USDC", snaps)
    # Buy on dydx (50000), sell on HL (50100) → ~20bps gross - 9.5bps fees
    assert result.arb_opportunity_bps > 0
    assert result.best_ask_venue == "dydx"
    assert result.best_bid_venue == "hyperliquid"


def test_build_result_funding_divergence():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    snaps = {
        "hyperliquid": DexVenueSnapshot(
            venue="hyperliquid", pair="BTC/USDC:USDC",
            mark_price=50000.0, funding_rate=0.0003,
        ),
        "dydx": DexVenueSnapshot(
            venue="dydx", pair="BTC/USDC:USDC",
            mark_price=50000.0, funding_rate=-0.0001,
        ),
    }
    result = scanner._build_result("BTC/USDC:USDC", snaps)
    assert "hyperliquid" in result.funding_divergence
    assert "dydx" in result.funding_divergence
    assert result.funding_divergence["hyperliquid"] == 0.0003
    assert result.funding_divergence["dydx"] == -0.0001


def test_build_result_empty():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    result = scanner._build_result("BTC/USDC:USDC", {})
    assert result.best_bid_venue is None
    assert result.best_ask_venue is None


# --- Cache tests ---

def test_scanner_cache_ttl():
    """Scan result should be cached and reused within TTL."""
    exchange = MagicMock()
    config = DexScannerConfig(scan_interval_sec=60)
    scanner = DexScanner(exchange, config)

    # Manually inject a cached result
    cached = DexScanResult(pair="BTC/USDC:USDC", scan_time="2026-01-01T00:00:00")
    scanner._scan_cache["BTC/USDC:USDC"] = cached
    import time
    scanner._cache_timestamps["BTC/USDC:USDC"] = time.monotonic()

    # scan_pair should return cached without hitting network
    result = asyncio.get_event_loop().run_until_complete(
        scanner.scan_pair("BTC/USDC:USDC")
    )
    assert result.scan_time == "2026-01-01T00:00:00"


# --- get_best_venue tests ---

def test_get_best_venue_no_cache():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    # No cached scan — should return default
    assert scanner.get_best_venue("BTC/USDC:USDC", "buy") is not None


def test_get_best_venue_buy():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    scanner._scan_cache["BTC/USDC:USDC"] = DexScanResult(
        pair="BTC/USDC:USDC",
        best_ask_venue="dydx",
        best_bid_venue="hyperliquid",
    )
    assert scanner.get_best_venue("BTC/USDC:USDC", "buy") == "dydx"


def test_get_best_venue_sell():
    exchange = MagicMock()
    scanner = DexScanner(exchange, DexScannerConfig())
    scanner._scan_cache["BTC/USDC:USDC"] = DexScanResult(
        pair="BTC/USDC:USDC",
        best_ask_venue="dydx",
        best_bid_venue="hyperliquid",
    )
    assert scanner.get_best_venue("BTC/USDC:USDC", "sell") == "hyperliquid"


# --- HyperliquidAdapter fetch tests ---

def _mock_json_response(data, status=200):
    """Create a mock aiohttp response that works with _read_json (resp.read())."""
    import json as _json
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.read = AsyncMock(return_value=_json.dumps(data).encode())
    return mock_resp


@pytest.mark.asyncio
async def test_hl_adapter_fetch_success():
    adapter = HyperliquidAdapter()
    mock_resp = _mock_json_response([
        {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
        [
            {"markPx": "50000", "midPx": "50000", "funding": "0.0003", "openInterest": "1000"},
            {"markPx": "3000", "midPx": "3000", "funding": "0.0002", "openInterest": "500"},
        ],
    ])

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_resp),
        __aexit__=AsyncMock(return_value=False),
    ))

    snap = await adapter.fetch_snapshot("BTC/USDC:USDC", mock_session)
    assert snap is not None
    assert snap.venue == "hyperliquid"
    assert snap.mark_price == 50000.0
    assert snap.funding_rate == 0.0003


@pytest.mark.asyncio
async def test_hl_adapter_fetch_api_error():
    adapter = HyperliquidAdapter()
    mock_resp = AsyncMock()
    mock_resp.status = 500

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_resp),
        __aexit__=AsyncMock(return_value=False),
    ))

    snap = await adapter.fetch_snapshot("BTC/USDC:USDC", mock_session)
    assert snap is None


@pytest.mark.asyncio
async def test_hl_adapter_pair_not_found():
    adapter = HyperliquidAdapter()
    mock_resp = _mock_json_response([
        {"universe": [{"name": "ETH"}]},
        [{"markPx": "3000", "midPx": "3000", "funding": "0.0002", "openInterest": "500"}],
    ])

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_resp),
        __aexit__=AsyncMock(return_value=False),
    ))

    snap = await adapter.fetch_snapshot("BTC/USDC:USDC", mock_session)
    assert snap is None


# --- DydxAdapter fetch tests ---

@pytest.mark.asyncio
async def test_dydx_adapter_unsupported_pair():
    adapter = DydxAdapter()
    session = AsyncMock()
    snap = await adapter.fetch_snapshot("XYZ-GOLD/USDC:USDC", session)
    assert snap is None


@pytest.mark.asyncio
async def test_dydx_adapter_fetch_success():
    adapter = DydxAdapter()

    market_resp = _mock_json_response({
        "markets": {
            "BTC-USD": {
                "oraclePrice": "50000",
                "nextFundingRate": "0.0001",
                "openInterest": "2000",
            }
        }
    })

    ob_resp = _mock_json_response({
        "bids": [{"price": "49990", "size": "10"}],
        "asks": [{"price": "50010", "size": "10"}],
    })

    mock_session = AsyncMock()
    call_count = 0

    def make_context(url, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = market_resp if call_count == 1 else ob_resp
        return AsyncMock(
            __aenter__=AsyncMock(return_value=resp),
            __aexit__=AsyncMock(return_value=False),
        )

    mock_session.get = MagicMock(side_effect=make_context)

    snap = await adapter.fetch_snapshot("BTC/USDC:USDC", mock_session)
    assert snap is not None
    assert snap.venue == "dydx"
    assert snap.mark_price == 50000.0
    assert snap.best_bid == 49990.0
    assert snap.best_ask == 50010.0
    assert snap.funding_rate == 0.0001


# --- GmxAdapter fetch tests ---

@pytest.mark.asyncio
async def test_gmx_adapter_unsupported_pair():
    adapter = GmxAdapter()
    session = AsyncMock()
    snap = await adapter.fetch_snapshot("AAVE/USDC:USDC", session)
    assert snap is None


# --- Config tests ---

def test_dex_scanner_config_defaults():
    config = DexScannerConfig()
    assert config.enabled is True
    assert config.scan_interval_sec == 60
    assert config.divergence_alert_pct == 1.0
    assert len(config.venues) == 2


def test_dex_venue_config():
    v = DexVenueConfig(name="dydx", enabled=True)
    assert v.name == "dydx"
    assert v.enabled is True
