"""Tests for the execution router."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.config import ExecutionRouterConfig
from src.dex_scanner import DexScanResult, DexVenueSnapshot
from src.execution_router import ExecutionPlan, ExecutionRouter, VenueScore


# --- ExecutionPlan/VenueScore dataclass tests ---

def test_execution_plan_defaults():
    plan = ExecutionPlan(
        venue="hyperliquid", pair="BTC/USDC:USDC", side="buy",
        amount=0.01, expected_price=50000.0, expected_fee_bps=4.5,
        expected_slippage_bps=5.0, reasoning="test",
    )
    assert plan.alternatives == []
    assert plan.mid_market_price is None


def test_venue_score():
    score = VenueScore(
        venue="dydx", effective_price=50000.0,
        taker_fee_bps=5.0, spread_bps=3.0,
        depth_at_size=100.0, latency_ms=150.0,
        score=50000.0,
    )
    assert score.venue == "dydx"


# --- Router initialization ---

def test_router_init_default():
    exchange = MagicMock()
    config = ExecutionRouterConfig()
    router = ExecutionRouter(exchange, config)
    assert router._config.primary_venue == "hyperliquid"
    assert router._dex_scanner is None


def test_router_init_with_scanner():
    exchange = MagicMock()
    config = ExecutionRouterConfig()
    scanner = MagicMock()
    router = ExecutionRouter(exchange, config, dex_scanner=scanner)
    assert router._dex_scanner is scanner


# --- plan_execution tests ---

@pytest.mark.asyncio
async def test_plan_execution_basic():
    exchange = AsyncMock()
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    config = ExecutionRouterConfig()
    router = ExecutionRouter(exchange, config)

    plan = await router.plan_execution("BTC/USDC:USDC", "buy", 0.01)
    assert plan.venue == "hyperliquid"
    assert plan.pair == "BTC/USDC:USDC"
    assert plan.side == "buy"
    assert plan.amount == 0.01
    assert plan.expected_price == 50000.0
    assert plan.mid_market_price == 50000.0
    assert "Primary venue" in plan.reasoning


@pytest.mark.asyncio
async def test_plan_execution_ticker_fails():
    exchange = AsyncMock()
    exchange.fetch_ticker = AsyncMock(side_effect=Exception("timeout"))
    config = ExecutionRouterConfig()
    router = ExecutionRouter(exchange, config)

    plan = await router.plan_execution("BTC/USDC:USDC", "buy", 0.01)
    assert plan.venue == "hyperliquid"
    assert plan.mid_market_price is None


@pytest.mark.asyncio
async def test_plan_execution_multi_venue_disabled():
    """Even with scanner, multi-venue off means primary always wins."""
    exchange = AsyncMock()
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    config = ExecutionRouterConfig(enable_multi_venue=False)
    scanner = MagicMock()
    router = ExecutionRouter(exchange, config, dex_scanner=scanner)

    plan = await router.plan_execution("BTC/USDC:USDC", "buy", 0.01)
    assert plan.venue == "hyperliquid"


# --- execute tests ---

@pytest.mark.asyncio
async def test_execute_routes_to_exchange():
    exchange = AsyncMock()
    exchange.create_market_order = AsyncMock(return_value={
        "id": "order-1", "price": 50005.0, "fee": 0.0225,
    })
    config = ExecutionRouterConfig()
    router = ExecutionRouter(exchange, config)

    plan = ExecutionPlan(
        venue="hyperliquid", pair="BTC/USDC:USDC", side="buy",
        amount=0.01, expected_price=50000.0, expected_fee_bps=4.5,
        expected_slippage_bps=5.0, reasoning="test",
        mid_market_price=50000.0,
    )

    order = await router.execute(plan)
    assert order["execution_venue"] == "hyperliquid"
    assert "slippage_actual_bps" in order
    exchange.create_market_order.assert_called_once_with("BTC/USDC:USDC", "buy", 0.01)


@pytest.mark.asyncio
async def test_execute_slippage_tracking_buy():
    exchange = AsyncMock()
    exchange.create_market_order = AsyncMock(return_value={
        "price": 50050.0, "fee": 0.02,
    })
    config = ExecutionRouterConfig()
    router = ExecutionRouter(exchange, config)

    plan = ExecutionPlan(
        venue="hyperliquid", pair="BTC/USDC:USDC", side="buy",
        amount=0.01, expected_price=50000.0, expected_fee_bps=4.5,
        expected_slippage_bps=5.0, reasoning="test",
        mid_market_price=50000.0,
    )

    order = await router.execute(plan)
    # Bought at 50050 vs mid 50000 → 10bps slippage
    assert order["slippage_actual_bps"] == pytest.approx(10.0, abs=0.1)


@pytest.mark.asyncio
async def test_execute_slippage_tracking_sell():
    exchange = AsyncMock()
    exchange.create_market_order = AsyncMock(return_value={
        "price": 49950.0, "fee": 0.02,
    })
    config = ExecutionRouterConfig()
    router = ExecutionRouter(exchange, config)

    plan = ExecutionPlan(
        venue="hyperliquid", pair="BTC/USDC:USDC", side="sell",
        amount=0.01, expected_price=50000.0, expected_fee_bps=4.5,
        expected_slippage_bps=5.0, reasoning="test",
        mid_market_price=50000.0,
    )

    order = await router.execute(plan)
    # Sold at 49950 vs mid 50000 → 10bps slippage
    assert order["slippage_actual_bps"] == pytest.approx(10.0, abs=0.1)


@pytest.mark.asyncio
async def test_execute_unsupported_venue_raises_when_multi_disabled():
    """Non-hyperliquid venue should raise when multi-venue is disabled."""
    exchange = AsyncMock()
    config = ExecutionRouterConfig(enable_multi_venue=False)
    router = ExecutionRouter(exchange, config)

    plan = ExecutionPlan(
        venue="dydx", pair="BTC/USDC:USDC", side="buy",
        amount=0.01, expected_price=50000.0, expected_fee_bps=5.0,
        expected_slippage_bps=5.0, reasoning="dydx better",
    )

    with pytest.raises(ValueError, match="multi-venue is disabled"):
        await router.execute(plan)


@pytest.mark.asyncio
async def test_execute_unsupported_venue_falls_back_when_multi_enabled():
    """Non-hyperliquid venue should warn and fall back when multi-venue enabled."""
    exchange = AsyncMock()
    exchange.create_market_order = AsyncMock(return_value={
        "price": 50000.0, "fee": 0.02,
    })
    config = ExecutionRouterConfig(enable_multi_venue=True)
    router = ExecutionRouter(exchange, config)

    plan = ExecutionPlan(
        venue="dydx", pair="BTC/USDC:USDC", side="buy",
        amount=0.01, expected_price=50000.0, expected_fee_bps=5.0,
        expected_slippage_bps=5.0, reasoning="dydx better",
    )

    order = await router.execute(plan)
    exchange.create_market_order.assert_called_once()


# --- calculate_slippage tests ---

def test_calculate_slippage_buy():
    router = ExecutionRouter(MagicMock(), ExecutionRouterConfig())
    # Bought at 50050 vs mid 50000
    bps = router.calculate_slippage(50050.0, 50000.0, "buy")
    assert bps == pytest.approx(10.0, abs=0.1)


def test_calculate_slippage_sell():
    router = ExecutionRouter(MagicMock(), ExecutionRouterConfig())
    # Sold at 49950 vs mid 50000
    bps = router.calculate_slippage(49950.0, 50000.0, "sell")
    assert bps == pytest.approx(10.0, abs=0.1)


def test_calculate_slippage_zero_mid():
    router = ExecutionRouter(MagicMock(), ExecutionRouterConfig())
    assert router.calculate_slippage(50000.0, 0, "buy") is None


def test_calculate_slippage_zero_fill():
    router = ExecutionRouter(MagicMock(), ExecutionRouterConfig())
    assert router.calculate_slippage(0, 50000.0, "buy") is None


# --- Config tests ---

def test_execution_router_config_defaults():
    config = ExecutionRouterConfig()
    assert config.primary_venue == "hyperliquid"
    assert config.enable_multi_venue is False
    assert config.slippage_tracking is True
