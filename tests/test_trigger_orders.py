"""Tests for the on-chain trigger order manager."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import TriggerOrderConfig
from src.database import Database
from src.models import Trade
from src.trigger_orders import TriggerOrder, TriggerOrderManager


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def paper_exchange():
    ex = MagicMock()
    ex.mode = "paper"
    return ex


@pytest.fixture
def live_exchange():
    ex = MagicMock()
    ex.mode = "live"
    ex.create_trigger_order = AsyncMock(return_value={"id": "hl-order-default"})
    ex.fetch_order = AsyncMock(return_value={"status": "open"})
    ex.cancel_order = AsyncMock(return_value={})
    return ex


@pytest.fixture
def config():
    return TriggerOrderConfig(
        enabled=True,
        trailing_update_min_interval_sec=300,
        trailing_update_min_move_pct=0.5,
    )


def make_trade(**kwargs) -> Trade:
    defaults = dict(
        id=1, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01,
        stop_loss=48000.0, take_profit=54000.0,
        status="open", entry_time="2026-01-01T00:00:00",
        user_take_profit=53000.0, bot_take_profit=54000.0,
    )
    defaults.update(kwargs)
    return Trade(**defaults)


# --- TriggerOrder dataclass ---

def test_trigger_order_defaults():
    t = TriggerOrder(
        id=1, trade_id=1, exchange_order_id=None,
        pair="BTC/USDC:USDC", side="sell",
        trigger_price=48000.0, quantity=0.01,
        order_type="stop_loss", status="pending",
    )
    assert t.placed_time is None
    assert t.error is None


# --- place_triggers_for_trade (paper mode) ---

@pytest.mark.asyncio
async def test_place_triggers_paper_long(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade()
    # Save trade to DB first so it has an ID
    trade.id = db.save_trade(trade)

    triggers = await mgr.place_triggers_for_trade(trade)

    # Should place SL + user TP + bot TP = 3 triggers
    assert len(triggers) == 3
    types = [t.order_type for t in triggers]
    assert "stop_loss" in types
    assert "user_take_profit" in types
    assert "bot_take_profit" in types

    # All should be "placed" in paper mode
    assert all(t.status == "placed" for t in triggers)

    # Trade should be marked as having triggers
    assert trade.trigger_orders_placed is True


@pytest.mark.asyncio
async def test_place_triggers_paper_short(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade(
        direction="short", stop_loss=52000.0,
        user_take_profit=47000.0, bot_take_profit=46000.0,
    )
    trade.id = db.save_trade(trade)

    triggers = await mgr.place_triggers_for_trade(trade)
    assert len(triggers) == 3

    # For short: SL is buy, TP is buy
    sl = [t for t in triggers if t.order_type == "stop_loss"][0]
    assert sl.side == "buy"
    assert sl.trigger_price == 52000.0


@pytest.mark.asyncio
async def test_place_triggers_no_trade_id(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade(id=None)

    triggers = await mgr.place_triggers_for_trade(trade)
    assert triggers == []


@pytest.mark.asyncio
async def test_place_triggers_no_stop_loss(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade(stop_loss=0, user_take_profit=None, bot_take_profit=None)
    trade.id = db.save_trade(trade)

    triggers = await mgr.place_triggers_for_trade(trade)
    assert len(triggers) == 0  # No SL, no TPs


@pytest.mark.asyncio
async def test_place_triggers_sl_only(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade(user_take_profit=None, bot_take_profit=None)
    trade.id = db.save_trade(trade)

    triggers = await mgr.place_triggers_for_trade(trade)
    assert len(triggers) == 1
    assert triggers[0].order_type == "stop_loss"


# --- place_triggers_for_trade (live mode) ---

@pytest.mark.asyncio
async def test_place_triggers_live(live_exchange, db, config):
    live_exchange.create_trigger_order = AsyncMock(return_value={"id": "hl-order-123"})
    mgr = TriggerOrderManager(live_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    triggers = await mgr.place_triggers_for_trade(trade)
    assert len(triggers) == 3
    assert all(t.status == "placed" for t in triggers)
    assert all(t.exchange_order_id == "hl-order-123" for t in triggers)

    # Verify native API was called
    calls = live_exchange.create_trigger_order.call_args_list
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_place_triggers_live_api_failure(live_exchange, db, config):
    live_exchange.create_trigger_order = AsyncMock(side_effect=Exception("API error"))
    mgr = TriggerOrderManager(live_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    triggers = await mgr.place_triggers_for_trade(trade)
    # All triggers should fail but not crash
    assert len(triggers) == 0  # Failed triggers return None


# --- cancel_triggers_for_trade ---

@pytest.mark.asyncio
async def test_cancel_triggers_paper(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    # Place triggers first
    await mgr.place_triggers_for_trade(trade)

    # Cancel them
    cancelled = await mgr.cancel_triggers_for_trade(trade.id)
    assert cancelled == 3

    # Verify all marked as cancelled in DB
    triggers = db.get_trigger_orders_for_trade(trade.id)
    assert all(t["status"] == "cancelled" for t in triggers)


@pytest.mark.asyncio
async def test_cancel_triggers_live(live_exchange, db, config):
    live_exchange.create_trigger_order = AsyncMock(return_value={"id": "hl-123"})
    mgr = TriggerOrderManager(live_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    await mgr.place_triggers_for_trade(trade)
    cancelled = await mgr.cancel_triggers_for_trade(trade.id)
    assert cancelled == 3

    # Should have called cancel_order for each
    assert live_exchange.cancel_order.call_count == 3


@pytest.mark.asyncio
async def test_cancel_triggers_already_cancelled(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    await mgr.place_triggers_for_trade(trade)
    await mgr.cancel_triggers_for_trade(trade.id)

    # Second cancel should return 0 (already cancelled)
    cancelled = await mgr.cancel_triggers_for_trade(trade.id)
    assert cancelled == 0


# --- sync_trigger_status ---

@pytest.mark.asyncio
async def test_sync_paper_mode_noop(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    filled = await mgr.sync_trigger_status()
    assert filled == []


@pytest.mark.asyncio
async def test_sync_live_mode_filled(live_exchange, db, config):
    live_exchange.create_trigger_order = AsyncMock(return_value={"id": "hl-456"})
    live_exchange.fetch_order = AsyncMock(return_value={"status": "closed"})

    mgr = TriggerOrderManager(live_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    await mgr.place_triggers_for_trade(trade)
    filled = await mgr.sync_trigger_status()
    assert len(filled) == 3  # All three triggers "filled"

    # Check DB updated
    triggers = db.get_trigger_orders_for_trade(trade.id)
    assert all(t["status"] == "triggered" for t in triggers)


@pytest.mark.asyncio
async def test_sync_live_mode_not_filled(live_exchange, db, config):
    live_exchange.create_trigger_order = AsyncMock(return_value={"id": "hl-789"})
    live_exchange.fetch_order = AsyncMock(return_value={"status": "open"})

    mgr = TriggerOrderManager(live_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    await mgr.place_triggers_for_trade(trade)
    filled = await mgr.sync_trigger_status()
    assert len(filled) == 0


# --- update_trailing_triggers ---

@pytest.mark.asyncio
async def test_update_trailing_respects_time_limit(paper_exchange, db, config):
    config.trailing_update_min_interval_sec = 300
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    await mgr.place_triggers_for_trade(trade)

    # First update should work
    result = await mgr.update_trailing_triggers(trade, 48500.0)
    assert result is True

    # Second immediate update should be rate-limited
    result = await mgr.update_trailing_triggers(trade, 49000.0)
    assert result is False


@pytest.mark.asyncio
async def test_update_trailing_respects_price_limit(paper_exchange, db, config):
    config.trailing_update_min_interval_sec = 0  # Disable time limit for this test
    config.trailing_update_min_move_pct = 0.5
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    await mgr.place_triggers_for_trade(trade)

    # First update
    result = await mgr.update_trailing_triggers(trade, 48500.0)
    assert result is True

    # Tiny move (less than 0.5%) should be rejected
    result = await mgr.update_trailing_triggers(trade, 48510.0)
    assert result is False

    # Big move (>0.5%) should work
    result = await mgr.update_trailing_triggers(trade, 49000.0)
    assert result is True


@pytest.mark.asyncio
async def test_update_trailing_no_trade_id(paper_exchange, db, config):
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade(id=None)
    result = await mgr.update_trailing_triggers(trade, 48500.0)
    assert result is False


# --- DB persistence ---

def test_trigger_orders_persisted(db, paper_exchange, config):
    """Trigger orders should survive DB read after write."""
    import asyncio
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    asyncio.get_event_loop().run_until_complete(
        mgr.place_triggers_for_trade(trade)
    )

    # Read back from DB
    triggers = db.get_trigger_orders_for_trade(trade.id)
    assert len(triggers) == 3
    assert triggers[0]["pair"] == "BTC/USDC:USDC"
    assert triggers[0]["status"] == "placed"


def test_active_trigger_orders_query(db, paper_exchange, config):
    import asyncio
    mgr = TriggerOrderManager(paper_exchange, db, config)
    trade = make_trade()
    trade.id = db.save_trade(trade)

    asyncio.get_event_loop().run_until_complete(
        mgr.place_triggers_for_trade(trade)
    )

    active = db.get_active_trigger_orders()
    assert len(active) == 3

    # Cancel and re-check
    asyncio.get_event_loop().run_until_complete(
        mgr.cancel_triggers_for_trade(trade.id)
    )
    active = db.get_active_trigger_orders()
    assert len(active) == 0


# --- Config ---

def test_trigger_order_config_defaults():
    config = TriggerOrderConfig()
    assert config.enabled is True
    assert config.trailing_update_min_interval_sec == 300
    assert config.trailing_update_min_move_pct == 0.5
