"""Tests for the executor — paper trading simulation, trigger orders, error handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import ExchangeConfig, RiskConfig
from src.database import Database
from src.exchange import Exchange
from src.executor import Executor
from src.journal import Journal
from src.models import PortfolioSnapshot, Signal, Trade
from src.risk_manager import RiskManager


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def exchange():
    return Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=200.0)


@pytest.fixture
def risk_manager(db):
    return RiskManager(RiskConfig(), db)


@pytest.fixture
def journal(db):
    return Journal(db)


@pytest.fixture
def executor(exchange, risk_manager, db, journal):
    return Executor(exchange, risk_manager, db, journal)


@pytest.fixture
def healthy_portfolio():
    return PortfolioSnapshot(
        timestamp="2024-01-01T00:00:00",
        total_value=1000.0,
        cash=800.0,
        positions_value=200.0,
        open_positions=1,
        drawdown_pct=0.05,
        high_water_mark=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        total_pnl_pct=0.0,
        bot_balance=700.0,
        user_balance=300.0,
    )


@pytest.fixture
def sample_signal():
    return Signal(
        pair="BTC/USDC:USDC",
        timeframe="15m",
        direction="long",
        confidence=0.75,
        strategy_name="momentum",
        indicators={"rsi": 28.0, "macd_hist": 0.001},
        reasoning="RSI oversold + MACD turning up",
        timestamp="2024-01-01T00:00:00",
    )


def test_paper_balance_init(exchange):
    assert exchange.paper_balance["USDC"] == 200.0


@pytest.mark.asyncio
async def test_paper_market_order(exchange):
    """Paper market orders should update balances."""
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    exchange.fetch_ticker = mock_ticker

    order = await exchange.create_market_order("BTC/USDC:USDC", "buy", 0.001)

    assert order["status"] == "closed"
    assert order["side"] == "buy"
    assert exchange.paper_balance["BTC"] == 0.001
    assert exchange.paper_balance["USDC"] < 200.0  # Spent some USDC


@pytest.mark.asyncio
async def test_paper_insufficient_balance(exchange):
    """Should raise on insufficient balance."""
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker

    with pytest.raises(Exception, match="Insufficient"):
        await exchange.create_market_order("BTC/USDC:USDC", "buy", 1.0)  # Way too much


def test_paper_portfolio_value(exchange):
    exchange.paper_balance = {"USDC": 100.0, "BTC": 0.001}
    value = exchange.get_paper_portfolio_value({"BTC/USDC:USDC": 50000.0})
    assert value == pytest.approx(150.0, abs=0.01)


# --- execute_signal tests ---


@pytest.mark.asyncio
async def test_execute_signal_risk_block(executor, sample_signal, healthy_portfolio):
    """When risk manager blocks trading, execute_signal returns None."""
    healthy_portfolio.drawdown_pct = 0.35  # Exceeds 30% limit
    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI says go",
        atr_value=500.0,
    )
    assert result is None


@pytest.mark.asyncio
async def test_execute_signal_no_price(executor, sample_signal, healthy_portfolio):
    """When exchange returns no price, execute_signal returns None."""
    async def mock_ticker(pair):
        return {"last": None, "bid": None, "ask": None}

    executor.exchange.fetch_ticker = mock_ticker
    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI says go",
        atr_value=500.0,
    )
    assert result is None


@pytest.mark.asyncio
async def test_execute_signal_dry_run(executor, sample_signal, healthy_portfolio):
    """Dry run should log but not place any order, returns None."""
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    executor.exchange.fetch_ticker = mock_ticker
    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI says go",
        atr_value=500.0,
        dry_run=True,
    )
    assert result is None


@pytest.mark.asyncio
async def test_execute_signal_below_minimum_order(executor, healthy_portfolio):
    """Quantity * price < $5 should be rejected."""
    signal = Signal(
        pair="BTC/USDC:USDC",
        timeframe="15m",
        direction="long",
        confidence=0.001,
        strategy_name="momentum",
        reasoning="test",
        timestamp="2024-01-01",
    )
    healthy_portfolio.cash = 3.0
    healthy_portfolio.bot_balance = 3.0
    healthy_portfolio.total_value = 3.0

    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    executor.exchange.fetch_ticker = mock_ticker
    result = await executor.execute_signal(
        signal=signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI test",
        atr_value=500.0,
    )
    assert result is None


@pytest.mark.asyncio
async def test_execute_signal_success_paper_mode(executor, sample_signal, healthy_portfolio):
    """Successful execution in paper mode returns a Trade with filled data."""
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    executor.exchange.fetch_ticker = mock_ticker

    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI approves",
        atr_value=500.0,
    )
    assert result is not None
    assert result.pair == "BTC/USDC:USDC"
    assert result.direction == "long"
    assert result.status == "open"
    assert result.id is not None
    assert result.user_take_profit is not None
    assert result.bot_take_profit is not None


@pytest.mark.asyncio
async def test_execute_signal_order_failure(executor, sample_signal, healthy_portfolio):
    """When exchange raises during order, execute_signal returns None gracefully."""
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    async def mock_order_fail(pair, side, qty):
        raise Exception("Exchange connection timeout")

    executor.exchange.fetch_ticker = mock_ticker
    executor.exchange.create_market_order = mock_order_fail

    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI approves",
        atr_value=500.0,
    )
    assert result is None


@pytest.mark.asyncio
async def test_execute_signal_with_router(executor, sample_signal, healthy_portfolio):
    """When a router is provided, orders go through the router."""
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    executor.exchange.fetch_ticker = mock_ticker

    mock_router = AsyncMock()
    mock_router.plan_execution.return_value = MagicMock()
    mock_router.execute.return_value = {
        "price": 50005.0,
        "fee": 0.50,
        "execution_venue": "dydx",
        "slippage_actual_bps": 1.5,
    }
    executor.router = mock_router

    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI approves",
        atr_value=500.0,
    )
    assert result is not None
    assert result.execution_venue == "dydx"
    assert result.slippage_actual_bps == 1.5
    assert result.entry_price == 50005.0
    mock_router.plan_execution.assert_called_once()
    mock_router.execute.assert_called_once()


@pytest.mark.asyncio
async def test_execute_signal_trigger_order_placement(executor, sample_signal, healthy_portfolio):
    """When trigger manager is present, triggers are placed after order."""
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    executor.exchange.fetch_ticker = mock_ticker

    mock_trigger = AsyncMock()
    mock_trigger.place_triggers_for_trade.return_value = []
    executor.trigger_manager = mock_trigger

    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI approves",
        atr_value=500.0,
    )
    assert result is not None
    mock_trigger.place_triggers_for_trade.assert_called_once()


@pytest.mark.asyncio
async def test_execute_signal_trigger_failure_non_fatal(executor, sample_signal, healthy_portfolio):
    """Trigger order failure should not prevent trade from being returned."""
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999, "ask": 50001}

    executor.exchange.fetch_ticker = mock_ticker

    mock_trigger = AsyncMock()
    mock_trigger.place_triggers_for_trade.side_effect = Exception("Trigger API down")
    executor.trigger_manager = mock_trigger

    result = await executor.execute_signal(
        signal=sample_signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI approves",
        atr_value=500.0,
    )
    assert result is not None
    assert result.status == "open"


@pytest.mark.asyncio
async def test_execute_signal_short_direction(executor, healthy_portfolio):
    """Short signal uses 'sell' side for order placement."""
    signal = Signal(
        pair="ETH/USDC:USDC",
        timeframe="15m",
        direction="short",
        confidence=0.75,
        strategy_name="momentum",
        reasoning="Overbought",
        timestamp="2024-01-01",
    )

    async def mock_ticker(pair):
        return {"last": 3000.0, "bid": 2999, "ask": 3001}

    order_sides = []

    async def mock_order(pair, side, qty):
        order_sides.append(side)
        return {"price": 3000.0, "fee": 0.10, "status": "closed"}

    executor.exchange.fetch_ticker = mock_ticker
    executor.exchange.create_market_order = mock_order

    result = await executor.execute_signal(
        signal=signal,
        portfolio=healthy_portfolio,
        ai_reasoning="AI approves short",
        atr_value=50.0,
    )
    assert result is not None
    assert result.direction == "short"
    assert order_sides == ["sell"]


# --- close_trade tests ---


@pytest.mark.asyncio
async def test_close_trade_long_profit(executor, db):
    """Closing a profitable long trade calculates correct P&L."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        take_profit=52000.0, status="open", entry_time="2024-01-01", fees=0.50,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 51000.0, "fee": 0.50, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "take_profit", current_price=51000.0)
    assert result.status == "take_profit"
    assert result.exit_price == 51000.0
    # P&L = (51000 - 50000) * 0.01 - 0.50 - 0.50 = 10 - 1 = 9.0
    assert result.pnl == pytest.approx(9.0, abs=0.01)


@pytest.mark.asyncio
async def test_close_trade_short_profit(executor, db):
    """Closing a profitable short trade calculates correct P&L."""
    trade = Trade(
        id=None, pair="ETH/USDC:USDC", direction="short",
        entry_price=3000.0, quantity=0.1, stop_loss=3100.0,
        status="open", entry_time="2024-01-01", fees=0.30,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 2900.0, "fee": 0.30, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "take_profit", current_price=2900.0)
    assert result.status == "take_profit"
    # P&L = (3000 - 2900) * 0.1 - 0.30 - 0.30 = 10 - 0.60 = 9.40
    assert result.pnl == pytest.approx(9.40, abs=0.01)


@pytest.mark.asyncio
async def test_close_trade_long_loss(executor, db):
    """Closing a losing long trade results in negative P&L."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.50,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 49000.0, "fee": 0.50, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "stop_loss", current_price=49000.0)
    assert result.status == "stop_loss"
    assert result.pnl == pytest.approx(-11.0, abs=0.01)
    assert result.pnl < 0


@pytest.mark.asyncio
async def test_close_trade_fetches_price_if_not_provided(executor, db):
    """When current_price is None, close_trade fetches it from exchange."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    ticker_called = False

    async def mock_ticker(pair):
        nonlocal ticker_called
        ticker_called = True
        return {"last": 50500.0}

    async def mock_order(pair, side, qty):
        return {"price": 50500.0, "fee": 0.0, "status": "closed"}

    executor.exchange.fetch_ticker = mock_ticker
    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "closed")
    assert ticker_called
    assert result.exit_price == 50500.0


@pytest.mark.asyncio
async def test_close_trade_exchange_failure_returns_open_trade(executor, db):
    """If exchange fails to close, trade stays in original state."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order_fail(pair, side, qty):
        raise Exception("Network error")

    executor.exchange.create_market_order = mock_order_fail

    result = await executor.close_trade(trade, "stop_loss", current_price=49000.0)
    assert result.status == "open"
    assert result.exit_price is None


@pytest.mark.asyncio
async def test_close_trade_with_profit_split(executor, db):
    """When portfolio is present, profit split is applied."""
    mock_portfolio = MagicMock()
    mock_portfolio.apply_profit_split.return_value = (7.0, 3.0)
    executor.portfolio = mock_portfolio

    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 51000.0, "fee": 0.0, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "take_profit", current_price=51000.0)
    assert result.profit_split_bot == 7.0
    assert result.profit_split_user == 3.0
    mock_portfolio.apply_profit_split.assert_called_once()


@pytest.mark.asyncio
async def test_close_trade_without_portfolio(executor, db):
    """Without portfolio, all profit goes to bot."""
    executor.portfolio = None

    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 51000.0, "fee": 0.0, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "take_profit", current_price=51000.0)
    assert result.profit_split_bot == result.pnl
    assert result.profit_split_user == 0.0


@pytest.mark.asyncio
async def test_close_trade_cancels_triggers(executor, db):
    """On close, trigger orders for the trade are cancelled."""
    mock_trigger = AsyncMock()
    mock_trigger.cancel_triggers_for_trade.return_value = 2
    executor.trigger_manager = mock_trigger

    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 50500.0, "fee": 0.0, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    await executor.close_trade(trade, "closed", current_price=50500.0)
    mock_trigger.cancel_triggers_for_trade.assert_called_once_with(trade.id)


@pytest.mark.asyncio
async def test_close_trade_trigger_cancel_failure_non_fatal(executor, db):
    """Trigger cancellation failure should not prevent trade close."""
    mock_trigger = AsyncMock()
    mock_trigger.cancel_triggers_for_trade.side_effect = Exception("Cancel API down")
    executor.trigger_manager = mock_trigger

    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 50500.0, "fee": 0.0, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "closed", current_price=50500.0)
    assert result.status == "closed"


@pytest.mark.asyncio
async def test_close_trade_updates_review_counter(executor, db):
    """Closing a trade increments the trades_since_review counter."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 50500.0, "fee": 0.0, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    await executor.close_trade(trade, "closed", current_price=50500.0)
    count = db.get_state("trades_since_review")
    assert count == "1"

    trade2 = Trade(
        id=None, pair="ETH/USDC:USDC", direction="short",
        entry_price=3000.0, quantity=0.1, stop_loss=3100.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade2.id = db.save_trade(trade2)

    await executor.close_trade(trade2, "closed", current_price=2900.0)
    count = db.get_state("trades_since_review")
    assert count == "2"


@pytest.mark.asyncio
async def test_close_trade_pnl_pct(executor, db):
    """P&L percentage is correctly computed relative to position value."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 55000.0, "fee": 0.0, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    result = await executor.close_trade(trade, "take_profit", current_price=55000.0)
    # pnl_pct = 50 / (50000 * 0.01) = 0.10
    assert result.pnl_pct == pytest.approx(0.10, abs=0.001)


# --- check_and_close_stops / emergency_close_all tests ---


@pytest.mark.asyncio
async def test_check_and_close_stops(executor, db):
    """check_and_close_stops finds and closes trades that hit stop losses."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        take_profit=52000.0, status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    async def mock_order(pair, side, qty):
        return {"price": 48500.0, "fee": 0.0, "status": "closed"}

    executor.exchange.create_market_order = mock_order

    closed = await executor.check_and_close_stops({"BTC/USDC:USDC": 48500.0})
    assert len(closed) == 1
    assert closed[0].status == "stop_loss"


@pytest.mark.asyncio
async def test_check_and_close_stops_no_hits(executor, db):
    """When no stops are hit, nothing is closed."""
    trade = Trade(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        take_profit=52000.0, status="open", entry_time="2024-01-01", fees=0.0,
    )
    trade.id = db.save_trade(trade)

    closed = await executor.check_and_close_stops({"BTC/USDC:USDC": 50500.0})
    assert len(closed) == 0


@pytest.mark.asyncio
async def test_emergency_close_all(executor, db):
    """Emergency close all should close every open trade."""
    for i in range(3):
        trade = Trade(
            id=None, pair="BTC/USDC:USDC", direction="long",
            entry_price=50000.0 + i * 100, quantity=0.01, stop_loss=49000.0,
            status="open", entry_time="2024-01-01", fees=0.0,
        )
        db.save_trade(trade)

    async def mock_ticker(pair):
        return {"last": 48000.0}

    async def mock_order(pair, side, qty):
        return {"price": 48000.0, "fee": 0.0, "status": "closed"}

    executor.exchange.fetch_ticker = mock_ticker
    executor.exchange.create_market_order = mock_order

    closed = await executor.emergency_close_all("Circuit breaker")
    assert len(closed) == 3
    for trade in closed:
        assert trade.status == "emergency_close"
