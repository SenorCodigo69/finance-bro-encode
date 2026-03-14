"""Tests for data models."""

from dataclasses import fields

from src.models import OHLCV, PortfolioSnapshot, Signal, Trade


# --- Signal tests ---


def test_signal_defaults():
    sig = Signal(
        pair="BTC/USDC:USDC",
        timeframe="1h",
        direction="long",
        confidence=0.8,
        strategy_name="trend_following",
    )
    assert sig.indicators == {}
    assert sig.reasoning == ""
    assert sig.timestamp == ""


def test_signal_with_all_fields():
    sig = Signal(
        pair="ETH/USDC:USDC",
        timeframe="15m",
        direction="short",
        confidence=0.65,
        strategy_name="mean_reversion",
        indicators={"rsi": 72.5, "bb_upper": 3200.0},
        reasoning="RSI overbought; Price at upper BB",
        timestamp="2024-01-15T12:00:00",
    )
    assert sig.pair == "ETH/USDC:USDC"
    assert sig.direction == "short"
    assert sig.indicators["rsi"] == 72.5
    assert "overbought" in sig.reasoning


def test_signal_confidence_is_mutable():
    """Signal confidence should be mutable (used by regime/funding filters)."""
    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="1h", direction="long",
        confidence=0.5, strategy_name="momentum",
    )
    sig.confidence = 0.85
    assert sig.confidence == 0.85


def test_signal_indicators_dict_independence():
    """Each Signal instance should have its own indicators dict."""
    sig1 = Signal(pair="A", timeframe="1h", direction="long",
                  confidence=0.5, strategy_name="x")
    sig2 = Signal(pair="B", timeframe="1h", direction="long",
                  confidence=0.5, strategy_name="x")
    sig1.indicators["key"] = "value"
    assert "key" not in sig2.indicators


# --- Trade tests ---


def test_trade_defaults():
    t = Trade(id=None, pair="ETH/USDC:USDC", direction="long", entry_price=3000.0, quantity=0.1, stop_loss=2900.0)
    assert t.status == "open"
    assert t.pnl is None
    assert t.fees == 0.0
    assert t.bot_take_profit is None
    assert t.user_portion_closed is False
    assert t.profit_split_bot is None


def test_trade_with_dual_tp():
    t = Trade(
        id=1, pair="BTC/USDC:USDC", direction="short", entry_price=50000.0,
        quantity=0.01, stop_loss=51000.0,
        user_take_profit=48000.0, bot_take_profit=46000.0,
        user_trailing_stop=51000.0, bot_trailing_stop=51500.0,
        profit_split_bot=5.0, profit_split_user=3.0,
    )
    assert t.user_take_profit == 48000.0
    assert t.bot_take_profit == 46000.0
    assert t.profit_split_bot == 5.0


def test_trade_execution_venue_default():
    """Trade default execution venue should be hyperliquid."""
    t = Trade(id=None, pair="BTC/USDC:USDC", direction="long",
              entry_price=50000.0, quantity=0.01, stop_loss=49000.0)
    assert t.execution_venue == "hyperliquid"
    assert t.slippage_actual_bps is None
    assert t.trigger_orders_placed is False


def test_trade_all_close_statuses():
    """Trade can have various status values."""
    for status in ["open", "closed", "stopped_out", "cancelled", "take_profit", "emergency_close"]:
        t = Trade(id=1, pair="BTC/USDC:USDC", direction="long",
                  entry_price=50000.0, quantity=0.01, stop_loss=49000.0, status=status)
        assert t.status == status


def test_trade_signal_data_independence():
    """Each Trade should have its own signal_data dict."""
    t1 = Trade(id=1, pair="A", direction="long", entry_price=100, quantity=1, stop_loss=90)
    t2 = Trade(id=2, pair="B", direction="long", entry_price=100, quantity=1, stop_loss=90)
    t1.signal_data["key"] = "value"
    assert "key" not in t2.signal_data


def test_trade_portion_closed_flags():
    """Portion closed flags should be independently settable."""
    t = Trade(id=1, pair="BTC/USDC:USDC", direction="long",
              entry_price=50000.0, quantity=0.01, stop_loss=49000.0)
    assert not t.user_portion_closed
    assert not t.bot_portion_closed

    t.user_portion_closed = True
    assert t.user_portion_closed
    assert not t.bot_portion_closed

    t.bot_portion_closed = True
    assert t.user_portion_closed
    assert t.bot_portion_closed


def test_trade_has_expected_fields():
    """Verify Trade has all required fields for the trading system."""
    field_names = {f.name for f in fields(Trade)}
    required_fields = {
        "id", "pair", "direction", "entry_price", "quantity", "stop_loss",
        "take_profit", "exit_price", "status", "pnl", "pnl_pct",
        "entry_time", "exit_time", "signal_data", "ai_reasoning",
        "fees", "bot_take_profit", "user_take_profit",
        "bot_trailing_stop", "user_trailing_stop",
        "user_portion_closed", "bot_portion_closed",
        "profit_split_bot", "profit_split_user",
        "execution_venue", "slippage_actual_bps", "trigger_orders_placed",
    }
    assert required_fields.issubset(field_names)


# --- PortfolioSnapshot tests ---


def test_portfolio_snapshot_defaults():
    snap = PortfolioSnapshot(
        timestamp="2026-03-12T00:00:00",
        total_value=1000.0, cash=800.0, positions_value=200.0,
        open_positions=1, drawdown_pct=0.05, high_water_mark=1050.0,
        daily_pnl=10.0, total_pnl=50.0, total_pnl_pct=0.05,
    )
    assert snap.bot_balance == 0.0
    assert snap.user_balance == 0.0


def test_portfolio_snapshot_with_balances():
    snap = PortfolioSnapshot(
        timestamp="2024-01-01", total_value=1000.0, cash=600.0,
        positions_value=400.0, open_positions=2, drawdown_pct=0.10,
        high_water_mark=1100.0, daily_pnl=-5.0, total_pnl=-100.0,
        total_pnl_pct=-0.10, bot_balance=700.0, user_balance=300.0,
    )
    assert snap.bot_balance == 700.0
    assert snap.user_balance == 300.0
    assert snap.bot_balance + snap.user_balance == snap.total_value


def test_portfolio_snapshot_zero_state():
    """Portfolio at initialization (no positions, no P&L)."""
    snap = PortfolioSnapshot(
        timestamp="2024-01-01", total_value=1000.0, cash=1000.0,
        positions_value=0.0, open_positions=0, drawdown_pct=0.0,
        high_water_mark=1000.0, daily_pnl=0.0, total_pnl=0.0,
        total_pnl_pct=0.0,
    )
    assert snap.open_positions == 0
    assert snap.positions_value == 0.0
    assert snap.drawdown_pct == 0.0


# --- OHLCV tests ---


def test_ohlcv():
    candle = OHLCV(timestamp=1000, open=100.0, high=110.0, low=90.0, close=105.0, volume=500.0)
    assert candle.high > candle.low


def test_ohlcv_all_fields():
    candle = OHLCV(
        timestamp=1710000000000,
        open=50000.0, high=51000.0, low=49500.0,
        close=50800.0, volume=12345.6,
    )
    assert candle.timestamp == 1710000000000
    assert candle.open == 50000.0
    assert candle.close == 50800.0
    assert candle.volume > 0
