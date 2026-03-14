"""Tests for the risk manager."""

from datetime import datetime, timedelta, timezone

import pytest

from src.config import EventBlockingConfig, ProfitSplitConfig, RiskConfig
from src.database import Database
from src.models import PortfolioSnapshot, Trade
from src.risk_manager import RiskManager, _ABSOLUTE_MAX_DRAWDOWN, _ABSOLUTE_MAX_POSITION_PCT, _ABSOLUTE_MAX_AGGREGATE_EXPOSURE


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def risk_config():
    return RiskConfig(
        starting_capital=200.0,
        max_drawdown_pct=0.30,
        max_position_pct=0.10,
        default_stop_loss_pct=0.03,
        max_open_positions=3,
        max_daily_trades=10,
        cooldown_after_loss_min=30,
    )


@pytest.fixture
def risk_manager(risk_config, db):
    return RiskManager(risk_config, db)


@pytest.fixture
def healthy_portfolio():
    return PortfolioSnapshot(
        timestamp="2024-01-01T00:00:00",
        total_value=200.0,
        cash=150.0,
        positions_value=50.0,
        open_positions=1,
        drawdown_pct=0.0,
        high_water_mark=200.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        total_pnl_pct=0.0,
    )


def test_drawdown_ok(risk_manager, healthy_portfolio):
    assert risk_manager.check_drawdown(healthy_portfolio) is True


def test_drawdown_breaker(risk_manager, healthy_portfolio):
    healthy_portfolio.drawdown_pct = 0.31
    assert risk_manager.check_drawdown(healthy_portfolio) is False


def test_drawdown_at_limit(risk_manager, healthy_portfolio):
    healthy_portfolio.drawdown_pct = 0.30
    assert risk_manager.check_drawdown(healthy_portfolio) is False


def test_drawdown_warning_zone(risk_manager, healthy_portfolio):
    """Drawdown at 80% of limit should still allow trading (just warn)."""
    healthy_portfolio.drawdown_pct = 0.24  # 80% of 0.30
    assert risk_manager.check_drawdown(healthy_portfolio) is True


def test_hardcoded_max_cannot_be_exceeded(db):
    """Config cannot set drawdown above the hardcoded 30% limit."""
    config = RiskConfig(max_drawdown_pct=0.50)
    rm = RiskManager(config, db)
    assert rm.max_drawdown == _ABSOLUTE_MAX_DRAWDOWN


def test_hardcoded_max_position_cannot_be_exceeded(db):
    """Config cannot set max position pct above the hardcoded 25% limit."""
    config = RiskConfig(max_position_pct=0.50)
    rm = RiskManager(config, db)
    assert rm.max_position_pct == _ABSOLUTE_MAX_POSITION_PCT


def test_can_trade_basic(risk_manager, healthy_portfolio):
    ok, reason = risk_manager.check_can_trade(healthy_portfolio)
    assert ok is True
    assert reason == ""


def test_can_trade_max_positions(risk_manager, healthy_portfolio):
    healthy_portfolio.open_positions = 3
    ok, reason = risk_manager.check_can_trade(healthy_portfolio)
    assert ok is False
    assert "positions" in reason.lower()


def test_aggregate_exposure_blocks_trade(risk_manager, healthy_portfolio):
    """Aggregate exposure >= 30% of portfolio should block new trades."""
    healthy_portfolio.positions_value = 70.0  # 35% of $200
    healthy_portfolio.cash = 130.0
    ok, reason = risk_manager.check_can_trade(healthy_portfolio)
    assert ok is False
    assert "aggregate exposure" in reason.lower()


def test_aggregate_exposure_allows_below_cap(risk_manager, healthy_portfolio):
    """Aggregate exposure below 30% should allow trades."""
    healthy_portfolio.positions_value = 40.0  # 20% of $200
    healthy_portfolio.cash = 160.0
    ok, reason = risk_manager.check_can_trade(healthy_portfolio)
    assert ok is True


def test_aggregate_exposure_hardcoded_cap():
    """The 30% aggregate exposure cap cannot be changed by config."""
    assert _ABSOLUTE_MAX_AGGREGATE_EXPOSURE == 0.30


def test_can_trade_daily_limit(risk_manager, healthy_portfolio, db):
    """Daily trade limit should block new trades when exhausted."""
    # Simulate 10 trades today by inserting them
    for i in range(10):
        trade = Trade(
            id=None, pair="BTC/USDC:USDC", direction="long",
            entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
            status="closed", entry_time=datetime.now(timezone.utc).isoformat(),
            fees=0.0,
        )
        db.save_trade(trade)

    ok, reason = risk_manager.check_can_trade(healthy_portfolio)
    assert ok is False
    assert "daily" in reason.lower()


def test_can_trade_earnings_blackout(risk_manager, healthy_portfolio):
    """Earnings blackout blocks trades for specific pairs."""
    event_config = EventBlockingConfig(enabled=True)
    risk_manager.set_tradfi_context(
        event_blocking=event_config,
        earnings_blackout_fn=lambda pair: pair == "BTC/USDC:USDC",
    )

    ok, reason = risk_manager.check_can_trade(healthy_portfolio, pair="BTC/USDC:USDC")
    assert ok is False
    assert "blackout" in reason.lower()

    # Other pairs unaffected
    ok2, reason2 = risk_manager.check_can_trade(healthy_portfolio, pair="ETH/USDC:USDC")
    assert ok2 is True


def test_can_trade_correlated_positions(risk_manager, healthy_portfolio, db):
    """Correlated pair positions should count as a single risk unit."""
    # Open 2 positions
    healthy_portfolio.open_positions = 2

    # Simulate an open ETH trade
    trade = Trade(
        id=None, pair="ETH/USDC:USDC", direction="long",
        entry_price=3000.0, quantity=0.1, stop_loss=2900.0,
        status="open", entry_time="2024-01-01", fees=0.0,
    )
    db.save_trade(trade)

    # Set BTC as correlated with ETH
    risk_manager.set_tradfi_context(
        correlated_pairs={"BTC/USDC:USDC": ["ETH/USDC:USDC"]},
    )

    # Trying to open BTC should be blocked because ETH is open and correlated
    ok, reason = risk_manager.check_can_trade(healthy_portfolio, pair="BTC/USDC:USDC")
    assert ok is False
    assert "correlated" in reason.lower()


def test_position_sizing(risk_manager, healthy_portfolio):
    qty = risk_manager.size_position(
        entry_price=50000.0,
        stop_loss_price=49000.0,
        portfolio=healthy_portfolio,
        confidence=0.7,
    )
    assert qty > 0
    # Position value should not exceed max_position_pct of portfolio
    assert qty * 50000 <= 200 * 0.10 * 1.01  # Small tolerance


def test_position_sizing_zero_stop(risk_manager, healthy_portfolio):
    qty = risk_manager.size_position(
        entry_price=50000.0,
        stop_loss_price=50000.0,  # Zero distance
        portfolio=healthy_portfolio,
        confidence=0.7,
    )
    assert qty == 0.0


def test_position_sizing_negative_prices(risk_manager, healthy_portfolio):
    """Invalid negative or zero prices return 0 quantity."""
    qty = risk_manager.size_position(
        entry_price=-100.0,
        stop_loss_price=49000.0,
        portfolio=healthy_portfolio,
        confidence=0.7,
    )
    assert qty == 0.0

    qty2 = risk_manager.size_position(
        entry_price=50000.0,
        stop_loss_price=0.0,
        portfolio=healthy_portfolio,
        confidence=0.7,
    )
    assert qty2 == 0.0


def test_position_sizing_uses_bot_balance(db):
    """Sizing uses bot_balance, not total_value, when available."""
    config = RiskConfig(max_position_pct=0.25)
    rm = RiskManager(config, db)

    portfolio = PortfolioSnapshot(
        timestamp="2024-01-01", total_value=1000.0, cash=800.0,
        positions_value=200.0, open_positions=0, drawdown_pct=0.0,
        high_water_mark=1000.0, daily_pnl=0.0, total_pnl=0.0,
        total_pnl_pct=0.0, bot_balance=700.0, user_balance=300.0,
    )
    qty = rm.size_position(
        entry_price=50000.0,
        stop_loss_price=49000.0,
        portfolio=portfolio,
        confidence=0.8,
    )
    # Max value from bot_balance: 700 * 0.25 = 175
    # Max quantity from bot_balance cap: 175 / 50000 = 0.0035
    assert qty > 0
    assert qty * 50000 <= 700 * 0.25 * 1.01


def test_position_sizing_capped_by_cash(risk_manager):
    """Position cannot exceed 95% of available cash."""
    portfolio = PortfolioSnapshot(
        timestamp="2024-01-01", total_value=200.0, cash=10.0,
        positions_value=190.0, open_positions=1, drawdown_pct=0.0,
        high_water_mark=200.0, daily_pnl=0.0, total_pnl=0.0,
        total_pnl_pct=0.0,
    )
    qty = risk_manager.size_position(
        entry_price=50000.0,
        stop_loss_price=49000.0,
        portfolio=portfolio,
        confidence=0.8,
    )
    # Max from cash: 10 * 0.95 / 50000 = 0.00019
    assert qty * 50000 <= 10.0 * 0.95 * 1.01


def test_position_sizing_confidence_scaling(db):
    """Higher confidence should result in larger position sizes."""
    # Use wide stop distance so position cap doesn't mask confidence effect
    config = RiskConfig(max_position_pct=0.25)
    rm = RiskManager(config, db)
    portfolio = PortfolioSnapshot(
        timestamp="2024-01-01", total_value=100.0, cash=100.0,
        positions_value=0.0, open_positions=0, drawdown_pct=0.0,
        high_water_mark=100.0, daily_pnl=0.0, total_pnl=0.0,
        total_pnl_pct=0.0, bot_balance=100.0,
    )
    qty_low = rm.size_position(
        entry_price=10.0, stop_loss_price=1.0,
        portfolio=portfolio, confidence=0.1,
    )
    qty_high = rm.size_position(
        entry_price=10.0, stop_loss_price=1.0,
        portfolio=portfolio, confidence=0.4,
    )
    assert qty_high > qty_low


def test_position_sizing_iv_adjustment(db):
    """Elevated IV should reduce position size by 50%."""
    # Use wide stop distance so IV reduction isn't masked by position cap
    config = RiskConfig(max_position_pct=0.25)
    portfolio = PortfolioSnapshot(
        timestamp="2024-01-01", total_value=100.0, cash=100.0,
        positions_value=0.0, open_positions=0, drawdown_pct=0.0,
        high_water_mark=100.0, daily_pnl=0.0, total_pnl=0.0,
        total_pnl_pct=0.0, bot_balance=100.0,
    )

    rm_iv = RiskManager(config, db)
    rm_iv.set_tradfi_context(
        iv_context={"TEST/USDC:USDC": {"avg_iv": 0.70}},
    )
    qty_with_iv = rm_iv.size_position(
        entry_price=10.0, stop_loss_price=1.0,
        portfolio=portfolio, confidence=0.2,
        pair="TEST/USDC:USDC",
    )

    rm_no_iv = RiskManager(config, db)
    qty_no_iv = rm_no_iv.size_position(
        entry_price=10.0, stop_loss_price=1.0,
        portfolio=portfolio, confidence=0.2,
        pair="TEST/USDC:USDC",
    )

    # With IV > 0.5, size should be smaller
    assert qty_with_iv < qty_no_iv


def test_stop_loss_long(risk_manager):
    sl = risk_manager.calculate_stop_loss("long", 50000.0, 500.0)
    assert sl < 50000.0
    # min stop = 3% of 50000 = 1500, ATR stop = 2 * 500 = 1000
    # Risk manager takes the larger distance, so stop = 50000 - 1500 = 48500
    assert sl == 50000.0 - 50000.0 * 0.03  # 48500


def test_stop_loss_short(risk_manager):
    sl = risk_manager.calculate_stop_loss("short", 50000.0, 500.0)
    assert sl > 50000.0


def test_stop_loss_atr_dominates(risk_manager):
    """When ATR stop is larger than default pct, ATR should dominate."""
    # ATR stop = 2 * 1500 = 3000 > 3% of 50000 = 1500
    sl = risk_manager.calculate_stop_loss("long", 50000.0, 1500.0)
    assert sl == pytest.approx(50000.0 - 3000.0, abs=0.01)


def test_stop_loss_short_atr_dominates(risk_manager):
    """Short stop loss with ATR dominance."""
    sl = risk_manager.calculate_stop_loss("short", 50000.0, 1500.0)
    assert sl == pytest.approx(50000.0 + 3000.0, abs=0.01)


def test_take_profit(risk_manager):
    tp = risk_manager.calculate_take_profit(50000.0, 49000.0, "long", rr_ratio=2.0)
    assert tp == 52000.0  # 2x the $1000 risk = $2000 profit target

    tp_short = risk_manager.calculate_take_profit(50000.0, 51000.0, "short", rr_ratio=2.0)
    assert tp_short == 48000.0


def test_take_profit_custom_rr_ratio(risk_manager):
    """Custom risk-reward ratio should adjust TP distance."""
    tp = risk_manager.calculate_take_profit(50000.0, 49000.0, "long", rr_ratio=3.0)
    assert tp == 53000.0  # 3x the $1000 risk


# --- Dual take-profit tests ---


def test_dual_take_profits_long(risk_manager):
    """Dual TP for long direction: user TP < bot TP (user is more conservative)."""
    user_tp, bot_tp, user_trail, bot_trail = risk_manager.calculate_dual_take_profits(
        entry_price=50000.0, direction="long"
    )
    # User TP (4%) closer than bot TP (8%)
    assert user_tp < bot_tp
    assert user_tp == pytest.approx(50000 * 1.04, abs=1.0)
    assert bot_tp == pytest.approx(50000 * 1.08, abs=1.0)
    # Trailing stops below entry for longs
    assert user_trail < 50000.0
    assert bot_trail < 50000.0
    # User trail tighter (2%) than bot trail (3.5%)
    assert user_trail > bot_trail


def test_dual_take_profits_short(risk_manager):
    """Dual TP for short direction: user TP > bot TP (closer to entry)."""
    user_tp, bot_tp, user_trail, bot_trail = risk_manager.calculate_dual_take_profits(
        entry_price=50000.0, direction="short"
    )
    # Short TPs are below entry; user TP closer to entry than bot TP
    assert user_tp > bot_tp
    assert user_tp == pytest.approx(50000 * 0.96, abs=1.0)
    assert bot_tp == pytest.approx(50000 * 0.92, abs=1.0)
    # Trailing stops above entry for shorts
    assert user_trail > 50000.0
    assert bot_trail > 50000.0
    # User trail tighter than bot trail
    assert user_trail < bot_trail


def test_dual_take_profits_custom_config(db):
    """Custom profit split config produces expected levels."""
    ps = ProfitSplitConfig(
        user_take_profit_pct=0.05,
        bot_take_profit_pct=0.10,
        user_trailing_stop_pct=0.03,
        bot_trailing_stop_pct=0.05,
    )
    config = RiskConfig(profit_split=ps)
    rm = RiskManager(config, db)

    user_tp, bot_tp, user_trail, bot_trail = rm.calculate_dual_take_profits(
        entry_price=10000.0, direction="long"
    )
    assert user_tp == pytest.approx(10500.0, abs=1.0)
    assert bot_tp == pytest.approx(11000.0, abs=1.0)
    assert user_trail == pytest.approx(9700.0, abs=1.0)
    assert bot_trail == pytest.approx(9500.0, abs=1.0)


# --- check_stop_losses with dual TP ---


def test_check_stop_losses():
    trades = [
        Trade(id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
              quantity=0.01, stop_loss=49000, take_profit=52000, status="open",
              entry_time="2024-01-01"),
    ]
    # Price above stop — no close
    config = RiskConfig()
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)
    result = rm.check_stop_losses(trades, {"BTC/USDC:USDC": 50500})
    assert len(result) == 0

    # Price below stop — close
    result = rm.check_stop_losses(trades, {"BTC/USDC:USDC": 48900})
    assert len(result) == 1
    assert result[0][1] == "stop_loss"

    # Price above take profit — close
    result = rm.check_stop_losses(trades, {"BTC/USDC:USDC": 52100})
    assert len(result) == 1
    assert result[0][1] == "take_profit"


def test_check_stop_losses_short():
    """Short trades trigger on the opposite price direction."""
    config = RiskConfig()
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    trades = [
        Trade(id=1, pair="ETH/USDC:USDC", direction="short", entry_price=3000,
              quantity=0.1, stop_loss=3100, take_profit=2800, status="open",
              entry_time="2024-01-01"),
    ]

    # Price below stop — no close
    result = rm.check_stop_losses(trades, {"ETH/USDC:USDC": 2950})
    assert len(result) == 0

    # Price above stop — close (short gets stopped out when price rises)
    result = rm.check_stop_losses(trades, {"ETH/USDC:USDC": 3150})
    assert len(result) == 1
    assert result[0][1] == "stop_loss"

    # Price below take profit — close (short profits when price drops)
    result = rm.check_stop_losses(trades, {"ETH/USDC:USDC": 2750})
    assert len(result) == 1
    assert result[0][1] == "take_profit"


def test_check_stop_losses_missing_price():
    """Trades with no price in current_prices should be skipped."""
    config = RiskConfig()
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    trades = [
        Trade(id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
              quantity=0.01, stop_loss=49000, status="open", entry_time="2024-01-01"),
    ]

    result = rm.check_stop_losses(trades, {"ETH/USDC:USDC": 3000})  # No BTC price
    assert len(result) == 0


def test_check_stop_losses_dual_tp_user_then_bot():
    """Dual TP: user portion closes first, trade fully closes when both done."""
    config = RiskConfig()
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time="2024-01-01",
        user_take_profit=52000, bot_take_profit=55000,
        user_trailing_stop=49500, bot_trailing_stop=49000,
    )

    # Price hits user TP but not bot TP
    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 52500})
    assert len(result) == 0  # Not fully closed yet
    assert trade.user_portion_closed is True
    assert trade.bot_portion_closed is False

    # Price rises to hit bot TP too
    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 55500})
    assert len(result) == 1
    assert result[0][1] == "take_profit"
    assert trade.bot_portion_closed is True


def test_check_stop_losses_dual_tp_stop_loss_closes_both():
    """Stop loss should close entire position regardless of dual TP state."""
    config = RiskConfig()
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time="2024-01-01",
        user_take_profit=52000, bot_take_profit=55000,
        user_trailing_stop=49500, bot_trailing_stop=49000,
    )

    # Price crashes below stop loss
    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 48500})
    assert len(result) == 1
    assert result[0][1] == "stop_loss"


def test_check_stop_losses_dual_tp_short():
    """Dual TP for short: user portion closes at higher price (closer to entry)."""
    config = RiskConfig()
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    trade = Trade(
        id=1, pair="ETH/USDC:USDC", direction="short", entry_price=3000,
        quantity=0.1, stop_loss=3100, status="open", entry_time="2024-01-01",
        user_take_profit=2880, bot_take_profit=2760,
        user_trailing_stop=3060, bot_trailing_stop=3100,
    )

    # Price drops to user TP
    result = rm.check_stop_losses([trade], {"ETH/USDC:USDC": 2850})
    assert len(result) == 0
    assert trade.user_portion_closed is True

    # Price drops further to bot TP
    result = rm.check_stop_losses([trade], {"ETH/USDC:USDC": 2700})
    assert len(result) == 1
    assert result[0][1] == "take_profit"


def test_check_stop_losses_user_trailing_stop():
    """User trailing stop triggers user portion close."""
    config = RiskConfig()
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=48000, status="open", entry_time="2024-01-01",
        user_take_profit=55000, bot_take_profit=58000,
        user_trailing_stop=50500, bot_trailing_stop=49500,
    )

    # Price drops to user trailing stop (but not below hard stop)
    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 50400})
    assert len(result) == 0
    assert trade.user_portion_closed is True
    assert trade.bot_portion_closed is False


# --- update_trailing_stops tests ---


def test_update_trailing_stops_long_profitable(risk_manager):
    """Trailing stop should ratchet up for profitable long trades."""
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time="2024-01-01",
        user_trailing_stop=49000, bot_trailing_stop=48500,
    )

    updated = risk_manager.update_trailing_stops(
        [trade],
        current_prices={"BTC/USDC:USDC": 52000.0},
        atr_values={"BTC/USDC:USDC": 500.0},
    )
    assert len(updated) == 1
    assert trade.stop_loss > 49000  # Should have ratcheted up


def test_update_trailing_stops_long_not_profitable(risk_manager):
    """Trailing stop should not move for unprofitable trades."""
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time="2024-01-01",
    )
    original_stop = trade.stop_loss

    updated = risk_manager.update_trailing_stops(
        [trade],
        current_prices={"BTC/USDC:USDC": 49500.0},  # Below entry
        atr_values={"BTC/USDC:USDC": 500.0},
    )
    assert len(updated) == 0
    assert trade.stop_loss == original_stop


def test_update_trailing_stops_short_profitable(risk_manager):
    """Trailing stop should ratchet down for profitable short trades."""
    trade = Trade(
        id=1, pair="ETH/USDC:USDC", direction="short", entry_price=3000,
        quantity=0.1, stop_loss=3100, status="open", entry_time="2024-01-01",
        user_trailing_stop=3060, bot_trailing_stop=3105,
    )

    updated = risk_manager.update_trailing_stops(
        [trade],
        current_prices={"ETH/USDC:USDC": 2800.0},
        atr_values={"ETH/USDC:USDC": 50.0},
    )
    assert len(updated) == 1
    assert trade.stop_loss < 3100  # Should have moved down


def test_update_trailing_stops_missing_data(risk_manager):
    """Trades missing price or ATR data should be skipped."""
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time="2024-01-01",
    )

    # Missing price
    updated = risk_manager.update_trailing_stops(
        [trade],
        current_prices={},
        atr_values={"BTC/USDC:USDC": 500.0},
    )
    assert len(updated) == 0

    # Missing ATR
    updated = risk_manager.update_trailing_stops(
        [trade],
        current_prices={"BTC/USDC:USDC": 52000.0},
        atr_values={},
    )
    assert len(updated) == 0


def test_update_trailing_stops_never_moves_backwards(risk_manager):
    """Trailing stop should never move further from price (backwards)."""
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=51000, status="open", entry_time="2024-01-01",
    )

    # Price at 52000 with ATR=500 => trail distance = 750
    # new_stop would be 52000 - 750 = 51250 > 51000 => update
    updated = risk_manager.update_trailing_stops(
        [trade],
        current_prices={"BTC/USDC:USDC": 52000.0},
        atr_values={"BTC/USDC:USDC": 500.0},
    )
    assert len(updated) == 1
    new_stop = trade.stop_loss

    # Now price drops back but is still above entry
    updated = risk_manager.update_trailing_stops(
        [trade],
        current_prices={"BTC/USDC:USDC": 51500.0},
        atr_values={"BTC/USDC:USDC": 500.0},
    )
    # 51500 - 750 = 50750 < new_stop, so stop should NOT move backwards
    assert trade.stop_loss == new_stop


# --- max_hold_hours tests ---


def test_max_hold_time_closes_expired_trade():
    """Trade held longer than max_hold_hours should be auto-closed."""
    config = RiskConfig(max_hold_hours=24)
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    # Trade opened 25 hours ago
    entry_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time=entry_time,
    )

    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 50500})
    assert len(result) == 1
    assert result[0][1] == "max_hold_time"


def test_max_hold_time_keeps_fresh_trade():
    """Trade within max_hold_hours should NOT be auto-closed."""
    config = RiskConfig(max_hold_hours=24)
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    # Trade opened 2 hours ago
    entry_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time=entry_time,
    )

    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 50500})
    assert len(result) == 0


def test_max_hold_time_disabled_by_default():
    """max_hold_hours=0 should disable the time-based exit."""
    config = RiskConfig(max_hold_hours=0)
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    # Trade opened 100 hours ago — should NOT be closed
    entry_time = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time=entry_time,
    )

    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 50500})
    assert len(result) == 0


def test_max_hold_time_takes_priority_over_price_checks():
    """Max hold time should close the trade even if price is fine."""
    config = RiskConfig(max_hold_hours=12)
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    entry_time = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, take_profit=55000, status="open",
        entry_time=entry_time,
        user_take_profit=52000, bot_take_profit=55000,
        user_trailing_stop=49500, bot_trailing_stop=49000,
    )

    # Price is perfectly fine (between stop and TP)
    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 51000})
    assert len(result) == 1
    assert result[0][1] == "max_hold_time"


def test_max_hold_time_malformed_entry_time():
    """Malformed entry_time should not crash — just skip hold check."""
    config = RiskConfig(max_hold_hours=24)
    db_mock = Database(":memory:")
    rm = RiskManager(config, db_mock)

    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
        quantity=0.01, stop_loss=49000, status="open", entry_time="not-a-date",
    )

    # Should not crash, just skip hold check and proceed to price checks
    result = rm.check_stop_losses([trade], {"BTC/USDC:USDC": 50500})
    assert len(result) == 0
