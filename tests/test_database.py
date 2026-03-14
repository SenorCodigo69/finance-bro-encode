"""Tests for the database layer."""

import time
from datetime import datetime, timezone

import pytest

from src.database import Database
from src.models import PortfolioSnapshot, Signal, Trade


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


def _make_trade(**overrides) -> Trade:
    defaults = dict(
        id=None,
        pair="BTC/USDC:USDC",
        direction="long",
        entry_price=50000.0,
        quantity=0.01,
        stop_loss=49000.0,
        take_profit=52000.0,
        status="open",
        pnl=None,
        pnl_pct=None,
        fees=0.5,
        entry_time="2026-03-12T10:00:00",
        exit_time=None,
        signal_data={"rsi": 35},
        ai_reasoning="Looks bullish",
        review_notes=None,
        bot_take_profit=53000.0,
        user_take_profit=51500.0,
        bot_trailing_stop=48000.0,
        user_trailing_stop=49500.0,
        user_portion_closed=False,
        bot_portion_closed=False,
        profit_split_bot=None,
        profit_split_user=None,
    )
    defaults.update(overrides)
    return Trade(**defaults)


def _make_signal(**overrides) -> Signal:
    defaults = dict(
        pair="ETH/USDC:USDC",
        timeframe="1h",
        direction="long",
        confidence=0.75,
        strategy_name="rsi_mean_reversion",
        indicators={"rsi": 28.5, "bb_pct": 0.1},
        reasoning="RSI oversold bounce",
        timestamp="2026-03-12T11:00:00",
    )
    defaults.update(overrides)
    return Signal(**defaults)


def _make_snapshot(**overrides) -> PortfolioSnapshot:
    defaults = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_value=1050.0,
        cash=800.0,
        positions_value=250.0,
        open_positions=1,
        drawdown_pct=0.0,
        high_water_mark=1050.0,
        daily_pnl=50.0,
        total_pnl=50.0,
        total_pnl_pct=5.0,
        bot_balance=525.0,
        user_balance=525.0,
    )
    defaults.update(overrides)
    return PortfolioSnapshot(**defaults)


# ── Table creation ──────────────────────────────────────────────────


def test_init_tables_creates_expected_tables(db):
    cursor = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row["name"] for row in cursor.fetchall()}
    expected = {
        "trades",
        "signals",
        "portfolio_snapshots",
        "ai_reviews",
        "agent_state",
        "model_accuracy",
        "market_data_cache",
    }
    assert expected.issubset(tables)


# ── Trades round-trip ───────────────────────────────────────────────


def test_save_and_get_open_trades(db):
    trade = _make_trade()
    tid = db.save_trade(trade)
    assert tid is not None and tid > 0

    open_trades = db.get_open_trades()
    assert len(open_trades) == 1
    t = open_trades[0]
    assert t.id == tid
    assert t.pair == "BTC/USDC:USDC"
    assert t.direction == "long"
    assert t.entry_price == 50000.0
    assert t.quantity == 0.01
    assert t.stop_loss == 49000.0
    assert t.take_profit == 52000.0
    assert t.status == "open"
    assert t.signal_data == {"rsi": 35}
    assert t.ai_reasoning == "Looks bullish"
    assert t.bot_take_profit == 53000.0
    assert t.user_take_profit == 51500.0


def test_get_recent_trades(db):
    for i in range(5):
        db.save_trade(_make_trade(entry_time=f"2026-03-12T1{i}:00:00"))
    recent = db.get_recent_trades(3)
    assert len(recent) == 3
    # Most recent first (ORDER BY id DESC)
    assert recent[0].id > recent[1].id > recent[2].id


def test_update_trade(db):
    trade = _make_trade()
    db.save_trade(trade)

    trade.status = "closed"
    trade.exit_price = 52000.0
    trade.pnl = 20.0
    trade.pnl_pct = 4.0
    trade.exit_time = "2026-03-12T14:00:00"
    trade.profit_split_bot = 10.0
    trade.profit_split_user = 10.0
    trade.user_portion_closed = True
    trade.bot_portion_closed = True
    db.update_trade(trade)

    # Should no longer appear in open trades
    assert len(db.get_open_trades()) == 0

    recent = db.get_recent_trades(1)
    assert len(recent) == 1
    closed = recent[0]
    assert closed.status == "closed"
    assert closed.exit_price == 52000.0
    assert closed.pnl == 20.0
    assert closed.profit_split_bot == 10.0
    assert closed.profit_split_user == 10.0
    assert closed.user_portion_closed is True
    assert closed.bot_portion_closed is True


def test_signal_data_json_roundtrip(db):
    data = {"rsi": 30, "macd_hist": -0.5, "nested": {"a": 1}}
    trade = _make_trade(signal_data=data)
    db.save_trade(trade)
    loaded = db.get_recent_trades(1)[0]
    assert loaded.signal_data == data


# ── Signals ─────────────────────────────────────────────────────────


def test_save_signal(db):
    sig = _make_signal()
    db.save_signal(sig, acted_on=True, reason="")

    row = db.conn.execute("SELECT * FROM signals").fetchone()
    assert row["pair"] == "ETH/USDC:USDC"
    assert row["direction"] == "long"
    assert row["confidence"] == 0.75
    assert row["strategy_name"] == "rsi_mean_reversion"
    assert row["acted_on"] == 1
    assert row["block_reason"] == ""


def test_save_signal_blocked(db):
    sig = _make_signal()
    db.save_signal(sig, acted_on=False, reason="drawdown breaker")

    row = db.conn.execute("SELECT * FROM signals").fetchone()
    assert row["acted_on"] == 0
    assert row["block_reason"] == "drawdown breaker"


# ── Agent state (key-value) ─────────────────────────────────────────


def test_get_state_missing_key(db):
    assert db.get_state("nonexistent") is None


def test_set_and_get_state(db):
    db.set_state("last_cycle", "2026-03-12T10:00:00")
    assert db.get_state("last_cycle") == "2026-03-12T10:00:00"


def test_set_state_overwrites(db):
    db.set_state("counter", "1")
    db.set_state("counter", "2")
    assert db.get_state("counter") == "2"

    # Exactly one row
    count = db.conn.execute("SELECT COUNT(*) as c FROM agent_state WHERE key='counter'").fetchone()["c"]
    assert count == 1


# ── Portfolio snapshots / equity curve ──────────────────────────────


def test_save_snapshot_and_get_equity_curve(db):
    snap = _make_snapshot()
    db.save_snapshot(snap)

    curve = db.get_equity_curve(days=1)
    assert len(curve) == 1
    s = curve[0]
    assert s.total_value == 1050.0
    assert s.cash == 800.0
    assert s.bot_balance == 525.0
    assert s.user_balance == 525.0


def test_equity_curve_filters_old_snapshots(db):
    # Snapshot from far in the past — should be excluded with days=1
    old = _make_snapshot(timestamp="2020-01-01T00:00:00")
    db.save_snapshot(old)

    curve = db.get_equity_curve(days=1)
    assert len(curve) == 0


# ── Market data cache ───────────────────────────────────────────────


def test_cache_ohlcv_and_retrieve(db):
    now_ms = int(time.time() * 1000)
    candles = [
        (now_ms - 60000, 50000, 50100, 49900, 50050, 100.0),
        (now_ms, 50050, 50200, 50000, 50150, 120.0),
    ]
    db.cache_ohlcv("BTC/USDC:USDC", "1m", candles)

    cached = db.get_cached_ohlcv("BTC/USDC:USDC", "1m", now_ms - 120000)
    assert len(cached) == 2
    assert cached[0][0] == now_ms - 60000
    assert cached[1][4] == 50150  # close


def test_get_cached_ohlcv_filters_by_since(db):
    now_ms = int(time.time() * 1000)
    candles = [
        (now_ms - 120000, 50000, 50100, 49900, 50050, 100.0),
        (now_ms, 50050, 50200, 50000, 50150, 120.0),
    ]
    db.cache_ohlcv("BTC/USDC:USDC", "1m", candles)

    # Only the second candle should match
    cached = db.get_cached_ohlcv("BTC/USDC:USDC", "1m", now_ms - 1000)
    assert len(cached) == 1
    assert cached[0][0] == now_ms


def test_cache_ohlcv_upserts(db):
    now_ms = int(time.time() * 1000)
    candles = [(now_ms, 50000, 50100, 49900, 50050, 100.0)]
    db.cache_ohlcv("BTC/USDC:USDC", "1m", candles)

    # Insert again with updated close
    updated = [(now_ms, 50000, 50100, 49900, 50200, 110.0)]
    db.cache_ohlcv("BTC/USDC:USDC", "1m", updated)

    cached = db.get_cached_ohlcv("BTC/USDC:USDC", "1m", 0)
    assert len(cached) == 1
    assert cached[0][4] == 50200  # Updated close


def test_cleanup_cache_removes_old_entries(db):
    old_ms = int((time.time() - 200 * 3600) * 1000)  # 200 hours ago
    recent_ms = int(time.time() * 1000)
    candles = [
        (old_ms, 50000, 50100, 49900, 50050, 100.0),
        (recent_ms, 51000, 51100, 50900, 51050, 200.0),
    ]
    db.cache_ohlcv("BTC/USDC:USDC", "1h", candles)

    db.cleanup_cache(max_age_hours=72)

    remaining = db.get_cached_ohlcv("BTC/USDC:USDC", "1h", 0)
    assert len(remaining) == 1
    assert remaining[0][0] == recent_ms


# ── Model accuracy ──────────────────────────────────────────────────


def test_save_and_get_model_outcomes(db):
    db.save_model_outcome(
        signal_id="sig_001",
        pair="BTC/USDC:USDC",
        direction="long",
        model_name="claude-opus",
        model_vote=True,
        actual_profitable=True,
        was_correct=True,
        timestamp="2026-03-12T10:00:00",
    )
    db.save_model_outcome(
        signal_id="sig_001",
        pair="BTC/USDC:USDC",
        direction="long",
        model_name="gpt-4o",
        model_vote=False,
        actual_profitable=True,
        was_correct=False,
        timestamp="2026-03-12T10:00:00",
    )

    outcomes = db.get_model_outcomes(limit=10)
    assert len(outcomes) == 2

    # Most recent first
    gpt_outcome = outcomes[0]
    assert gpt_outcome["model_name"] == "gpt-4o"
    assert gpt_outcome["model_vote"] is False
    assert gpt_outcome["was_correct"] is False

    claude_outcome = outcomes[1]
    assert claude_outcome["model_name"] == "claude-opus"
    assert claude_outcome["model_vote"] is True
    assert claude_outcome["was_correct"] is True


def test_get_model_outcomes_respects_limit(db):
    for i in range(5):
        db.save_model_outcome(
            signal_id=f"sig_{i:03d}",
            pair="BTC/USDC:USDC",
            direction="long",
            model_name="model_a",
            model_vote=True,
            actual_profitable=True,
            was_correct=True,
            timestamp=f"2026-03-12T1{i}:00:00",
        )
    assert len(db.get_model_outcomes(limit=3)) == 3


# ── Migration ───────────────────────────────────────────────────────


def test_migrate_profit_split_idempotent(tmp_path):
    """Running the migration twice should not error."""
    db_path = str(tmp_path / "migrate.db")
    db1 = Database(db_path)
    db1.close()

    # Re-open — migration runs again in __init__
    db2 = Database(db_path)
    # Verify migrated columns exist by inserting a trade that uses them
    trade = _make_trade(profit_split_bot=5.0, profit_split_user=5.0)
    db2.save_trade(trade)
    loaded = db2.get_recent_trades(1)[0]
    assert loaded.profit_split_bot == 5.0
    db2.close()


# ── get_trade_count_today ───────────────────────────────────────────


def test_get_trade_count_today(db):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    yesterday_str = "2020-01-01T10:00:00"

    db.save_trade(_make_trade(entry_time=today_str))
    db.save_trade(_make_trade(entry_time=today_str))
    db.save_trade(_make_trade(entry_time=yesterday_str))

    assert db.get_trade_count_today() == 2


# ── Edge cases ──────────────────────────────────────────────────────


def test_empty_database_queries(db):
    assert db.get_open_trades() == []
    assert db.get_recent_trades(10) == []
    assert db.get_equity_curve(days=30) == []
    assert db.get_model_outcomes() == []
    assert db.get_trade_count_today() == 0


def test_in_memory_database():
    """Database works with :memory: path (no file created)."""
    db = Database(":memory:")
    db.set_state("key", "value")
    assert db.get_state("key") == "value"
    db.close()
