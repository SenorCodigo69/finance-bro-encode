"""Tests for the trade journal."""

import csv
from pathlib import Path

import pytest

from src.database import Database
from src.journal import Journal
from src.models import Signal, Trade


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def journal(db):
    return Journal(db)


def _make_trade(**kwargs):
    defaults = dict(
        id=None, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
        entry_time="2026-03-12T00:00:00", status="open",
    )
    defaults.update(kwargs)
    return Trade(**defaults)


def _make_signal(**kwargs):
    defaults = dict(
        pair="BTC/USDC:USDC", timeframe="1h", direction="long",
        confidence=0.75, strategy_name="trend_following",
        timestamp="2026-03-12T00:00:00",
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def test_log_trade_new(journal, db):
    """Logging a new trade (id=None) should save it and assign an ID."""
    trade = _make_trade()
    journal.log_trade(trade)
    assert trade.id is not None

    # Should be in DB
    open_trades = db.get_open_trades()
    assert len(open_trades) == 1


def test_log_trade_update(journal, db):
    """Logging an existing trade should update it."""
    trade = _make_trade()
    journal.log_trade(trade)  # Save

    trade.exit_price = 51000.0
    trade.pnl = 10.0
    trade.status = "closed"
    journal.log_trade(trade)  # Update

    trades = db.get_recent_trades(10)
    assert trades[0].status == "closed"
    assert trades[0].pnl == 10.0


def test_log_signal(journal, db):
    sig = _make_signal()
    journal.log_signal(sig, acted_on=True)
    journal.log_signal(_make_signal(pair="ETH/USDC:USDC"), acted_on=False, reason="drawdown")


def test_performance_summary_no_trades(journal):
    summary = journal.get_performance_summary(days=30)
    assert summary["total_trades"] == 0


def test_performance_summary_with_trades(journal, db):
    from src.utils import now_iso

    # Create some closed trades
    for i, pnl in enumerate([50.0, -20.0, 30.0]):
        t = _make_trade(entry_time=now_iso())
        t.pnl = pnl
        t.pnl_pct = pnl / 500.0
        t.status = "closed"
        db.save_trade(t)

    summary = journal.get_performance_summary(days=30)
    assert summary["total_trades"] == 3
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["total_pnl"] == 60.0


def test_export_csv(journal, db, tmp_path):
    from src.utils import now_iso

    t = _make_trade(entry_time=now_iso())
    t.pnl = 25.0
    t.pnl_pct = 0.05
    t.status = "closed"
    db.save_trade(t)

    csv_path = str(tmp_path / "export.csv")
    journal.export_csv(csv_path, days=30)

    assert Path(csv_path).exists()
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "pair" in header
        row = next(reader)
        assert row[1] == "BTC/USDC:USDC"
