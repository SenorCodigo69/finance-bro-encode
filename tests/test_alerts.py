"""Tests for the alert system."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.alerts import AlertLevel, AlertManager
from src.models import PortfolioSnapshot, Trade


@pytest.fixture
def alert_mgr(tmp_path):
    mgr = AlertManager()
    # Override log path to use tmp
    mgr._log_path = tmp_path / "alerts.json"
    mgr._alert_log = []
    return mgr


@pytest.mark.asyncio
async def test_alert_info(alert_mgr):
    await alert_mgr.alert(AlertLevel.INFO, "Test", "Test message")
    assert len(alert_mgr._alert_log) == 1
    assert alert_mgr._alert_log[0]["level"] == "info"
    assert alert_mgr._alert_log[0]["title"] == "Test"


@pytest.mark.asyncio
async def test_alert_saves_to_file(alert_mgr):
    await alert_mgr.alert(AlertLevel.WARNING, "Warn", "Something happened")
    assert alert_mgr._log_path.exists()
    data = json.loads(alert_mgr._log_path.read_text())
    assert len(data) == 1


@pytest.mark.asyncio
async def test_alert_keeps_last_100(alert_mgr):
    for i in range(110):
        await alert_mgr.alert(AlertLevel.INFO, f"Alert {i}", "msg")
    data = json.loads(alert_mgr._log_path.read_text())
    assert len(data) <= 100


@pytest.mark.asyncio
async def test_webhook_called(tmp_path):
    mgr = AlertManager(webhook_url="https://example.com/webhook")
    mgr._log_path = tmp_path / "alerts.json"
    mgr._alert_log = []

    with patch("aiohttp.ClientSession") as mock_session:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value = mock_ctx
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=False)
        mock_session.return_value = mock_session_instance

        await mgr.alert(AlertLevel.CRITICAL, "CRITICAL", "Max drawdown")


@pytest.mark.asyncio
async def test_trade_opened_alert(alert_mgr):
    trade = Trade(
        id=1, pair="BTC/USDC:USDC", direction="long",
        entry_price=50000.0, quantity=0.01, stop_loss=49000.0,
    )
    await alert_mgr.trade_opened(trade)
    assert len(alert_mgr._alert_log) == 1
    assert "BTC/USDC:USDC" in alert_mgr._alert_log[0]["title"]


@pytest.mark.asyncio
async def test_trade_closed_alert_profit(alert_mgr):
    trade = Trade(
        id=1, pair="ETH/USDC:USDC", direction="long",
        entry_price=3000.0, quantity=0.1, stop_loss=2900.0,
        pnl=50.0, pnl_pct=0.05, status="closed",
    )
    await alert_mgr.trade_closed(trade)
    assert alert_mgr._alert_log[0]["level"] == "info"


@pytest.mark.asyncio
async def test_trade_closed_alert_loss(alert_mgr):
    trade = Trade(
        id=1, pair="ETH/USDC:USDC", direction="long",
        entry_price=3000.0, quantity=0.1, stop_loss=2900.0,
        pnl=-20.0, pnl_pct=-0.02, status="stopped_out",
    )
    await alert_mgr.trade_closed(trade)
    assert alert_mgr._alert_log[0]["level"] == "warning"


@pytest.mark.asyncio
async def test_circuit_breaker_alert(alert_mgr):
    snap = PortfolioSnapshot(
        timestamp="2026-03-12", total_value=700.0, cash=700.0,
        positions_value=0, open_positions=0, drawdown_pct=0.30,
        high_water_mark=1000.0, daily_pnl=-50.0, total_pnl=-300.0, total_pnl_pct=-0.30,
    )
    await alert_mgr.circuit_breaker(snap)
    assert alert_mgr._alert_log[0]["level"] == "critical"


def test_get_recent_alerts(alert_mgr):
    alert_mgr._alert_log = [{"id": i} for i in range(50)]
    recent = alert_mgr.get_recent_alerts(20)
    assert len(recent) == 20


def test_load_log_corrupt_file(tmp_path):
    log_path = tmp_path / "alerts.json"
    log_path.write_text("not json")
    mgr = AlertManager()
    mgr._log_path = log_path
    mgr._load_log()
    assert mgr._alert_log == []
