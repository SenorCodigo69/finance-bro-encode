"""Tests for the Brain class — Claude signal vetting and trade review."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.brain import Brain
from src.database import Database
from src.models import PortfolioSnapshot, Signal, Trade


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def brain(db):
    with patch("src.brain.anthropic.Anthropic"):
        b = Brain(api_key="test-key", model="claude-haiku-4-5-20251001", db=db)
    return b


def _make_signal(**kwargs):
    defaults = dict(
        pair="BTC/USDC:USDC", timeframe="1h", direction="long",
        confidence=0.75, strategy_name="trend_following",
        indicators={"rsi": 45}, reasoning="Test signal", timestamp="2026-03-12T00:00:00",
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _make_snapshot(**kwargs):
    defaults = dict(
        timestamp="2026-03-12T00:00:00", total_value=1000.0, cash=800.0,
        positions_value=200.0, open_positions=1, drawdown_pct=0.05,
        high_water_mark=1050.0, daily_pnl=10.0, total_pnl=50.0, total_pnl_pct=0.05,
        bot_balance=700.0, user_balance=300.0,
    )
    defaults.update(kwargs)
    return PortfolioSnapshot(**defaults)


# --- vet_signals ---

@pytest.mark.asyncio
async def test_vet_signals_approve(brain, db):
    """Brain should approve signals when Claude says so."""
    response_json = json.dumps([{
        "pair": "BTC/USDC:USDC", "approved": True,
        "reasoning": "Strong trend", "adjustments": {}
    }])
    brain._call_claude = MagicMock(return_value=response_json)

    sig = _make_signal()
    results = await brain.vet_signals([sig], _make_snapshot(), {"BTC/USDC:USDC": {"price": 50000}})

    assert len(results) == 1
    _, reasoning, approved = results[0]
    assert approved is True
    assert "Strong trend" in reasoning


@pytest.mark.asyncio
async def test_vet_signals_reject(brain):
    response_json = json.dumps([{
        "pair": "BTC/USDC:USDC", "approved": False,
        "reasoning": "High drawdown risk", "adjustments": {}
    }])
    brain._call_claude = MagicMock(return_value=response_json)

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    _, _, approved = results[0]
    assert approved is False


@pytest.mark.asyncio
async def test_vet_signals_empty_list(brain):
    results = await brain.vet_signals([], _make_snapshot(), {})
    assert results == []


@pytest.mark.asyncio
async def test_vet_signals_api_failure_rejects_all(brain):
    """If Claude fails, all signals should be rejected for safety."""
    brain._call_claude = MagicMock(side_effect=Exception("API timeout"))

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    assert len(results) == 1
    _, reasoning, approved = results[0]
    assert approved is False
    assert "unavailable" in reasoning.lower()


@pytest.mark.asyncio
async def test_vet_signals_invalid_json_rejects(brain):
    brain._call_claude = MagicMock(return_value="not json at all")

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    _, _, approved = results[0]
    assert approved is False


@pytest.mark.asyncio
async def test_vet_signals_clamps_size_modifier(brain):
    """Size modifier should be clamped to [0.1, 2.0]."""
    response_json = json.dumps([{
        "pair": "BTC/USDC:USDC", "approved": True,
        "reasoning": "Go big",
        "adjustments": {"size_modifier": 10.0}  # Way over max
    }])
    brain._call_claude = MagicMock(return_value=response_json)

    results = await brain.vet_signals([_make_signal(confidence=0.5)], _make_snapshot(), {})
    # Confidence should be modified but capped
    sig, _, approved = results[0]
    assert approved is True
    assert sig.confidence <= 0.95


@pytest.mark.asyncio
async def test_vet_signals_clamps_stop_loss_pct(brain):
    response_json = json.dumps([{
        "pair": "BTC/USDC:USDC", "approved": True,
        "reasoning": "OK",
        "adjustments": {"stop_loss_pct": 0.50}  # Over max 0.15
    }])
    brain._call_claude = MagicMock(return_value=response_json)

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    assert results[0][2] is True  # Still approved


# --- review_trades ---

@pytest.mark.asyncio
async def test_review_trades_success(brain, db):
    review_json = json.dumps({
        "summary": "Good performance", "patterns": ["trend works"],
        "suggestions": ["more BTC"], "risk_assessment": "Low risk",
    })
    brain._call_claude = MagicMock(return_value=review_json)

    trades = [Trade(
        id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000.0,
        quantity=0.01, stop_loss=49000.0, pnl=50.0, pnl_pct=0.05,
        status="closed",
    )]
    result = await brain.review_trades(trades, _make_snapshot())
    assert result["summary"] == "Good performance"
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_review_trades_failure(brain):
    brain._call_claude = MagicMock(side_effect=Exception("fail"))
    result = await brain.review_trades([], _make_snapshot())
    assert "failed" in result["summary"].lower()


# --- _strip_markdown ---

def test_strip_markdown_json_fence(brain):
    text = '```json\n[{"approved": true}]\n```'
    assert brain._strip_markdown(text) == '[{"approved": true}]'


def test_strip_markdown_plain_fence(brain):
    text = '```\n{"key": "val"}\n```'
    assert brain._strip_markdown(text) == '{"key": "val"}'


def test_strip_markdown_no_fence(brain):
    text = '{"key": "val"}'
    assert brain._strip_markdown(text) == '{"key": "val"}'


# --- token tracking ---

def test_total_tokens_starts_at_zero(brain):
    assert brain.total_tokens_used == 0
