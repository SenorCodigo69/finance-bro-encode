"""Tests for the Agent Negotiation Engine."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database import Database
from src.models import PortfolioSnapshot, Signal
from src.negotiation import (
    AGENT_PROFILES,
    AgentOpinion,
    NegotiationEngine,
    NegotiationRecord,
)


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "test.db"))
    yield d
    d.close()


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider that returns configurable JSON."""
    provider = MagicMock()
    provider.name = "mock"
    provider.total_tokens = 0
    return provider


@pytest.fixture
def signal():
    return Signal(
        pair="BTC/USDC:USDC",
        timeframe="15m",
        direction="long",
        confidence=0.72,
        strategy_name="momentum",
        indicators={"rsi": 32, "macd_hist": 0.5},
        reasoning="RSI oversold with MACD bullish crossover",
    )


@pytest.fixture
def portfolio():
    return PortfolioSnapshot(
        timestamp="2026-03-14T12:00:00",
        total_value=10000.0,
        cash=8000.0,
        positions_value=2000.0,
        open_positions=1,
        drawdown_pct=0.05,
        high_water_mark=10000.0,
        daily_pnl=50.0,
        total_pnl=150.0,
        total_pnl_pct=0.015,
    )


def _make_opinion_response(approved=True, risk_score=3, size_mod=1.0, reasoning="Looks good"):
    return json.dumps({
        "approved": approved,
        "confidence_modifier": 1.0,
        "size_modifier": size_mod,
        "risk_score": risk_score,
        "reasoning": reasoning,
    })


def _make_negotiation_response(action="maintain", approved=True, size_mod=1.0, reasoning="Sticking with my call"):
    return json.dumps({
        "action": action,
        "approved": approved,
        "size_modifier": size_mod,
        "reasoning": reasoning,
    })


@pytest.mark.asyncio
async def test_unanimous_approve(db, mock_provider, signal, portfolio):
    """All agents approve → unanimous approval, no negotiation rounds."""
    mock_provider.chat = AsyncMock(
        return_value=_make_opinion_response(approved=True, risk_score=2)
    )
    engine = NegotiationEngine([mock_provider], db)

    sig_out, record, approved = await engine.negotiate_signal(signal, portfolio, {})

    assert approved is True
    assert record.consensus_type == "unanimous"
    assert record.total_rounds == 0
    assert record.vetoed_by is None
    assert len(record.initial_opinions) == 4


@pytest.mark.asyncio
async def test_unanimous_reject(db, mock_provider, signal, portfolio):
    """All agents reject → unanimous rejection."""
    mock_provider.chat = AsyncMock(
        return_value=_make_opinion_response(approved=False, reasoning="Too risky")
    )
    engine = NegotiationEngine([mock_provider], db)

    sig_out, record, approved = await engine.negotiate_signal(signal, portfolio, {})

    assert approved is False
    assert record.consensus_type in ("unanimous", "vetoed")  # Delta may veto


@pytest.mark.asyncio
async def test_veto_blocks_trade(db, mock_provider, signal, portfolio):
    """If Delta (risk sentinel) rejects, it's a veto that blocks the trade."""
    call_count = 0

    async def mock_chat(system, user, **kwargs):
        nonlocal call_count
        call_count += 1
        # First 3 agents approve, Delta (4th) rejects = veto
        if call_count == 4:
            return _make_opinion_response(approved=False, risk_score=9, reasoning="Correlation risk too high")
        return _make_opinion_response(approved=True, risk_score=2)

    mock_provider.chat = mock_chat
    engine = NegotiationEngine([mock_provider], db)

    sig_out, record, approved = await engine.negotiate_signal(signal, portfolio, {})

    assert approved is False
    assert record.vetoed_by == "Delta"
    assert record.consensus_type == "vetoed"


@pytest.mark.asyncio
async def test_split_vote_triggers_negotiation(db, mock_provider, signal, portfolio):
    """Split vote triggers negotiation rounds."""
    call_count = 0

    async def mock_chat(system, user, **kwargs):
        nonlocal call_count
        call_count += 1
        # Initial opinions: Alpha rejects, Beta approves, Gamma approves, Delta approves
        if call_count <= 4:
            if call_count == 1:
                return _make_opinion_response(approved=False, reasoning="Too volatile")
            return _make_opinion_response(approved=True, risk_score=3)
        # Negotiation rounds: Alpha concedes
        return _make_negotiation_response(action="concede", approved=True, reasoning="OK, majority convinced me")

    mock_provider.chat = mock_chat
    engine = NegotiationEngine([mock_provider], db)

    sig_out, record, approved = await engine.negotiate_signal(signal, portfolio, {})

    assert approved is True
    assert record.total_rounds >= 1
    assert len(record.rounds) > 4  # 4 initial + at least 1 negotiation round


@pytest.mark.asyncio
async def test_negotiate_signals_batch(db, mock_provider, signal, portfolio):
    """Test the batch negotiate_signals method."""
    mock_provider.chat = AsyncMock(
        return_value=_make_opinion_response(approved=True)
    )
    engine = NegotiationEngine([mock_provider], db)

    signals = [signal, Signal(
        pair="ETH/USDC:USDC", timeframe="15m", direction="short",
        confidence=0.65, strategy_name="mean_reversion",
    )]

    results = await engine.negotiate_signals(signals, portfolio, {})

    assert len(results) == 2
    for sig, reasoning, approved in results:
        assert isinstance(reasoning, str)
        assert isinstance(approved, bool)


@pytest.mark.asyncio
async def test_provider_failure_fail_closed(db, mock_provider, signal, portfolio):
    """If all providers fail, agents default to reject (fail-closed)."""
    mock_provider.chat = AsyncMock(side_effect=RuntimeError("API down"))
    engine = NegotiationEngine([mock_provider], db)

    sig_out, record, approved = await engine.negotiate_signal(signal, portfolio, {})

    # All agents failed to respond → all defaulted to reject → unanimous reject or veto
    assert approved is False


@pytest.mark.asyncio
async def test_record_saved_to_db(db, mock_provider, signal, portfolio):
    """Negotiation records are persisted to the database."""
    mock_provider.chat = AsyncMock(
        return_value=_make_opinion_response(approved=True)
    )
    engine = NegotiationEngine([mock_provider], db)

    await engine.negotiate_signal(signal, portfolio, {})

    # Check that a negotiation_* key was saved
    rows = db.conn.execute(
        "SELECT key FROM agent_state WHERE key LIKE 'negotiation_%'"
    ).fetchall()
    assert len(rows) >= 1

    # Verify it's valid JSON
    data = json.loads(db.get_state(rows[0]["key"]))
    assert data["signal_pair"] == "BTC/USDC:USDC"
    assert "opinions" in data


@pytest.mark.asyncio
async def test_recent_records(db, mock_provider, signal, portfolio):
    """get_recent_records returns in-memory negotiation history."""
    mock_provider.chat = AsyncMock(
        return_value=_make_opinion_response(approved=True)
    )
    engine = NegotiationEngine([mock_provider], db)

    await engine.negotiate_signal(signal, portfolio, {})

    records = engine.get_recent_records()
    assert len(records) == 1
    assert records[0].signal_pair == "BTC/USDC:USDC"


def test_agent_profiles_valid():
    """Verify all 4 agent profiles are configured correctly."""
    assert len(AGENT_PROFILES) == 4
    names = [p.name for p in AGENT_PROFILES]
    assert names == ["Alpha", "Beta", "Gamma", "Delta"]

    # Delta has veto power
    delta = AGENT_PROFILES[3]
    assert delta.has_veto is True
    assert delta.weight == 1.5

    # All profiles have system prompts
    for p in AGENT_PROFILES:
        assert len(p.system_prompt) > 50
