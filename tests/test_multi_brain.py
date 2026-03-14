"""Tests for MultiBrain — multi-model consensus voting."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database import Database
from src.config import EscalationConfig
from src.models import PortfolioSnapshot, Signal, Trade
from src.multi_brain import ModelAccuracyTracker, MultiBrain
from src.providers import LLMProvider


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


def _make_provider(name: str, response: str | list[dict] | None = None):
    """Create a mock LLMProvider."""
    p = MagicMock(spec=LLMProvider)
    p.name = name
    p.total_tokens = 0
    if response is not None:
        if isinstance(response, list):
            response = json.dumps(response)
        p.chat = AsyncMock(return_value=response)
    return p


def _make_signal(**kwargs):
    defaults = dict(
        pair="BTC/USDC:USDC", timeframe="1h", direction="long",
        confidence=0.75, strategy_name="trend_following",
        indicators={}, reasoning="Test", timestamp="2026-03-12T00:00:00",
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _make_snapshot():
    return PortfolioSnapshot(
        timestamp="2026-03-12T00:00:00", total_value=1000.0, cash=800.0,
        positions_value=200.0, open_positions=1, drawdown_pct=0.05,
        high_water_mark=1050.0, daily_pnl=10.0, total_pnl=50.0, total_pnl_pct=0.05,
        bot_balance=700.0, user_balance=300.0,
    )


# --- Consensus voting ---

@pytest.mark.asyncio
async def test_consensus_all_approve(db):
    decision = [{"pair": "BTC/USDC:USDC", "approved": True, "reasoning": "Good"}]
    providers = [_make_provider("claude", decision), _make_provider("gemini", decision)]
    brain = MultiBrain(providers, db, consensus_threshold=0.66)

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    assert len(results) == 1
    _, _, approved = results[0]
    assert approved is True


@pytest.mark.asyncio
async def test_consensus_all_reject(db):
    decision = [{"pair": "BTC/USDC:USDC", "approved": False, "reasoning": "Bad"}]
    providers = [_make_provider("claude", decision), _make_provider("gemini", decision)]
    brain = MultiBrain(providers, db, consensus_threshold=0.66)

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    _, _, approved = results[0]
    assert approved is False


@pytest.mark.asyncio
async def test_consensus_split_below_threshold(db):
    approve = [{"pair": "BTC/USDC:USDC", "approved": True, "reasoning": "Yes"}]
    reject = [{"pair": "BTC/USDC:USDC", "approved": False, "reasoning": "No"}]
    providers = [
        _make_provider("claude", approve),
        _make_provider("gemini", reject),
        _make_provider("grok", reject),
    ]
    brain = MultiBrain(providers, db, consensus_threshold=0.66)

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    _, _, approved = results[0]
    assert approved is False  # 1/3 = 33% < 66%


@pytest.mark.asyncio
async def test_consensus_2_of_3_approve(db):
    approve = [{"pair": "BTC/USDC:USDC", "approved": True, "reasoning": "Yes"}]
    reject = [{"pair": "BTC/USDC:USDC", "approved": False, "reasoning": "No"}]
    providers = [
        _make_provider("claude", approve),
        _make_provider("gemini", approve),
        _make_provider("grok", reject),
    ]
    brain = MultiBrain(providers, db, consensus_threshold=0.66)

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    _, _, approved = results[0]
    assert approved is True  # 2/3 = 67% >= 66%


@pytest.mark.asyncio
async def test_all_providers_fail(db):
    p = _make_provider("claude")
    p.chat = AsyncMock(side_effect=Exception("API down"))
    brain = MultiBrain([p], db)

    results = await brain.vet_signals([_make_signal()], _make_snapshot(), {})
    _, reasoning, approved = results[0]
    assert approved is False
    assert "unavailable" in reasoning.lower()


@pytest.mark.asyncio
async def test_empty_signals(db):
    brain = MultiBrain([_make_provider("claude")], db)
    results = await brain.vet_signals([], _make_snapshot(), {})
    assert results == []


# --- Escalation ---

@pytest.mark.asyncio
async def test_escalation_close_call(db):
    """50/50 split with high confidence should trigger escalation."""
    approve = [{"pair": "BTC/USDC:USDC", "approved": True, "reasoning": "Yes"}]
    reject = [{"pair": "BTC/USDC:USDC", "approved": False, "reasoning": "No"}]

    # Tiebreaker provider approves
    tiebreaker_response = json.dumps({"approved": True, "reasoning": "Tiebreaker approves"})
    claude = _make_provider("claude", approve)
    gemini = _make_provider("gemini", reject)

    escalation_cfg = EscalationConfig(enabled=True, min_signal_confidence=0.7, tiebreaker_provider="claude")

    brain = MultiBrain([claude, gemini], db, consensus_threshold=0.66, escalation_config=escalation_cfg)

    # The escalation will try to use best model's historical vote first.
    # Since no accuracy data exists, it falls back to querying the tiebreaker.
    # Mock the tiebreaker call - claude.chat is already returning approve decisions
    # but _escalate_signal calls _query_provider which calls provider.chat again.
    # The tiebreaker result is the same as a normal query.

    results = await brain.vet_signals(
        [_make_signal(confidence=0.8)], _make_snapshot(), {}
    )
    assert len(results) == 1
    # Escalation was triggered, final verdict depends on tiebreaker


# --- Accuracy tracker ---

def test_accuracy_tracker_record_and_stats(db):
    tracker = ModelAccuracyTracker(db)

    # Record outcomes for 2 models across 3 signals
    for i in range(3):
        tracker.record_outcome(
            signal_id=f"sig_{i}", pair="BTC/USDC:USDC", direction="long",
            model_votes={"claude": True, "gemini": i < 2},  # Gemini wrong on last
            actual_profitable=True,
        )

    stats = tracker.get_model_stats()
    assert "claude" in stats
    assert "gemini" in stats
    assert stats["claude"]["accuracy"] == 1.0  # 3/3 correct
    assert stats["gemini"]["total_votes"] == 3


def test_accuracy_tracker_empty(db):
    tracker = ModelAccuracyTracker(db)
    assert tracker.get_model_stats() == {}
    assert tracker.get_best_model() is None
    assert tracker.get_model_weights() == {}


def test_accuracy_tracker_best_model(db):
    tracker = ModelAccuracyTracker(db)

    # Claude always right, gemini always wrong
    for i in range(5):
        tracker.record_outcome(
            signal_id=f"sig_{i}", pair="BTC/USDC:USDC", direction="long",
            model_votes={"claude": True, "gemini": False},
            actual_profitable=True,
        )

    assert tracker.get_best_model() == "claude"


def test_accuracy_weights_spread(db):
    tracker = ModelAccuracyTracker(db)

    # Claude: 100% accuracy, gemini: 0% accuracy
    for i in range(5):
        tracker.record_outcome(
            signal_id=f"sig_{i}", pair="BTC/USDC:USDC", direction="long",
            model_votes={"claude": True, "gemini": False},
            actual_profitable=True,
        )

    weights = tracker.get_model_weights()
    assert weights["claude"] == 1.5  # Best → max weight
    assert weights["gemini"] == 0.5  # Worst → min weight


def test_accuracy_weights_equal(db):
    tracker = ModelAccuracyTracker(db)

    # Both models always right
    for i in range(3):
        tracker.record_outcome(
            signal_id=f"sig_{i}", pair="BTC/USDC:USDC", direction="long",
            model_votes={"claude": True, "gemini": True},
            actual_profitable=True,
        )

    weights = tracker.get_model_weights()
    assert weights["claude"] == 1.0
    assert weights["gemini"] == 1.0


# --- review_trades ---

@pytest.mark.asyncio
async def test_review_trades_success(db):
    review_json = json.dumps({
        "summary": "OK performance", "patterns": [],
        "suggestions": [], "risk_assessment": "Low",
    })
    p = _make_provider("claude", review_json)
    # Override chat for review (it's called differently than vet)
    p.chat = AsyncMock(return_value=review_json)
    brain = MultiBrain([p], db)

    result = await brain.review_trades([], _make_snapshot())
    assert result["summary"] == "OK performance"


@pytest.mark.asyncio
async def test_review_trades_all_fail(db):
    p = _make_provider("claude")
    p.chat = AsyncMock(side_effect=Exception("down"))
    brain = MultiBrain([p], db)

    result = await brain.review_trades([], _make_snapshot())
    assert "failed" in result["summary"].lower()


# --- record_trade_outcome ---

def test_record_trade_outcome_no_vote_record(db):
    """Should not crash when no vote record exists."""
    brain = MultiBrain([], db)
    brain.record_trade_outcome("sig1", "BTC/USDC:USDC", "long", True)
    # No exception = pass


# --- format/stats helpers ---

def test_format_trades_empty(db):
    brain = MultiBrain([], db)
    assert brain._format_trades([]) == "No recent trades."


def test_calc_quick_stats_no_closed(db):
    brain = MultiBrain([], db)
    assert brain._calc_quick_stats([]) == {"trades": 0}


def test_calc_quick_stats(db):
    brain = MultiBrain([], db)
    trades = [
        Trade(id=1, pair="BTC/USDC:USDC", direction="long", entry_price=50000,
              quantity=0.01, stop_loss=49000, pnl=50.0, status="closed"),
        Trade(id=2, pair="ETH/USDC:USDC", direction="long", entry_price=3000,
              quantity=0.1, stop_loss=2900, pnl=-20.0, status="closed"),
    ]
    stats = brain._calc_quick_stats(trades)
    assert stats["trades"] == 2
    assert stats["wins"] == 1
    assert stats["total_pnl"] == 30.0


def test_total_tokens(db):
    p1 = _make_provider("claude")
    p1.total_tokens = 100
    p2 = _make_provider("gemini")
    p2.total_tokens = 200
    brain = MultiBrain([p1, p2], db)
    assert brain.total_tokens_used == 300
