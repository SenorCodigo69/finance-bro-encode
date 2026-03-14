"""Tests for LLM provider abstraction and multi-brain consensus."""

from __future__ import annotations

import json
import pytest
import asyncio

from src.providers import LLMProvider, _strip_markdown, AnthropicProvider, OpenAICompatibleProvider
from src.multi_brain import MultiBrain
from src.models import Signal, PortfolioSnapshot


# ── Helpers ──────────────────────────────────────────────────────────


class FakeProvider(LLMProvider):
    """Test provider that returns canned responses."""

    def __init__(self, name: str, responses: dict[str, bool] | None = None, fail: bool = False):
        super().__init__(name, "fake-model")
        self.responses = responses or {}  # pair -> approved
        self.fail = fail
        self.call_count = 0

    async def chat(self, system: str, user: str, max_tokens: int = 2000, timeout: float = 30.0) -> str:
        self.call_count += 1
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        result = []
        for pair, approved in self.responses.items():
            result.append({
                "pair": pair,
                "approved": approved,
                "reasoning": f"{self.name} says {'yes' if approved else 'no'}",
                "adjustments": {"stop_loss_pct": None, "take_profit_pct": None, "size_modifier": None},
            })
        return json.dumps(result)


class FakeDB:
    """Minimal fake database for tests."""

    def __init__(self):
        self._state = {}

    def get_recent_trades(self, n: int):
        return []

    def set_state(self, key: str, value: str):
        self._state[key] = value

    def get_state(self, key: str):
        return self._state.get(key)

    def get_model_outcomes(self, limit: int = 1000) -> list[dict]:
        return []

    def save_model_outcome(self, **kwargs) -> None:
        pass

    def save_review(self, review: dict):
        pass


def make_signal(pair: str = "BTC/USDC:USDC", direction: str = "long", confidence: float = 0.75) -> Signal:
    return Signal(
        pair=pair,
        timeframe="1h",
        direction=direction,
        confidence=confidence,
        strategy_name="trend_following",
        indicators={"rsi": 45.0},
        reasoning="Test signal",
    )


def make_snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp="2026-03-11T12:00:00",
        total_value=1000.0,
        cash=800.0,
        positions_value=200.0,
        open_positions=1,
        drawdown_pct=0.05,
        high_water_mark=1000.0,
        daily_pnl=10.0,
        total_pnl=50.0,
        total_pnl_pct=0.05,
    )


# ── Provider tests ───────────────────────────────────────────────────


def test_strip_markdown_json():
    assert _strip_markdown('```json\n[{"a": 1}]\n```') == '[{"a": 1}]'


def test_strip_markdown_plain():
    assert _strip_markdown('[{"a": 1}]') == '[{"a": 1}]'


def test_strip_markdown_backticks_only():
    assert _strip_markdown("```\nhello\n```") == "hello"


def test_openai_provider_known_urls():
    assert "gemini" in OpenAICompatibleProvider.KNOWN_URLS


def test_openai_provider_unknown_url_raises():
    with pytest.raises(ValueError, match="No base_url"):
        OpenAICompatibleProvider("unknown_provider", "key", "model")


def test_openai_provider_custom_url():
    p = OpenAICompatibleProvider("custom", "key", "model", base_url="http://localhost:1234/v1")
    assert p._base_url == "http://localhost:1234/v1"


def test_fake_provider_returns_json():
    p = FakeProvider("test", {"BTC/USDC:USDC": True})
    result = asyncio.get_event_loop().run_until_complete(p.chat("sys", "usr"))
    parsed = json.loads(result)
    assert parsed[0]["pair"] == "BTC/USDC:USDC"
    assert parsed[0]["approved"] is True


# ── MultiBrain consensus tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_unanimous_approve():
    """All 3 models approve → signal approved."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": True}),
        FakeProvider("model_b", {"BTC/USDC:USDC": True}),
        FakeProvider("model_c", {"BTC/USDC:USDC": True}),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.66)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    assert len(results) == 1
    _, reasoning, approved = results[0]
    assert approved is True
    assert "3/3" in reasoning


@pytest.mark.asyncio
async def test_unanimous_reject():
    """All 3 models reject → signal rejected."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": False}),
        FakeProvider("model_b", {"BTC/USDC:USDC": False}),
        FakeProvider("model_c", {"BTC/USDC:USDC": False}),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.66)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    _, _, approved = results[0]
    assert approved is False


@pytest.mark.asyncio
async def test_two_thirds_approve():
    """2 out of 3 approve (67%) → approved at 0.67 threshold."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": True}),
        FakeProvider("model_b", {"BTC/USDC:USDC": True}),
        FakeProvider("model_c", {"BTC/USDC:USDC": False}),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.66)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    _, reasoning, approved = results[0]
    assert approved is True
    assert "2/3" in reasoning


@pytest.mark.asyncio
async def test_one_third_approve_rejected():
    """1 out of 3 approve (33%) → rejected at 0.67 threshold."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": True}),
        FakeProvider("model_b", {"BTC/USDC:USDC": False}),
        FakeProvider("model_c", {"BTC/USDC:USDC": False}),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.66)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    _, _, approved = results[0]
    assert approved is False


@pytest.mark.asyncio
async def test_simple_majority_threshold():
    """2/3 approve with 0.5 threshold → approved."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": True}),
        FakeProvider("model_b", {"BTC/USDC:USDC": True}),
        FakeProvider("model_c", {"BTC/USDC:USDC": False}),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.5)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    _, _, approved = results[0]
    assert approved is True


@pytest.mark.asyncio
async def test_unanimous_threshold():
    """2/3 approve with 1.0 threshold → rejected (need 100%)."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": True}),
        FakeProvider("model_b", {"BTC/USDC:USDC": True}),
        FakeProvider("model_c", {"BTC/USDC:USDC": False}),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=1.0)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    _, _, approved = results[0]
    assert approved is False


@pytest.mark.asyncio
async def test_provider_failure_graceful():
    """One provider fails → vote with remaining 2."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": True}),
        FakeProvider("model_b", {"BTC/USDC:USDC": True}),
        FakeProvider("model_c", fail=True),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.66)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    _, reasoning, approved = results[0]
    assert approved is True
    assert "2/2" in reasoning  # Only 2 models responded


@pytest.mark.asyncio
async def test_all_providers_fail():
    """All providers fail → reject all signals for safety."""
    providers = [
        FakeProvider("model_a", fail=True),
        FakeProvider("model_b", fail=True),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.66)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})

    _, reasoning, approved = results[0]
    assert approved is False
    assert "unavailable" in reasoning.lower()


@pytest.mark.asyncio
async def test_multiple_signals():
    """Models can approve one signal and reject another."""
    providers = [
        FakeProvider("model_a", {"BTC/USDC:USDC": True, "ETH/USDC:USDC": False}),
        FakeProvider("model_b", {"BTC/USDC:USDC": True, "ETH/USDC:USDC": False}),
        FakeProvider("model_c", {"BTC/USDC:USDC": False, "ETH/USDC:USDC": True}),
    ]
    brain = MultiBrain(providers, FakeDB(), consensus_threshold=0.66)

    signals = [make_signal("BTC/USDC:USDC"), make_signal("ETH/USDC:USDC")]
    results = await brain.vet_signals(signals, make_snapshot(), {})

    assert len(results) == 2
    _, _, btc_approved = results[0]
    _, _, eth_approved = results[1]
    assert btc_approved is True   # 2/3 approve
    assert eth_approved is False  # 1/3 approve


@pytest.mark.asyncio
async def test_empty_signals():
    """No signals → empty results."""
    brain = MultiBrain([FakeProvider("a", {})], FakeDB())
    results = await brain.vet_signals([], make_snapshot(), {})
    assert results == []


@pytest.mark.asyncio
async def test_single_provider():
    """Single provider mode — approve if it approves."""
    brain = MultiBrain([FakeProvider("solo", {"BTC/USDC:USDC": True})], FakeDB(), consensus_threshold=0.5)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})
    _, _, approved = results[0]
    assert approved is True


@pytest.mark.asyncio
async def test_single_provider_reject():
    """Single provider mode — reject if it rejects."""
    brain = MultiBrain([FakeProvider("solo", {"BTC/USDC:USDC": False})], FakeDB(), consensus_threshold=0.5)
    results = await brain.vet_signals([make_signal()], make_snapshot(), {})
    _, _, approved = results[0]
    assert approved is False


@pytest.mark.asyncio
async def test_all_providers_queried():
    """All providers are called exactly once."""
    providers = [
        FakeProvider("a", {"BTC/USDC:USDC": True}),
        FakeProvider("b", {"BTC/USDC:USDC": True}),
        FakeProvider("c", {"BTC/USDC:USDC": False}),
    ]
    brain = MultiBrain(providers, FakeDB())
    await brain.vet_signals([make_signal()], make_snapshot(), {})

    for p in providers:
        assert p.call_count == 1


@pytest.mark.asyncio
async def test_vote_record_saved():
    """Vote records are persisted to the database."""
    db = FakeDB()
    providers = [
        FakeProvider("a", {"BTC/USDC:USDC": True}),
        FakeProvider("b", {"BTC/USDC:USDC": False}),
    ]
    brain = MultiBrain(providers, db, consensus_threshold=0.5)
    await brain.vet_signals([make_signal()], make_snapshot(), {})

    # Check that vote records were saved — one batch key (timestamp) and one
    # per-pair shortcut key (for record_trade_outcome lookups)
    vote_keys = [k for k in db._state if k.startswith("vote_record_")]
    assert len(vote_keys) >= 1

    # Find the batch record (it contains a "models" field)
    batch_records = [
        json.loads(db._state[k])
        for k in vote_keys
        if "models" in json.loads(db._state[k])
    ]
    assert len(batch_records) == 1
    record = batch_records[0]
    assert record["models"] == ["a", "b"]
    assert len(record["signals"]) == 1
    assert record["signals"][0]["votes"]["a"]["approved"] is True
    assert record["signals"][0]["votes"]["b"]["approved"] is False


@pytest.mark.asyncio
async def test_total_tokens():
    """Token tracking works across providers."""
    providers = [
        FakeProvider("a", {"BTC/USDC:USDC": True}),
        FakeProvider("b", {"BTC/USDC:USDC": True}),
    ]
    brain = MultiBrain(providers, FakeDB())
    # FakeProvider doesn't track tokens, so total should be 0
    assert brain.total_tokens_used == 0
