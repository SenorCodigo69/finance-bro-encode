"""Tests for the strategy evolver — validation logic, safety rails, leaderboard."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import StrategyConfig
from src.database import Database
from src.models import Signal, Trade
from src.strategy import BaseStrategy, StrategyEngine
from src.strategy_evolver import (
    ALLOWED_IMPORTS,
    EVOLVED_DIR,
    FORBIDDEN_TOKENS,
    MAX_ACTIVE_STRATEGIES,
    EvolvedStrategyMeta,
    StrategyEvolver,
    StrategyPerformance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Create a fresh in-memory-ish DB in a temp directory."""
    db = Database(db_path=str(tmp_path / "test.db"))
    return db


@pytest.fixture
def evolved_dir(tmp_path):
    """Override the evolved strategies directory to a temp location."""
    d = tmp_path / "evolved_strategies"
    d.mkdir()
    return d


@pytest.fixture
def evolver(tmp_db, evolved_dir, monkeypatch):
    """Create a StrategyEvolver with mocked Claude client and temp dirs."""
    monkeypatch.setattr("src.strategy_evolver.EVOLVED_DIR", evolved_dir)
    with patch("src.strategy_evolver.anthropic.Anthropic"):
        ev = StrategyEvolver(api_key="test-key", model="test-model", db=tmp_db)
    return ev


@pytest.fixture
def valid_strategy_code():
    """Return valid Python code for an evolved strategy."""
    return textwrap.dedent('''\
        """Auto-generated strategy: test_strat

        Generated: 2025-01-01T00:00:00
        Rationale: Test strategy
        """

        from __future__ import annotations

        import pandas as pd

        from src.config import StrategyConfig
        from src.indicators import compute_all
        from src.models import Signal
        from src.utils import now_iso
        from src.strategy import BaseStrategy


        class EvolvedStrategy(BaseStrategy):
            """Auto-generated strategy: test_strat"""

            name = "test_strat"

            def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig) -> Signal | None:
                # Simple RSI + volume strategy for testing
                tf = "15m" if "15m" in data else list(data.keys())[0]
                df = data[tf]
                if len(df) < 30:
                    return None

                enriched = compute_all(df, config.__dict__)
                latest = self._get_latest(enriched)
                if latest is None:
                    return None

                rsi_val = latest.get("rsi")
                vol_ratio = latest.get("volume_ratio")

                if pd.isna(rsi_val) or pd.isna(vol_ratio):
                    return None

                if rsi_val < 25 and vol_ratio > 1.5:
                    return Signal(
                        pair=pair, timeframe=tf, direction="long",
                        confidence=0.7, strategy_name=self.name,
                        indicators={"rsi": rsi_val, "volume_ratio": vol_ratio},
                        reasoning=f"RSI very oversold ({rsi_val:.1f}) with volume spike",
                        timestamp=now_iso(),
                    )
                return None
    ''')


@pytest.fixture
def write_strategy(evolved_dir, valid_strategy_code):
    """Write a valid strategy file and return the path."""
    fp = evolved_dir / "test_strat_abc123.py"
    fp.write_text(valid_strategy_code)
    return fp


@pytest.fixture
def sample_meta(write_strategy):
    """Return an EvolvedStrategyMeta pointing to the valid strategy file."""
    return EvolvedStrategyMeta(
        id=None,
        name="test_strat",
        file_path=str(write_strategy),
        code_hash="abc123",
        generation=1,
        parent_strategies=["momentum"],
        rationale="Test strategy",
        status="candidate",
        created_at="2025-01-01T00:00:00",
    )


@pytest.fixture
def make_trades():
    """Factory to create Trade objects with specific strategy names and P&L."""
    def _make(strategy_name: str, pnl: float, pair: str = "BTC/USDC:USDC") -> Trade:
        return Trade(
            id=None,
            pair=pair,
            direction="long",
            entry_price=100.0,
            quantity=0.1,
            stop_loss=95.0,
            take_profit=110.0,
            exit_price=100 + pnl * 10,
            status="closed",
            pnl=pnl,
            pnl_pct=pnl / 100,
            entry_time="2025-01-01T00:00:00",
            exit_time="2025-01-01T01:00:00",
            signal_data={"strategy": strategy_name},
            ai_reasoning="test",
        )
    return _make


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for validate_strategy() — the core safety gate."""

    def test_valid_strategy_passes(self, evolver, sample_meta):
        valid, reason = evolver.validate_strategy(sample_meta)
        assert valid is True
        assert reason == "OK"

    def test_missing_file_fails(self, evolver):
        meta = EvolvedStrategyMeta(
            name="ghost",
            file_path="/nonexistent/path.py",
            code_hash="xyz",
            status="candidate",
            created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "not found" in reason

    def test_forbidden_os_import(self, evolver, evolved_dir):
        code = textwrap.dedent('''\
            import os
            from src.strategy import BaseStrategy
            from src.models import Signal
            import pandas as pd
            from src.config import StrategyConfig

            class EvolvedStrategy(BaseStrategy):
                name = "evil"
                def analyze(self, data, pair, config):
                    os.system("rm -rf /")
                    return None
        ''')
        fp = evolved_dir / "evil_deadbeef.py"
        fp.write_text(code)
        meta = EvolvedStrategyMeta(
            name="evil", file_path=str(fp), code_hash="deadbeef",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "Forbidden" in reason or "import" in reason.lower()

    def test_forbidden_eval(self, evolver, evolved_dir):
        code = textwrap.dedent('''\
            from src.strategy import BaseStrategy
            from src.models import Signal
            import pandas as pd
            from src.config import StrategyConfig

            class EvolvedStrategy(BaseStrategy):
                name = "eval_user"
                def analyze(self, data, pair, config):
                    eval("1+1")
                    return None
        ''')
        fp = evolved_dir / "eval_user_aaa.py"
        fp.write_text(code)
        meta = EvolvedStrategyMeta(
            name="eval_user", file_path=str(fp), code_hash="aaa",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "Forbidden" in reason

    def test_forbidden_open(self, evolver, evolved_dir):
        code = textwrap.dedent('''\
            from src.strategy import BaseStrategy
            from src.models import Signal
            import pandas as pd
            from src.config import StrategyConfig

            class EvolvedStrategy(BaseStrategy):
                name = "file_reader"
                def analyze(self, data, pair, config):
                    f = open("/etc/passwd")
                    return None
        ''')
        fp = evolved_dir / "file_reader_bbb.py"
        fp.write_text(code)
        meta = EvolvedStrategyMeta(
            name="file_reader", file_path=str(fp), code_hash="bbb",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "Forbidden" in reason

    def test_forbidden_subprocess(self, evolver, evolved_dir):
        code = textwrap.dedent('''\
            import subprocess
            from src.strategy import BaseStrategy
            from src.models import Signal
            import pandas as pd
            from src.config import StrategyConfig

            class EvolvedStrategy(BaseStrategy):
                name = "cmd_exec"
                def analyze(self, data, pair, config):
                    subprocess.run(["ls"])
                    return None
        ''')
        fp = evolved_dir / "cmd_exec_ccc.py"
        fp.write_text(code)
        meta = EvolvedStrategyMeta(
            name="cmd_exec", file_path=str(fp), code_hash="ccc",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "Forbidden" in reason

    def test_disallowed_import_requests(self, evolver, evolved_dir):
        code = textwrap.dedent('''\
            import requests
            from src.strategy import BaseStrategy
            from src.models import Signal
            import pandas as pd
            from src.config import StrategyConfig

            class EvolvedStrategy(BaseStrategy):
                name = "http_caller"
                def analyze(self, data, pair, config):
                    requests.get("https://evil.com")
                    return None
        ''')
        fp = evolved_dir / "http_caller_ddd.py"
        fp.write_text(code)
        meta = EvolvedStrategyMeta(
            name="http_caller", file_path=str(fp), code_hash="ddd",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "Forbidden" in reason or "import" in reason.lower()

    def test_no_evolved_strategy_class(self, evolver, evolved_dir):
        code = textwrap.dedent('''\
            import pandas as pd
            from src.config import StrategyConfig
            from src.models import Signal

            class WrongName:
                name = "wrong"
                def analyze(self, data, pair, config):
                    return None
        ''')
        fp = evolved_dir / "wrong_eee.py"
        fp.write_text(code)
        meta = EvolvedStrategyMeta(
            name="wrong", file_path=str(fp), code_hash="eee",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "EvolvedStrategy" in reason or "not a BaseStrategy" in reason

    def test_strategy_returning_wrong_type(self, evolver, evolved_dir):
        code = textwrap.dedent('''\
            import pandas as pd
            from src.strategy import BaseStrategy
            from src.config import StrategyConfig
            from src.models import Signal

            class EvolvedStrategy(BaseStrategy):
                name = "bad_return"
                def analyze(self, data, pair, config):
                    return "this is not a Signal"
        ''')
        fp = evolved_dir / "bad_return_fff.py"
        fp.write_text(code)
        meta = EvolvedStrategyMeta(
            name="bad_return", file_path=str(fp), code_hash="fff",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        valid, reason = evolver.validate_strategy(meta)
        assert valid is False
        assert "Signal" in reason or "str" in reason


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Tests for evaluate_strategies() — performance aggregation."""

    def test_evaluate_empty(self, evolver):
        perf = evolver.evaluate_strategies()
        assert perf == {}

    def test_evaluate_with_trades(self, evolver, tmp_db, make_trades):
        # Insert some trades
        tmp_db.save_trade(make_trades("momentum", 5.0))
        tmp_db.save_trade(make_trades("momentum", -2.0))
        tmp_db.save_trade(make_trades("momentum", 3.0))
        tmp_db.save_trade(make_trades("breakout", -1.0))
        tmp_db.save_trade(make_trades("breakout", -0.5))

        perf = evolver.evaluate_strategies()

        assert "momentum" in perf
        assert "breakout" in perf

        m = perf["momentum"]
        assert m.total_trades == 3
        assert m.wins == 2
        assert m.losses == 1
        assert m.win_rate == pytest.approx(2 / 3)
        assert m.total_pnl == pytest.approx(6.0)
        assert m.max_win == pytest.approx(5.0)
        assert m.max_loss == pytest.approx(-2.0)

        b = perf["breakout"]
        assert b.total_trades == 2
        assert b.wins == 0
        assert b.losses == 2
        assert b.win_rate == pytest.approx(0.0)

    def test_evaluate_ignores_open_trades(self, evolver, tmp_db):
        # Open trade — no pnl
        open_trade = Trade(
            id=None, pair="BTC/USDC:USDC", direction="long",
            entry_price=100.0, quantity=0.1, stop_loss=95.0,
            status="open", pnl=None,
            entry_time="2025-01-01T00:00:00",
            signal_data={"strategy": "momentum"},
        )
        tmp_db.save_trade(open_trade)

        perf = evolver.evaluate_strategies()
        assert perf == {}


# ---------------------------------------------------------------------------
# Activation & retirement tests
# ---------------------------------------------------------------------------


class TestActivationRetirement:
    """Tests for activate_strategy() and retire_worst()."""

    def test_activate_valid_strategy(self, evolver, sample_meta, tmp_db):
        tmp_db.save_evolved_strategy(sample_meta)
        result = evolver.activate_strategy(sample_meta)
        assert result is True
        assert sample_meta.status == "active"
        assert sample_meta.activated_at is not None
        assert "test_strat" in [s.name for s in evolver.get_active_strategies()]

    def test_activate_fails_at_max(self, evolver, evolved_dir, tmp_db):
        # Fill up to max with fake active strategies
        for i in range(MAX_ACTIVE_STRATEGIES):
            m = EvolvedStrategyMeta(
                name=f"strat_{i}", file_path=f"/fake/{i}.py", code_hash=f"hash{i}",
                status="active", created_at="2025-01-01T00:00:00",
            )
            tmp_db.save_evolved_strategy(m)

        new_meta = EvolvedStrategyMeta(
            name="one_too_many", file_path="/fake/extra.py", code_hash="extra",
            status="candidate", created_at="2025-01-01T00:00:00",
        )
        result = evolver.activate_strategy(new_meta)
        assert result is False

    def test_retire_worst(self, evolver, tmp_db):
        # Create two active strategies with different performance
        good = EvolvedStrategyMeta(
            name="good_strat", file_path="/fake/good.py", code_hash="good",
            status="active", created_at="2025-01-01T00:00:00",
            total_trades=10, wins=8, losses=2, total_pnl=50.0, win_rate=0.8,
        )
        bad = EvolvedStrategyMeta(
            name="bad_strat", file_path="/fake/bad.py", code_hash="bad",
            status="active", created_at="2025-01-01T00:00:00",
            total_trades=10, wins=2, losses=8, total_pnl=-30.0, win_rate=0.2,
        )
        tmp_db.save_evolved_strategy(good)
        tmp_db.save_evolved_strategy(bad)

        retired = evolver.retire_worst(min_trades=5)
        assert retired is not None
        assert retired.name == "bad_strat"
        assert retired.status == "retired"
        assert retired.retired_at is not None

    def test_retire_none_if_no_active(self, evolver):
        retired = evolver.retire_worst()
        assert retired is None

    def test_retire_none_if_insufficient_trades(self, evolver, tmp_db):
        m = EvolvedStrategyMeta(
            name="new_strat", file_path="/fake/new.py", code_hash="new",
            status="active", created_at="2025-01-01T00:00:00",
            total_trades=2, wins=0, losses=2, total_pnl=-5.0, win_rate=0.0,
        )
        tmp_db.save_evolved_strategy(m)
        retired = evolver.retire_worst(min_trades=5)
        assert retired is None


# ---------------------------------------------------------------------------
# Leaderboard & stats tests
# ---------------------------------------------------------------------------


class TestLeaderboard:

    def test_leaderboard_sorted_by_performance(self, evolver, tmp_db):
        strats = [
            EvolvedStrategyMeta(
                name="worst", file_path="/f/w.py", code_hash="w",
                status="retired", created_at="2025-01-01T00:00:00",
                total_trades=10, wins=1, losses=9, total_pnl=-40.0, win_rate=0.1,
            ),
            EvolvedStrategyMeta(
                name="best", file_path="/f/b.py", code_hash="b",
                status="active", created_at="2025-01-01T00:00:00",
                total_trades=10, wins=9, losses=1, total_pnl=80.0, win_rate=0.9,
            ),
            EvolvedStrategyMeta(
                name="mid", file_path="/f/m.py", code_hash="m",
                status="active", created_at="2025-01-01T00:00:00",
                total_trades=10, wins=5, losses=5, total_pnl=10.0, win_rate=0.5,
            ),
        ]
        for s in strats:
            tmp_db.save_evolved_strategy(s)

        board = evolver.get_leaderboard()
        assert len(board) == 3
        assert board[0].name == "best"
        assert board[1].name == "mid"
        assert board[2].name == "worst"

    def test_update_strategy_stats(self, evolver, tmp_db, sample_meta):
        tmp_db.save_evolved_strategy(sample_meta)

        evolver.update_strategy_stats("test_strat", 5.0)
        evolver.update_strategy_stats("test_strat", -2.0)
        evolver.update_strategy_stats("test_strat", 3.0)

        updated = tmp_db.get_evolved_strategy_by_name("test_strat")
        assert updated.total_trades == 3
        assert updated.wins == 2
        assert updated.losses == 1
        assert updated.total_pnl == pytest.approx(6.0)
        assert updated.win_rate == pytest.approx(2 / 3)

    def test_update_stats_ignores_non_evolved(self, evolver):
        # Should not crash for built-in strategies
        evolver.update_strategy_stats("momentum", 5.0)


# ---------------------------------------------------------------------------
# Generation tests (mocked Claude)
# ---------------------------------------------------------------------------


class TestGeneration:

    def test_generate_parses_claude_response(self, evolver, evolved_dir, monkeypatch):
        """Test that _parse_and_store correctly handles a well-formed Claude response."""
        monkeypatch.setattr("src.strategy_evolver.EVOLVED_DIR", evolved_dir)

        response = textwrap.dedent('''\
            ```python
            # Volume divergence strategy
            tf = "15m" if "15m" in data else list(data.keys())[0]
            df = data[tf]
            if len(df) < 30:
                return None

            enriched = compute_all(df, config.__dict__)
            latest = self._get_latest(enriched)
            if latest is None:
                return None

            rsi_val = latest.get("rsi")
            vol_ratio = latest.get("volume_ratio")
            obv_val = latest.get("obv")

            if pd.isna(rsi_val) or pd.isna(vol_ratio):
                return None

            if rsi_val < 30 and vol_ratio > 2.0:
                return Signal(
                    pair=pair, timeframe=tf, direction="long",
                    confidence=0.75, strategy_name=self.name,
                    indicators={"rsi": rsi_val, "volume_ratio": vol_ratio},
                    reasoning=f"Volume divergence detected",
                    timestamp=now_iso(),
                )
            return None
            ```

            STRATEGY_NAME: volume_divergence
            RATIONALE: Exploits volume spikes during RSI extremes that other strategies miss.
        ''')

        perf = {"momentum": StrategyPerformance(strategy_name="momentum", total_trades=10)}
        meta = evolver._parse_and_store(response, perf)

        assert meta is not None
        assert meta.name == "volume_divergence"
        assert meta.rationale == "Exploits volume spikes during RSI extremes that other strategies miss."
        assert meta.status == "candidate"
        assert meta.code_hash != ""
        assert Path(meta.file_path).exists()

    def test_generate_rejects_duplicate_hash(self, evolver, evolved_dir, tmp_db, monkeypatch):
        """Second identical generation should return None."""
        monkeypatch.setattr("src.strategy_evolver.EVOLVED_DIR", evolved_dir)

        response = textwrap.dedent('''\
            ```python
            return None
            ```

            STRATEGY_NAME: duplicate_test
            RATIONALE: Test dedup.
        ''')

        perf = {}
        meta1 = evolver._parse_and_store(response, perf)
        assert meta1 is not None

        meta2 = evolver._parse_and_store(response, perf)
        assert meta2 is None  # Duplicate detected

    def test_generate_blocked_at_max_active(self, evolver, tmp_db):
        """generate_strategy() should refuse if already at max active."""
        for i in range(MAX_ACTIVE_STRATEGIES):
            m = EvolvedStrategyMeta(
                name=f"s{i}", file_path=f"/f/{i}.py", code_hash=f"h{i}",
                status="active", created_at="2025-01-01T00:00:00",
            )
            tmp_db.save_evolved_strategy(m)

        result = evolver.generate_strategy({})
        assert result is None

    def test_generate_missing_strategy_name(self, evolver, evolved_dir, monkeypatch):
        """Response without STRATEGY_NAME should return None."""
        monkeypatch.setattr("src.strategy_evolver.EVOLVED_DIR", evolved_dir)

        response = "```python\nreturn None\n```\nRATIONALE: missing name"
        meta = evolver._parse_and_store(response, {})
        assert meta is None


# ---------------------------------------------------------------------------
# StrategyEngine integration
# ---------------------------------------------------------------------------


class TestEngineIntegration:

    def test_load_evolved_into_engine(self, evolver, sample_meta, tmp_db):
        tmp_db.save_evolved_strategy(sample_meta)
        evolver.activate_strategy(sample_meta)

        engine = StrategyEngine(StrategyConfig())
        active = evolver.get_active_strategies()
        engine.load_evolved_strategies(active)

        # Should have 6 built-in + 1 evolved
        assert len(engine.strategies) == 7
        names = [s.name for s in engine.strategies]
        assert "test_strat" in names

    def test_reload_clears_old_evolved(self, evolver, sample_meta, tmp_db):
        tmp_db.save_evolved_strategy(sample_meta)
        evolver.activate_strategy(sample_meta)

        engine = StrategyEngine(StrategyConfig())
        active = evolver.get_active_strategies()
        engine.load_evolved_strategies(active)
        assert len(engine.strategies) == 7

        # Reload with empty list
        engine.load_evolved_strategies([])
        assert len(engine.strategies) == 6  # Back to built-in only

    def test_evolved_strategy_runs_in_engine(self, evolver, sample_meta, tmp_db):
        tmp_db.save_evolved_strategy(sample_meta)
        evolver.activate_strategy(sample_meta)

        engine = StrategyEngine(StrategyConfig())
        engine.load_evolved_strategies(evolver.get_active_strategies())

        np.random.seed(42)
        n = 100
        base = np.linspace(100, 110, n) + np.random.randn(n) * 0.5
        df = pd.DataFrame({
            "timestamp": range(n),
            "open": base + np.random.randn(n) * 0.1,
            "high": base + abs(np.random.randn(n) * 0.3),
            "low": base - abs(np.random.randn(n) * 0.3),
            "close": base,
            "volume": np.random.randint(1000, 50000, n).astype(float),
        })

        market_data = {"BTC/USDC:USDC": {"15m": df, "1h": df}}
        signals = engine.generate_signals(market_data)
        # Should not crash — signals can be empty, that's fine
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# Safety rails tests
# ---------------------------------------------------------------------------


class TestSafetyRails:

    def test_forbidden_tokens_list_complete(self):
        """Ensure FORBIDDEN_TOKENS blocks the most dangerous patterns."""
        dangerous = ["import os", "import subprocess", "eval(", "exec(", "open("]
        for d in dangerous:
            assert d in FORBIDDEN_TOKENS, f"{d!r} should be in FORBIDDEN_TOKENS"

    def test_allowed_imports_whitelist(self):
        """Verify the import whitelist is restrictive."""
        assert "os" not in ALLOWED_IMPORTS
        assert "sys" not in ALLOWED_IMPORTS
        assert "subprocess" not in ALLOWED_IMPORTS
        assert "pandas" in ALLOWED_IMPORTS
        assert "numpy" in ALLOWED_IMPORTS

    def test_max_active_strategies_constant(self):
        assert MAX_ACTIVE_STRATEGIES == 5
