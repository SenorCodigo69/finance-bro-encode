"""Strategy evolver — uses Claude to generate new trading strategies from performance data.

Claude reviews win rates, P&L, and market regime to generate Python code for new
BaseStrategy subclasses. Strategies are validated in a sandbox, stored as .py files
with metadata, and loaded into the StrategyEngine at runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import pandas as pd

from src.config import StrategyConfig
from src.database import Database
from src.models import Signal
from src.strategy import BaseStrategy
from src.utils import log, now_iso

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVOLVED_DIR = PROJECT_ROOT / "data" / "evolved_strategies"

MAX_ACTIVE_STRATEGIES = 5

# Modules allowed in generated strategy code — anything outside this is rejected.
ALLOWED_IMPORTS = frozenset({
    "__future__",
    "pandas", "pd",
    "numpy", "np",
    "math",
    "src.indicators",
    "src.models",
    "src.config",
    "src.utils",
})

# Dangerous tokens that must never appear in generated code.
FORBIDDEN_TOKENS = [
    # Direct dangerous imports
    "import os", "import sys", "import subprocess", "import shutil",
    "import socket", "import http", "import urllib", "import requests",
    "import importlib", "import ctypes", "import pickle", "import marshal",
    "import io", "import codecs", "import posixpath", "import ntpath",
    # Dynamic code execution
    "__import__", "eval(", "exec(", "compile(",
    # Filesystem access
    "open(", "Path(", "pathlib",
    "read_text", "write_text", "read_bytes", "write_bytes",
    # Introspection / sandbox escape
    "globals()", "locals()",
    "getattr(", "setattr(", "delattr(",
    "__builtins__", "__class__", "__subclasses__",
    "__bases__", "__mro__", "__getattribute__",
    "type(", "vars(",
    # System access
    ".system(", ".popen(",
    # Dynamic import
    "importlib", "import_module",
]

# Template that Claude must fill in — ensures structural consistency.
STRATEGY_TEMPLATE = '''\
"""Auto-generated strategy: {name}

Generated: {timestamp}
Rationale: {rationale}
"""

from __future__ import annotations

import pandas as pd

from src.config import StrategyConfig
from src.indicators import compute_all
from src.models import Signal
from src.strategy import BaseStrategy
from src.utils import now_iso


class EvolvedStrategy(BaseStrategy):
    """Auto-generated strategy: {name}"""

    name = "{name}"

    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig) -> Signal | None:
{body}
'''


@dataclass
class StrategyPerformance:
    """Aggregated performance stats for a single strategy."""
    strategy_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_confidence: float = 0.0
    recent_streak: int = 0  # positive = wins, negative = losses


@dataclass
class EvolvedStrategyMeta:
    """Metadata for an evolved strategy stored in the DB."""
    id: int | None = None
    name: str = ""
    file_path: str = ""
    code_hash: str = ""
    generation: int = 1
    parent_strategies: list[str] = field(default_factory=list)
    rationale: str = ""
    status: str = "candidate"  # candidate | active | retired
    created_at: str = ""
    activated_at: str | None = None
    retired_at: str | None = None
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0


class StrategyEvolver:
    """Generates, validates, and manages evolved trading strategies via Claude."""

    def __init__(self, api_key: str, model: str, db: Database):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.db = db
        self._total_tokens = 0
        self._loaded_strategies: dict[str, BaseStrategy] = {}

        # Ensure storage directory exists
        EVOLVED_DIR.mkdir(parents=True, exist_ok=True)

        # Ensure DB table exists
        self.db.init_evolved_strategies_table()

        # Load any previously-active strategies
        self._load_active_strategies()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_strategies(self, lookback_trades: int = 50) -> dict[str, StrategyPerformance]:
        """Pull recent closed trades and compute per-strategy performance."""
        trades = self.db.get_recent_trades(lookback_trades)
        closed = [t for t in trades if t.pnl is not None]

        perf_map: dict[str, StrategyPerformance] = {}

        for t in closed:
            strat_name = t.signal_data.get("strategy", "unknown")
            if strat_name not in perf_map:
                perf_map[strat_name] = StrategyPerformance(strategy_name=strat_name)
            p = perf_map[strat_name]
            p.total_trades += 1
            p.total_pnl += t.pnl
            if t.pnl > 0:
                p.wins += 1
                p.max_win = max(p.max_win, t.pnl)
            else:
                p.losses += 1
                p.max_loss = min(p.max_loss, t.pnl)

        for p in perf_map.values():
            if p.total_trades > 0:
                p.win_rate = p.wins / p.total_trades
                p.avg_pnl = p.total_pnl / p.total_trades

        # Compute recent streak per strategy
        for t in reversed(closed):
            strat_name = t.signal_data.get("strategy", "unknown")
            p = perf_map.get(strat_name)
            if p is None:
                continue
            if p.recent_streak == 0:
                p.recent_streak = 1 if t.pnl > 0 else -1
            elif p.recent_streak > 0 and t.pnl > 0:
                p.recent_streak += 1
            elif p.recent_streak < 0 and t.pnl < 0:
                p.recent_streak -= 1
            # Streak broken — stop counting for this strategy
            # (we only track from the most recent trade backward, handled by reversed order)

        log.info(f"Evaluated {len(perf_map)} strategies across {len(closed)} closed trades")
        return perf_map

    def generate_strategy(
        self,
        performance: dict[str, StrategyPerformance],
        market_regime: str = "unknown",
        hint: str = "",
    ) -> EvolvedStrategyMeta | None:
        """Ask Claude to generate a new strategy based on performance data.

        Returns metadata for the generated strategy, or None on failure.
        """
        active_count = self._count_active()
        if active_count >= MAX_ACTIVE_STRATEGIES:
            log.warning(
                f"Already {active_count} active evolved strategies "
                f"(max {MAX_ACTIVE_STRATEGIES}). Retire one first."
            )
            return None

        perf_summary = self._format_performance(performance)

        system_prompt = textwrap.dedent("""\
            You are an expert quantitative trading strategist. Your job is to write a new
            Python trading strategy class for a crypto trading bot.

            The bot uses pandas DataFrames with OHLCV data and technical indicators computed
            by `src.indicators.compute_all()`. Available indicator columns after enrichment:
                ema_fast, ema_slow, sma_20, rsi, macd, macd_signal, macd_hist,
                bb_upper, bb_middle, bb_lower, atr, adx, stoch_k, stoch_d,
                obv, volume_sma, volume_ratio

            Raw columns: open, high, low, close, volume

            The strategy must subclass BaseStrategy and implement `analyze()`:

                def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig) -> Signal | None:
                    ...

            Key rules:
            1. Return Signal(pair=pair, timeframe=tf, direction="long"|"short",
               confidence=<0.0-1.0>, strategy_name=self.name,
               indicators={...}, reasoning="...", timestamp=now_iso())
               OR return None for no signal.
            2. Use `compute_all(df, config.__dict__)` to enrich the DataFrame.
            3. Use `self._get_latest(enriched)` to get the latest complete candle.
            4. Check for NaN values before using indicators.
            5. Keep the logic under 80 lines.
            6. Do NOT import os, sys, subprocess, open files, or use eval/exec.
            7. Only import from: pandas, numpy, math, src.indicators, src.models, src.utils.
            8. Be creative but grounded — combine indicators in novel ways.

            You must respond with ONLY the Python code for the analyze() method body
            (the part that goes inside the method, indented with 8 spaces).
            Do NOT include the class definition, imports, or method signature — only the body.
            Also include a one-line comment at the top describing the strategy idea.

            After the code block, on a new line, write:
            STRATEGY_NAME: <snake_case_name>
            RATIONALE: <one sentence explaining why this strategy should work given the performance data>
        """)

        user_msg = f"""## Current Strategy Performance
{perf_summary}

## Market Regime
{market_regime}

## Additional Context
{hint or "No additional hints."}

## Existing Built-in Strategies
- momentum: RSI oversold/overbought + MACD histogram direction + volume spike
- trend_following: EMA crossover + ADX strength + BB position
- mean_reversion: BB band touch + RSI extremes + stochastic
- breakout: 20-period high/low break + volume confirmation

## Active Evolved Strategies
{self._list_active_names() or "None yet."}

Generate a NEW strategy that complements the existing ones. Focus on patterns or
market conditions that the current strategies are missing. Avoid duplicating logic
that already exists."""

        try:
            response = self._call_claude(system_prompt, user_msg, max_tokens=3000)
        except Exception as e:
            log.error(f"Claude API call failed during strategy generation: {e}")
            return None

        return self._parse_and_store(response, performance)

    def validate_strategy(self, meta: EvolvedStrategyMeta) -> tuple[bool, str]:
        """Validate a generated strategy by:
        1. Static analysis (forbidden tokens, allowed imports)
        2. Loading in a sandbox module
        3. Running it against synthetic data to ensure it returns Signal | None
        """
        file_path = Path(meta.file_path)
        if not file_path.exists():
            return False, f"Strategy file not found: {file_path}"

        code = file_path.read_text()

        # --- Static checks ---
        for token in FORBIDDEN_TOKENS:
            if token in code:
                return False, f"Forbidden token found: {token!r}"

        # Check imports are allowed
        import_lines = re.findall(r"^\s*(?:from|import)\s+(\S+)", code, re.MULTILINE)
        for imp in import_lines:
            root = imp.split(".")[0]
            if root not in ALLOWED_IMPORTS and imp not in ALLOWED_IMPORTS:
                # Allow importing from our own strategy module (BaseStrategy)
                if not imp.startswith("src."):
                    return False, f"Disallowed import: {imp}"

        # --- Dynamic sandbox load ---
        try:
            strategy_instance = self._load_module(meta.name, file_path)
        except Exception as e:
            return False, f"Failed to load strategy module: {e}"

        if not isinstance(strategy_instance, BaseStrategy):
            return False, "Loaded object is not a BaseStrategy subclass"

        if not hasattr(strategy_instance, "analyze"):
            return False, "Strategy has no analyze() method"

        # --- Test with synthetic data ---
        try:
            result = self._run_synthetic_test(strategy_instance)
        except Exception as e:
            return False, f"Synthetic test failed: {e}"

        if result is not None and not isinstance(result, Signal):
            return False, f"analyze() returned {type(result).__name__}, expected Signal or None"

        log.info(f"Strategy '{meta.name}' passed validation")
        return True, "OK"

    def activate_strategy(self, meta: EvolvedStrategyMeta) -> bool:
        """Activate a validated strategy so StrategyEngine can use it."""
        if self._count_active() >= MAX_ACTIVE_STRATEGIES:
            log.warning(f"Cannot activate '{meta.name}' — already at max ({MAX_ACTIVE_STRATEGIES})")
            return False

        valid, reason = self.validate_strategy(meta)
        if not valid:
            log.warning(f"Cannot activate '{meta.name}' — validation failed: {reason}")
            return False

        # Load the strategy instance
        try:
            instance = self._load_module(meta.name, Path(meta.file_path))
        except Exception as e:
            log.error(f"Failed to load strategy for activation: {e}")
            return False

        self._loaded_strategies[meta.name] = instance
        meta.status = "active"
        meta.activated_at = now_iso()
        self.db.update_evolved_strategy(meta)

        log.info(f"Activated evolved strategy: {meta.name}")
        return True

    def retire_worst(self, min_trades: int = 5) -> EvolvedStrategyMeta | None:
        """Retire the worst-performing active evolved strategy.

        Only considers strategies with at least `min_trades` trades.
        Returns the retired metadata, or None if nothing to retire.
        """
        active = self.db.get_evolved_strategies(status="active")
        if not active:
            log.info("No active evolved strategies to retire")
            return None

        # Only retire strategies with enough data
        eligible = [m for m in active if m.total_trades >= min_trades]
        if not eligible:
            log.info(f"No active evolved strategies with >= {min_trades} trades to retire")
            return None

        # Find worst by win_rate, tie-break by total_pnl
        worst = min(eligible, key=lambda m: (m.win_rate, m.total_pnl))

        worst.status = "retired"
        worst.retired_at = now_iso()
        self.db.update_evolved_strategy(worst)

        # Unload from memory
        self._loaded_strategies.pop(worst.name, None)

        log.info(
            f"Retired evolved strategy '{worst.name}': "
            f"{worst.wins}/{worst.total_trades} wins, P&L ${worst.total_pnl:+.2f}"
        )
        return worst

    def get_active_strategies(self) -> list[BaseStrategy]:
        """Return loaded strategy instances for the StrategyEngine to use."""
        return list(self._loaded_strategies.values())

    def get_leaderboard(self) -> list[EvolvedStrategyMeta]:
        """Return all evolved strategies sorted by performance (best first)."""
        all_strats = self.db.get_evolved_strategies()
        # Sort by win_rate desc, then total_pnl desc
        all_strats.sort(key=lambda m: (m.win_rate, m.total_pnl), reverse=True)
        return all_strats

    def update_strategy_stats(self, strategy_name: str, pnl: float):
        """Update win/loss/P&L counters after a trade closes.

        Called by the main agent loop when a trade using an evolved strategy closes.
        """
        meta = self.db.get_evolved_strategy_by_name(strategy_name)
        if meta is None:
            return  # Not an evolved strategy — skip
        meta.total_trades += 1
        meta.total_pnl += pnl
        if pnl > 0:
            meta.wins += 1
        else:
            meta.losses += 1
        meta.win_rate = meta.wins / meta.total_trades if meta.total_trades > 0 else 0.0
        self.db.update_evolved_strategy(meta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_claude(self, system: str, user: str, max_tokens: int = 2000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text
        self._total_tokens += response.usage.input_tokens + response.usage.output_tokens
        return text

    def _format_performance(self, perf: dict[str, StrategyPerformance]) -> str:
        if not perf:
            return "No performance data yet — this is the first generation."
        lines = []
        for name, p in sorted(perf.items(), key=lambda x: x[1].total_pnl, reverse=True):
            lines.append(
                f"- {name}: {p.total_trades} trades, "
                f"{p.win_rate:.0%} win rate, "
                f"P&L ${p.total_pnl:+.2f} (avg ${p.avg_pnl:+.2f}), "
                f"streak {p.recent_streak:+d}"
            )
        return "\n".join(lines)

    def _list_active_names(self) -> str:
        return ", ".join(self._loaded_strategies.keys())

    def _count_active(self) -> int:
        return len(self.db.get_evolved_strategies(status="active"))

    def _parse_and_store(
        self,
        response: str,
        performance: dict[str, StrategyPerformance],
    ) -> EvolvedStrategyMeta | None:
        """Parse Claude's response, assemble full .py file, write to disk, store metadata."""
        # Extract strategy name
        name_match = re.search(r"STRATEGY_NAME:\s*(\S+)", response)
        if not name_match:
            log.warning("Claude response missing STRATEGY_NAME line")
            return None
        raw_name = name_match.group(1).strip()
        # Sanitize — only allow alphanumeric + underscore
        strategy_name = re.sub(r"[^a-z0-9_]", "", raw_name.lower())
        if not strategy_name:
            log.warning(f"Invalid strategy name after sanitization: {raw_name!r}")
            return None

        # Extract rationale
        rationale_match = re.search(r"RATIONALE:\s*(.+)", response)
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided"

        # Extract code body — everything in ```python ... ``` or before STRATEGY_NAME
        code_body = self._extract_code_body(response)
        if not code_body:
            log.warning("Could not extract code body from Claude response")
            return None

        # Ensure the body is indented at 8 spaces (method body), preserving relative indent
        raw_lines = code_body.split("\n")
        # Find the minimum indent of non-empty lines
        min_indent = float("inf")
        for line in raw_lines:
            stripped = line.lstrip()
            if stripped:
                min_indent = min(min_indent, len(line) - len(stripped))
        if min_indent == float("inf"):
            min_indent = 0

        body_lines = []
        for line in raw_lines:
            stripped = line.rstrip()
            if not stripped:
                body_lines.append("")
            else:
                # Remove the common base indent, then add 8 spaces
                relative = line[min_indent:].rstrip()
                body_lines.append("        " + relative)
        body = "\n".join(body_lines)

        # Assemble full file
        timestamp = now_iso()
        full_code = STRATEGY_TEMPLATE.format(
            name=strategy_name,
            timestamp=timestamp,
            rationale=rationale,
            body=body,
        )

        # Prepend the BaseStrategy import (template uses it in the class def)
        full_code = full_code.replace(
            "class EvolvedStrategy(BaseStrategy):",
            "from src.strategy import BaseStrategy\n\n\nclass EvolvedStrategy(BaseStrategy):",
        )

        # Compute code hash for dedup
        code_hash = hashlib.sha256(body.encode()).hexdigest()[:16]

        # Check for duplicate
        existing = self.db.get_evolved_strategy_by_hash(code_hash)
        if existing:
            log.info(f"Duplicate strategy detected (hash={code_hash}), skipping")
            return None

        # Determine generation number
        parent_names = list(performance.keys())
        generation = 1
        for parent in parent_names:
            parent_meta = self.db.get_evolved_strategy_by_name(parent)
            if parent_meta and parent_meta.generation >= generation:
                generation = parent_meta.generation + 1

        # Write to disk (tmp first, rename after validation)
        file_name = f"{strategy_name}_{code_hash}.py"
        file_path = EVOLVED_DIR / file_name

        # Path containment check
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(EVOLVED_DIR.resolve())):
            log.error(f"Path traversal detected: {resolved}")
            return None

        tmp_path = file_path.with_suffix(".tmp")
        tmp_path.write_text(full_code)

        meta = EvolvedStrategyMeta(
            name=strategy_name,
            file_path=str(file_path),
            code_hash=code_hash,
            generation=generation,
            parent_strategies=parent_names,
            rationale=rationale,
            status="candidate",
            created_at=timestamp,
        )

        # Rename tmp to .py for validation (importlib needs .py extension)
        tmp_path.rename(file_path)

        # Validate — delete if it fails
        valid, reason = self.validate_strategy(meta)
        if not valid:
            file_path.unlink(missing_ok=True)
            log.warning(f"Generated strategy failed validation: {reason}")
            return None
        meta.id = self.db.save_evolved_strategy(meta)
        log.info(f"Generated new strategy '{strategy_name}' (gen {generation}, hash {code_hash})")
        return meta

    def _extract_code_body(self, response: str) -> str | None:
        """Extract the Python code body from Claude's response."""
        # Try fenced code block first
        fenced = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        # Fallback: everything before STRATEGY_NAME line, excluding markdown
        lines = []
        for line in response.split("\n"):
            if line.strip().startswith("STRATEGY_NAME:"):
                break
            if line.strip().startswith("RATIONALE:"):
                break
            # Skip markdown headers
            if line.strip().startswith("#"):
                continue
            lines.append(line)

        body = "\n".join(lines).strip()
        return body if body else None

    def _load_module(self, name: str, file_path: Path) -> BaseStrategy:
        """Dynamically load a strategy module and return the EvolvedStrategy instance."""
        # Path containment check — refuse to load files outside evolved directory
        resolved = Path(file_path).resolve()
        if not str(resolved).startswith(str(EVOLVED_DIR.resolve())):
            raise ImportError(f"Refusing to load file outside evolved directory: {resolved}")

        module_name = f"evolved_{name}"

        # Remove old module if reloading
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        strategy_cls = getattr(module, "EvolvedStrategy", None)
        if strategy_cls is None:
            sys.modules.pop(module_name, None)
            raise ImportError(f"Module {file_path} has no EvolvedStrategy class")

        return strategy_cls()

    def _run_synthetic_test(self, strategy: BaseStrategy) -> Signal | None:
        """Run the strategy against synthetic OHLCV data to check it doesn't crash.
        Times out after 5 seconds to prevent infinite loops."""
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

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

        config = StrategyConfig()
        data = {"15m": df, "1h": df}

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(strategy.analyze, data, "BTC/USDC:USDC", config)
            try:
                return future.result(timeout=5)
            except FuturesTimeout:
                raise TimeoutError("Strategy took > 5 seconds — possible infinite loop")

    def _load_active_strategies(self):
        """Load all strategies marked 'active' in the DB."""
        active = self.db.get_evolved_strategies(status="active")
        for meta in active:
            fp = Path(meta.file_path)
            if not fp.exists():
                log.warning(f"Active strategy file missing: {fp}, retiring")
                meta.status = "retired"
                meta.retired_at = now_iso()
                self.db.update_evolved_strategy(meta)
                continue
            try:
                instance = self._load_module(meta.name, fp)
                self._loaded_strategies[meta.name] = instance
                log.info(f"Loaded active evolved strategy: {meta.name}")
            except Exception as e:
                log.warning(f"Failed to load evolved strategy '{meta.name}': {e}")

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens
