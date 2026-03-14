# Finance Bro

*Your personal hedge fund.*

An autonomous crypto trading agent with multi-model AI consensus, agent-to-agent negotiation, and self-evolving strategies. Built for the **Encode Hackathon** (March 14, 2026).

![Dashboard](docs/ENCODE-HACK-PLAN.md)

## What It Does

Finance Bro is a 24/7 autonomous trading agent that:

- **3 AI models** (Claude, Gemini, Qwen) vote on every trade — 2/3 majority required
- **4 AI agents** with distinct risk personalities debate and negotiate trade decisions in real-time
- **6 trading strategies** including a flight-to-safety crisis detector
- **Self-improving** — generates its own strategies at runtime, validates in a sandbox
- **Live on Hyperliquid DEX** — trading real USDC, no CEX, no KYC
- **Fail-closed** — if any AI model fails, all trades are rejected

## Encode Hack Features (Built Today)

### 1. Agent Negotiation Engine (`src/negotiation.py`)
4 AI agents with distinct risk personalities debate every trade:
- **Alpha** (Conservative) — 10% max position, wants 60%+ win rate
- **Beta** (Balanced) — multi-timeframe confirmation, 55% threshold
- **Gamma** (Aggressive) — 25% max position, momentum seeker
- **Delta** (Risk Sentinel) — veto power when risk >= 8/10

Multi-phase negotiation: initial opinions → conflict detection → up to 2 compromise rounds → weighted consensus.

### 2. Live Dashboard (`scripts/live_dashboard.py`)
Mac OS System 6 brutalist aesthetic with Redaction 35 typography:
- Real-time WebSocket updates (5s refresh)
- Live negotiation debate transcripts with agent colors
- Dark/light mode toggle
- Pixel dissolve splash animation
- Portfolio, trades, signals, model accuracy, strategy leaderboard

### 3. Flight-to-Safety Strategy (`src/strategy.py`)
5-factor crisis detection:
1. Sharp drop below EMA-slow
2. RSI panic (<25) or blow-off top (>80)
3. Volatility spike (BB width >6%)
4. Death cross (fast EMA diverging)
5. Consecutive red candles (4+/5)

When triggered: longs safe havens (GOLD), shorts risk assets (crypto, tech).

### 4. Strategy Evolution (`src/strategy_evolver.py`)
Claude generates new trading strategies at runtime:
- Sandboxed execution (30+ forbidden tokens, import allowlist, 5s timeout)
- Backtested against live data before activation
- Per-strategy P&L tracking and leaderboard

## Architecture

```
src/
├── main.py              # Agent loop orchestrator
├── negotiation.py       # 4-agent negotiation engine
├── multi_brain.py       # 3-model consensus voting
├── strategy.py          # 6 strategies + flight-to-safety
├── strategy_evolver.py  # Runtime strategy generation
├── risk_manager.py      # Position sizing, drawdown circuit breaker
├── exchange.py          # Hyperliquid DEX (paper + live)
├── indicators.py        # 11 technical indicators
├── regime.py            # Bull/Bear/Sideways detection
├── macro_analyst.py     # 20 macro data sources
└── ml_signals.py        # ML regime classifier + LSTM predictor

scripts/
├── live_dashboard.py    # FastAPI + WebSocket dashboard
└── demo_evolution.py    # Strategy evolution demo
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in your API keys

# Paper trading (safe)
python -m src.main --once

# Live dashboard
python scripts/live_dashboard.py

# Trigger a negotiation demo
curl -X POST "http://localhost:8421/api/trigger?pair=BTC&direction=long&confidence=0.75"
```

## Safety Rails

- **30% max drawdown** hard stop (hardcoded, config can't override)
- **25% max position size** cap
- **Multi-model consensus** required (2/3 majority)
- **Fail-closed brain** — API failure = reject all signals
- **Strategy sandbox** — 30+ forbidden tokens, import allowlist, 5s timeout
- **Risk sentinel agent** with veto power

## Tech Stack

- **Python 3.12+** with asyncio
- **Hyperliquid DEX** — native REST API + SDK for live trading
- **AI Models:** Claude (Anthropic), Gemini (Google), Qwen (Ollama)
- **FastAPI + WebSocket** for real-time dashboard
- **SQLite** for persistence
- **661 tests** passing

## License

MIT
