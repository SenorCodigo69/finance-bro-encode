# Encode Hack Day — Battle Plan

**Date:** March 14, 2026 (9am-5pm, 8 hours)
**Prize:** Mac Mini 16GB
**Rules:** Existing projects OK, consequential code must ship during hours. Must involve agents/agentic workflows.
**Venue:** Encode Hub (bring laptop, charger, headphones)

---

## What We're Shipping (3 Features)

### Feature 1: Agent-to-Agent Negotiation Engine (THE WOW FACTOR)
Multiple AI agents with different risk profiles **debate and negotiate** trade decisions in real-time. Not just voting — actual back-and-forth reasoning, compromise proposals, veto power.

**Why it wins:** Judges see 4 AI agents with distinct personalities arguing about trades live. Way more impressive than a single model making decisions.

### Feature 2: Live Strategy Evolution Demo
The agent generates its own trading strategies at runtime using AI, validates them in a sandbox (forbidden tokens, timeout), backtests, and deploys winners. Self-improving agent.

**Why it wins:** Self-modifying code is inherently impressive. The sandbox security story is also strong.

### Feature 3: Real-Time Dashboard (What Judges SEE)
WebSocket-powered dashboard showing everything live: positions, P&L, model votes, negotiation debates, strategy evolution, alert feed.

**Why it wins:** You can't win if judges can't see it. This is the demo layer.

---

## Pre-Hack Checklist (Tonight)

- [x] Model fixes: Gemini 429 retry, Ollama fallback to Hetzner
- [x] FastAPI + uvicorn + websockets installed
- [ ] Install Ollama locally: `brew install ollama` then `ollama pull qwen3:4b`
  - Or rely on Hetzner fallback (needs wifi at venue)
- [ ] Verify wifi at venue will reach Hetzner (204.168.157.171:11434)
- [ ] Charge laptop, pack charger
- [ ] Have this plan open on phone/tablet as reference

---

## Hour-by-Hour Timeline

### Hour 1 (9:00-10:00) — NEGOTIATION ENGINE: Data Models + Profiles

**Create:** `src/negotiation.py`

Define 4 agent profiles with distinct risk personalities:

```
Alpha (Conservative)  — 10% max pos, 15% max DD, wants 60%+ win rate, prefers trends
Beta  (Balanced)      — 15% max pos, 25% max DD, wants 55%+ win rate, multi-timeframe
Gamma (Aggressive)    — 25% max pos, 30% max DD, accepts 50% win rate, momentum plays
Delta (Risk Sentinel) — no positions, VETO POWER, monitors correlation + systemic risk
```

Data models needed:
- `AgentProfile` — name, risk params, strategy focus, veto_power, weight
- `AgentOpinion` — agent's take on a signal (approved, size_mod, risk_score, reasoning)
- `NegotiationRecord` — full debate transcript (phases, conflicts, proposals, outcome)

**Files to touch:** `src/models.py` (add dataclasses), new `src/negotiation.py`

---

### Hour 2 (10:00-11:00) — NEGOTIATION ENGINE: Core Logic

**In `src/negotiation.py`:**

```python
class NegotiationEngine:
    async def negotiate_signal(signal, portfolio, context) -> (Signal, NegotiationRecord, bool):
        # Phase 1: Each agent analyzes signal through own risk lens (parallel)
        # Phase 2: Detect conflicts (direction splits, size disagreement, veto)
        # Phase 3: Up to 3 negotiation rounds (propose compromises)
        # Phase 4: Consensus → weighted average position size
```

Key: each agent gets a DIFFERENT system prompt with its risk personality. Same underlying LLM models, different personas.

The prompt difference is what makes it work:
- Alpha gets: "You are a conservative trader. Reject anything below 70% confidence..."
- Gamma gets: "You are an aggressive momentum trader. Seize breakouts..."
- Delta gets: "You are the risk sentinel. Your ONLY job is blocking catastrophic trades..."

**Wire into:** `src/multi_brain.py` — replace simple voting with negotiation call.

---

### Hour 3 (11:00-12:00) — STRATEGY EVOLUTION: Demo Script

**Create:** `scripts/demo_evolution.py`

The evolver (`src/strategy_evolver.py`) is fully built but needs 10+ trades to trigger. Create a demo script that:

1. Feeds mock performance data (simulates 20 trades, 60% win rate)
2. Triggers `generate_strategy()` — Claude generates a new strategy
3. Shows the sandbox validation (forbidden tokens check, 5s timeout test)
4. Activates the strategy
5. Runs it against live OHLCV data for 1 pair
6. Prints signals generated vs baseline strategies

```python
# Pseudocode
evolver = StrategyEvolver(api_key, model, db)
mock_perf = {"momentum": StrategyPerformance(total_trades=20, wins=12, ...)}
new_strat = await evolver.generate_strategy(mock_perf, "bull", hint="RSI divergence")
# Show generated code, validation results, activation
```

**Files to touch:** new `scripts/demo_evolution.py`, no changes to existing evolver

---

### Hour 4 (12:00-1:00) — DASHBOARD: FastAPI + WebSocket Backend

**Create:** `scripts/live_dashboard.py`

Upgrade from SimpleHTTPRequestHandler to FastAPI + WebSocket:

```python
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = gather_all_data()  # portfolio, trades, signals, votes, negotiations
        await websocket.send_json(data)
        await asyncio.sleep(5)

# Keep all existing REST endpoints from web_dashboard.py
@app.get("/api/portfolio") ...
@app.get("/api/trades") ...

# NEW endpoints
@app.get("/api/negotiations")  # Recent negotiation transcripts
@app.get("/api/model-accuracy")  # Per-model voting stats
@app.get("/api/evolved-strategies")  # Strategy leaderboard
```

Existing REST endpoints can be lifted directly from `scripts/web_dashboard.py` (6 endpoints already work).

**Files to touch:** new `scripts/live_dashboard.py`

---

### Hour 5 (1:00-2:00) — DASHBOARD: Frontend with Live Panels

Add to the dashboard HTML:

**New panels:**
1. **Negotiation Feed** — live debate transcript (Alpha says X, Gamma counters Y, Delta vetoes)
2. **Model Accuracy Heatmap** — per-model win rates by pair
3. **Strategy Leaderboard** — evolved strategies ranked by P&L
4. **Alert Console** — scrolling feed of agent events

Keep the existing panels (portfolio, equity curve, trades, signals).

Use WebSocket for real-time updates instead of 30s polling.

**Files to touch:** `scripts/live_dashboard.py` (embedded HTML, same pattern as existing)

---

### Hour 6 (2:00-3:00) — INTEGRATION + WIRING

Wire negotiation engine into main agent loop:

1. `src/main.py` — after signals generated, route through NegotiationEngine instead of MultiBrain
2. Dashboard reads negotiation records from DB
3. Strategy evolver outputs visible in dashboard leaderboard panel
4. Test full cycle: signal → negotiation → execution → dashboard shows everything

**Files to touch:** `src/main.py` (swap in negotiation), `src/database.py` (add negotiations table if needed)

---

### Hour 7 (3:00-4:00) — POLISH + DEMO PREP

1. Run full agent cycle with all 3 features active
2. Fix any bugs from integration
3. Record a backup screen recording (in case live demo fails)
4. Prepare talking points (see below)
5. Test dashboard on a second browser/device

---

### Hour 8 (4:00-5:00) — DEMO + JUDGING

**Demo script (3-5 min):**

1. **"This is an autonomous crypto trading agent running 24/7"** — show dashboard, 13 pairs, real Hyperliquid data
2. **"Today I built agent negotiation"** — trigger a signal, show 4 agents debating:
   - "Alpha rejected — too volatile"
   - "Gamma approved — momentum breakout"
   - "Beta proposed compromise — half position"
   - "Delta: no systemic risk, approved"
   - Final consensus: approved at 60% size
3. **"The agent writes its own strategies"** — trigger evolution demo:
   - Claude generates "RSI divergence" strategy
   - Sandbox validates (forbidden token check, timeout)
   - Strategy activates, generates first signal
4. **"All of this runs autonomously"** — show the dashboard, explain the safety rails (30% drawdown hard stop, consensus required, fail-closed brain)

**Key talking points:**
- Multi-model consensus (3 AI models: Claude + Gemini + Qwen voting on every trade)
- Fail-closed safety (if AI fails, all trades rejected — not the other way around)
- Self-improving (strategy evolution + per-model accuracy tracking)
- Real exchange (Hyperliquid DEX, not a simulation)

---

## Files Created During Hack (Consequential Code)

| File | Feature | Est. Lines |
|------|---------|-----------|
| `src/negotiation.py` | Negotiation engine + agent profiles | ~250 |
| `src/models.py` (additions) | AgentProfile, AgentOpinion, NegotiationRecord | ~40 |
| `scripts/demo_evolution.py` | Strategy evolution demo script | ~80 |
| `scripts/live_dashboard.py` | FastAPI + WebSocket dashboard | ~350 |
| `src/main.py` (modifications) | Wire in negotiation engine | ~20 |
| `src/database.py` (additions) | Negotiation records table | ~30 |
| **Total** | | **~770 lines** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Wifi down at venue | Ollama fallback to local (install tonight). Gemini + Claude work on mobile hotspot |
| LLM API slow/down | Each model has 45s timeout. Negotiation degrades to simple voting |
| Dashboard bugs | Keep existing web_dashboard.py as backup (already works) |
| Runs out of time | Features are independent. Dashboard alone is demo-able. Negotiation alone is demo-able |
| Agent crashes during demo | Run `--dry-run` mode (log only, no execution). Still shows full reasoning |

---

## Quick Reference

```bash
# Run agent (paper mode, single cycle for demo)
python -m src.main --once

# Run agent (dry-run, no execution — safest for demo)
python -m src.main --dry-run --once

# Run existing dashboard (backup)
python scripts/web_dashboard.py

# Run new live dashboard
python scripts/live_dashboard.py

# Run strategy evolution demo
python scripts/demo_evolution.py

# Run tests
pytest
```

---

## Architecture (What Exists)

```
src/
├── main.py              # Agent loop (orchestrator)
├── multi_brain.py       # 3-model consensus voting (Claude+Gemini+Qwen)
├── brain.py             # Single model fallback
├── providers.py         # LLM abstraction (Anthropic, OpenAI-compat, Ollama)
├── strategy.py          # 4 mechanical strategies + regime weighting
├── strategy_evolver.py  # Runtime strategy generation (FULLY BUILT)
├── risk_manager.py      # Position sizing, drawdown circuit breaker
├── executor.py          # Order execution
├── exchange.py          # Hyperliquid DEX interface
├── data_fetcher.py      # OHLCV + price fetching
├── indicators.py        # 11 technical indicators
├── regime.py            # Bull/Bear/Sideways detection
├── macro_analyst.py     # 20 macro data sources
├── ml_signals.py        # ML regime classifier + LSTM predictor
├── trigger_orders.py    # On-chain SL/TP
├── negotiation.py       # NEW: Agent negotiation engine
├── models.py            # Data models
├── database.py          # SQLite persistence
└── config.py            # Settings loader

scripts/
├── web_dashboard.py     # Existing dashboard (backup)
├── live_dashboard.py    # NEW: FastAPI + WebSocket dashboard
├── demo_evolution.py    # NEW: Strategy evolution demo
├── dashboard.py         # Terminal UI (Rich)
├── backtest.py          # Backtesting framework
└── train_ml.py          # ML model training
```
