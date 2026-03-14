# CLAUDE.md

## Project Overview

Autonomous crypto trading agent. Multi-model AI brain (Gemini + Claude + Qwen) vets mechanical trading signals via consensus voting, manages risk, and learns from its own performance. Deployed 24/7 on Hetzner server, paper trading by default.

- **Exchange:** Hyperliquid DEX (permissionless, no KYC, USDC settlement, 0.045% fees)
- **10 pairs:** BTC, ETH, AAVE (crypto) + AAPL, TSLA, NVDA, MSFT (synth stocks) + Gold, Oil, Silver (commodities)
- **Capital:** $10,000 (paper), 25% max position, 30% max drawdown hard stop
- **Server:** `root@204.168.157.171` — systemd service, auto-restart, logs at `/var/log/finance-agent.log`
- **USDC only** — no USDT (EU MiCA regulations)
- **DEX only** — no CEX ever for execution (Binance/Bybit for data fallback only)

## Commands

```bash
python -m venv .venv && source .venv/bin/activate  # Setup venv
pip install -r requirements.txt                      # Install deps
python -m src.main                                   # Run agent (paper mode)
python -m src.main --once                            # Single cycle
python -m src.main --dry-run                         # Log only, no execution
python -m src.main --live                            # Live trading (CAREFUL)
pytest                                               # Run tests
```

### Server Commands

```bash
ssh root@204.168.157.171 "tail -100 /var/log/finance-agent.log"          # Check logs
ssh root@204.168.157.171 "systemctl status finance-agent"                 # Service status
ssh root@204.168.157.171 "systemctl restart finance-agent"                # Restart
ssh root@204.168.157.171 "sqlite3 /opt/finance_agent/data/agent.db 'SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT 5;'"  # Check DB
```

### Deploy to Server

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'data/agent.db' --exclude 'data/agent.db-shm' --exclude 'data/agent.db-wal' \
  -e "ssh -i ~/.ssh/id_ed25519" \
  ~/Desktop/claude_projects/finance_agent/ root@204.168.157.171:/opt/finance_agent/ \
  && ssh -i ~/.ssh/id_ed25519 root@204.168.157.171 "systemctl restart finance-agent"
```

## Architecture

- `src/main.py` — Agent loop orchestrator
- `src/config.py` — Settings loader (.env + YAML)
- `src/models.py` — Data models (Trade, Signal, etc.)
- `src/database.py` — SQLite persistence
- `src/exchange.py` — ccxt wrapper (paper + live via Hyperliquid)
- `src/data_fetcher.py` — OHLCV + multi-source price fetching
- `src/data_sources.py` — 3x price sources per asset class (cross-validation, anomaly detection)
- `src/indicators.py` — 11 technical indicators (RSI, MACD, BB, ATR, ADX, etc.)
- `src/strategy.py` — 4 mechanical strategies + regime-aware weighting
- `src/regime.py` — Market regime detection (BULL/BEAR/SIDEWAYS)
- `src/brain.py` — Claude AI for signal vetting + learning
- `src/multi_brain.py` — Multi-model consensus voting + per-model accuracy tracking
- `src/macro_analyst.py` — 20 macro data sources
- `src/ml_signals.py` — ML regime classifier, LSTM price predictor, RL position sizer
- `src/risk_manager.py` — Position sizing, stop losses, drawdown circuit breaker
- `src/executor.py` — Order execution
- `src/execution_router.py` — Cross-DEX venue selection
- `src/dex_scanner.py` — Multi-venue price/funding/OI scanner
- `src/trigger_orders.py` — On-chain SL/TP (execute even if bot offline)
- `src/tradfi_intel.py` — Earnings calendar, options IV, FRED, correlation matrix
- `src/strategy_evolver.py` — Runtime strategy generation (sandboxed)
- `src/portfolio.py` — Portfolio state + P&L
- `src/journal.py` — Trade journal logging
- `src/alerts.py` — Alert system

## Key Constraints

- Paper mode by default — must explicitly switch to live
- 30% max drawdown hard stop — hardcoded safety rail, config can't override
- 25% max position size — hardcoded cap in risk_manager.py
- All trades logged with reasoning
- Multi-model consensus required before execution (2/3 majority)
- Risk manager cannot be bypassed
- Fail-closed brain — API failure = reject all signals
- Strategy sandbox — 30+ forbidden tokens, import allowlist, 5s timeout
- No KOL/Twitter sentiment — rejected as noise
- Whale signals are adversarial — never use as standalone triggers
- No Google Trends — rejected as noise

## Key Docs

| File | Purpose |
|---|---|
| `SESSION-LOG.md` | Engineering journal — detailed log of every session |
| `NEXT-SESSION-PROMPT.md` | Copy-paste context prompt for starting the next session |
| `ROADMAP.md` | 8-phase roadmap with milestones and principles |
| `SECURITY-AUDIT.md` | Full security audit trail (Sessions 1-8) |
| `config/settings.yaml` | All runtime config (pairs, risk, models, data sources) |
| `docs/hackathon/` | Synthesis hackathon docs (separate from main agent) |

## End-of-Session Protocol (MANDATORY)

**Before the conversation ends, you MUST do ALL of the following.** If the user says "done", "wrap up", "that's it", "end session", "save everything", or is clearly finishing work — execute this protocol automatically. Also offer to run it if the conversation has been long and productive.

### 1. Update SESSION-LOG.md

Append a new section following the existing format:

```markdown
# Session N — [Short Title] (YYYY-MM-DD)

## What Was Done
### [Feature/Fix/Change Name]
- Bullet points of what was built/changed
- Include specific details: files changed, lines added, design decisions

## Stats
- **X/X tests passing** (was Y)
- **N files changed**, ~M insertions
- Any other relevant metrics

### Git Log
\```
[paste `git log --oneline` for this session's commits]
\```
```

Determine the session number by reading the last entry in SESSION-LOG.md and incrementing by 1.

### 2. Update NEXT-SESSION-PROMPT.md

Rewrite completely with:
- Current project context (what's deployed, what's running)
- What was done this session (brief)
- Roadmap status table (update percentages)
- Concrete tasks for next session (numbered, actionable)
- Server details
- Known issues (current, not stale)
- User preferences section
- Commands section

### 3. Update ROADMAP.md (if applicable)

If this session completed or advanced any roadmap phase, update the roadmap to reflect current status.

### 4. Commit the docs

```bash
git add SESSION-LOG.md NEXT-SESSION-PROMPT.md ROADMAP.md
git commit -m "Session N log + next-session prompt"
```

### 5. Hackathon conversation log (if hackathon work was done)

Update `docs/hackathon/CONVERSATION-LOG.md` with the session entry following its format (decisions, pivots, breakthroughs, agent/human contributions).

### Rules
- NEVER skip this protocol — it's how session continuity works across conversations
- If you're unsure whether the session is ending, ask: "Want me to save the session log before we wrap?"
- Include ALL significant work, not just the last thing you did
- Be specific in the log — future Claude instances need enough detail to understand what happened
- Keep NEXT-SESSION-PROMPT.md self-contained — assume the next Claude has zero context beyond this file and CLAUDE.md
