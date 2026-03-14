# Prompt for Next Session

Copy-paste this into a new Claude Code terminal:

---

I'm continuing work on an autonomous crypto trading agent at `~/Desktop/claude_projects/finance_agent/`.

## Context
Read `SESSION-LOG.md`, `CLAUDE.md` for full history. Built from scratch across 25 sessions. Key points:

- **Python project** with venv at `.venv/`, 661 tests passing (`pytest tests/`)
- **GitHub repo**: github.com/SenorCodigo69/finance_agent (private)
- **Exchange: Hyperliquid DEX** — permissionless, no KYC, USDC settlement, 0.045% fees
- **LIVE TRADING ACTIVE** — $19.80 USDC on Hyperliquid mainnet (BTC + ETH + GOLD)
- **Paper mode also available** — $10k capital, 15 pairs, full config
- **3-model brain:** Gemini 2.5 Flash (paid) + Claude Haiku 4.5 + Ollama/Qwen3:4b. Consensus voting + 4-agent negotiation engine
- **6 strategies:** Momentum, Trend Following, Mean Reversion, Breakout, Range, Flight-to-Safety
- **Dashboard:** Mac System 6 brutalist UI, Redaction 35 font, dark/light toggle, pixel dissolve splash
- **Server:** `root@204.168.157.171` — two services: `finance-agent-live` (live $20) + `finance-dashboard` (port 8421)
- **USDC only** — no USDT (EU MiCA regulations)
- **DEX only** — no CEX ever

## What Was Done Last Session (Session 25)
- Dashboard redesign: System 6 brutalist, B&W, Redaction 35, dark mode toggle
- Live trading: Hyperliquid SDK integration, $19.80 USDC deployed
- Flight-to-safety strategy: 5-factor crisis detection, gold rotation
- Security audit: all HIGH/MEDIUM issues fixed
- Deployed to Hetzner server

## Server Access
```bash
ssh root@204.168.157.171
ssh -L 8421:localhost:8421 root@204.168.157.171  # Dashboard tunnel
systemctl status finance-agent-live finance-dashboard
journalctl -u finance-agent-live -f  # Live logs
```

## Config Files
- `config/settings.yaml` — paper mode, 15 pairs, $10k
- `config/settings-live-20.yaml` — live mode, BTC+ETH+GOLD, $20
- `.env` — API keys (Anthropic, Gemini, Hyperliquid wallet)

## Known Issues
- Hetzner cloud firewall blocks port 8421 externally — use SSH tunnel
- Strategy evolution demo sometimes generates invalid code (LLM code gen)
- Model accuracy + trades empty until agent makes and closes trades
- Rotate Hyperliquid API key (was shared in conversation)

## Next Steps
1. Monitor live trading results
2. Add more pairs when comfortable (config change only)
3. Consider opening port 8421 via Hetzner cloud console
4. Hackathon pitch prep if not done
5. Synthesis hackathon work (Days 6-10)
