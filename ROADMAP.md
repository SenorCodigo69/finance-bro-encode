# Roadmap

## Phase 0: Multi-Model Ensemble (Immediate)

### Step 1 — Free/cheap models first (prove the architecture)
- **Groq (Llama 3.3 70B)** — free, 14,400 req/day, 300+ tok/s
- **Google Gemini 2.5 Flash** — free, 250 req/day
- **Grok 4.1 Fast** (xAI API) — ~$1.50/mo
- Cost: ~$1.50/month total

### Step 2 — Add paid models once free tier is proven
- **Claude Haiku 4.5** — ~$13.50/mo, best nuanced reasoning
- **Qwen3 Max** (OpenRouter API) — ~$3-5/mo, won Alpha Arena competition
- Cost: ~$18-20/month for 5-model ensemble

### Step 3 — Add local model when hardware arrives
- 32GB machine: run Qwen3-14B locally ($0, always available)
- 48GB+ machine (future): run Qwen3-32B locally
- Local model = zero-cost fallback if APIs go down

### Consensus logic
- All models get same prompt (macro data + signal + portfolio state)
- Require 2/3 or 3/3 agreement to execute trades
- Log all opinions + final decision for post-analysis
- Track per-model accuracy over time — weight votes by historical performance
- Escalation: if models disagree on high-stakes signal, defer to highest-accuracy model

### Hardware (phased)
- **Now:** Beelink N100 16GB (~$150) + UPS (~$50) — 24/7 API-only server, 6W idle
- **Soon:** 32GB machine — add Qwen3-14B locally alongside API models
- **Later:** 48GB+ machine — Qwen3-32B locally, reduce API dependency

## Phase 1: Data Expansion + Fact Checking ✅ COMPLETE (Session 23)

### Implemented (20+ sources)
- ✅ On-chain data — BTC on-chain (blockchain.info, mempool.space), BGeometrics (MVRV, SOPR, exchange netflow)
- ✅ CryptoQuant on-chain analytics — MVRV, exchange flows, miner outflows, exchange reserves (optional API key)
- ✅ Funding rates — Hyperliquid native + Binance fallback + predicted funding
- ✅ Open interest — Hyperliquid native, cross-venue via DEX scanner
- ✅ Liquidation signals — derived from HL OI + funding (cascade scoring 0-3) + optional Coinglass API
- ✅ Order book depth — Hyperliquid L2 data
- ✅ Stablecoin flows — DeFi Llama (USDT/USDC mint/burn)
- ✅ Economic calendar — FOMC, CPI, NFP via Fair Economy
- ✅ Whale movements — Etherscan + blockchain.info known addresses
- ✅ SEC filings — EDGAR RSS for crypto ETF/regulatory filings
- ✅ GitHub activity — dev commits on major crypto + competitor repos
- ✅ Sentiment — Fear & Greed, Reddit, CryptoPanic, Hacker News, Polymarket, RSS news
- ✅ TradFi — Yahoo macro (DXY, VIX, yields), earnings calendar, options IV, correlation matrix
- ❌ Google Trends — rejected as noise
- ❌ Twitter/X KOL — rejected as noise
- SEC/regulatory filings — EDGAR RSS for crypto-related filings
- Whale wallet tracking — known whale address movements
- GitHub activity — dev commits on major crypto repos
- Mempool data — large pending transactions
- Polymarket / prediction markets — event-driven sentiment

### Data integrity machine
- **Cross-reference sources** — if Reddit says crash but Fear & Greed says greed, flag conflict
- **Anomaly detection** — price feed 10%+ off from other sources → discard
- **Source reliability scoring** — track which sources correlate with actual price moves, weight accordingly
- **Fake news filter** — Claude reviews headlines, flags misinformation / pump-and-dump shills
- **Sentiment consensus** — require 3+ sources to agree before weighting heavily
- **Stale data detection** — flag any source returning data older than expected
- **Source credibility score** — build per-source credibility over time
- **Prompt injection defense** — sanitize all external data before feeding to LLMs

## Phase 2: Strategy Perfection

### Per-pair tuning
- Fine-tune indicator parameters per pair (BTC and ETH behave differently)
- BTC → favor trend-following (tends to trend)
- ETH → favor mean-reversion (more volatile)
- Backtest across multiple timeframes and market regimes (bull/bear/sideways)
- Let strategy evolver generate pair-specific strategies

### Day trading / short-term strategy profile
- Aggressive config: tighter TP/SL (3-4%), 24h max hold time, 2-min cycles, looser RSI bands
- Already implemented: `config/settings-aggressive.yaml` + `--config` CLI flag + `max_hold_hours` auto-close
- Backtest aggressive vs conservative profiles across different market regimes
- Grid trading, DCA, scalping strategies for sideways markets (current strategies underperform in SIDEWAYS regime)

### Multi-agent architecture (strategy specialization)
- Run multiple trader agents simultaneously with different risk profiles:
  - **Conservative agent** — trend-following only, wide stops, long hold times, low leverage
  - **Aggressive day trader** — tight TP/SL, 24h max hold, high turnover, all strategies
  - **Mean-reversion scalper** — BB + RSI only, sub-hour holds, small positions
  - **Macro swing trader** — macro-driven, days-to-weeks holds, event-driven entries
- Each agent gets its own capital allocation, DB, and P&L tracking
- Compare agent performance over time — shift capital toward consistent winners
- Shared market data layer (one data fetcher, multiple strategy engines)
- Orchestrator process manages lifecycle, capital allocation, and aggregate risk limits

### Correlation analysis
- If BTC dumps, ETH usually follows — don't long ETH when BTC is dumping
- Cross-pair signal confirmation before entry
- Regime detection (trending vs ranging vs choppy)

### Stablecoin yield composability
- Park idle USDC in yield protocols (Aave, Morpho, Ethena) instead of 0%
- Auto-withdraw when trading signals appear
- Track yield as separate P&L line
- Yield-bearing positions as collateral → borrow against them → redeploy
- Route: Trading gains → stables → yield → collateral → more trading capital

## Phase 3: Competitive Intelligence — Learn From Other Agents

### Open-source agent analysis bot
- Periodically scan repos: ElizaOS, Freqtrade (FreqAI), Hummingbot, OctoBot, FinRL, PowerTrader AI
- Extract: new strategies, data sources, risk management techniques, architecture patterns
- Claude summarizes diffs/changelogs into actionable insights
- Flag features we're missing that winning agents have
- Track their backtest results vs ours

### Features to steal from the best
- **ElizaOS**: Trust scoring system (assign reliability scores to data sources/signals)
- **FreqAI**: Adaptive ML retraining (continuously retrain models on new market data)
- **Hummingbot**: Market making strategies, cross-exchange execution
- **FinRL**: Ensemble reinforcement learning (multiple RL agents voting)
- **AIXBT**: KOL monitoring network (track 400+ crypto influencers)
- **Olas**: Multi-chain autonomous execution (Base, Optimism, Arbitrum)

## Phase 4: ML + Advanced Signals

- ML-based signal generation (train on historical trades + outcomes)
- Reinforcement learning for position sizing (FinRL-style ensemble RL)
- CNN-LSTM price prediction (hybrid quant + LLM approach)
- Orderflow analysis (CVD, delta, footprint)
- Sentiment NLP models trained on crypto-specific language
- Regime classification model (auto-detect bull/bear/sideways)
- Continuous model retraining on new data (FreqAI approach — retrain every N cycles)

## Phase 5: Multi-Exchange + Cross-Chain

- **Multi-exchange execution** — reduce single-exchange risk (Hyperliquid + Bybit + others)
- **Jupiter/Raydium (Solana)** — DEX aggregator, arbitrage opportunities
- **Cross-DEX arbitrage** — scan 50+ order books for price divergence
- **MEV-aware execution** — Jito-MEV integration on Solana, flashbots on Ethereum L2s
- **Bridge arbitrage** — cross-chain price differences via fast bridges (Base ↔ Arbitrum ↔ Optimism)

## Phase 6: DeFi Integration

- Yield farming strategies (automated LP management)
- Flash loan arbitrage detection
- DEX vs CEX price divergence trading
- Separate risk model for DeFi (smart contract risk, impermanent loss)
- Gas optimization for on-chain execution
- Stablecoin yield routing (auto-find best rates across Aave/Compound/Morpho)

## Phase 7: Hyperliquid Synthetics — Stocks, Commodities, Bonds ✅ COMPLETE (Session 23)

### Synthetic stocks (perps on Hyperliquid)
- US equities as perps — AAPL, TSLA, NVDA, MSFT, etc. (24/7 trading, no broker)
- Stock index perps — S&P500, NASDAQ (when available)
- Adapt existing strategies to equity price dynamics (different volatility profile)
- Cross-asset correlation signals — crypto + equities moving together = risk-on regime

### Synthetic commodities
- Gold, oil, silver perps on Hyperliquid
- Macro hedge: gold rising + crypto rising = inflation narrative
- Cross-market correlation signals (DXY, VIX, 10Y yield)

### Synthetic bonds / rates
- Treasury yield perps (when available)
- Rate-sensitive strategy — tightening = risk-off, easing = risk-on
- Auto-adjust crypto exposure based on yield curve signals

### On-chain private equity / RWA
- Research RWA protocols: Securitize, Maple Finance, Centrifuge, Ondo Finance
- Tokenized private credit/equity — longer duration yield
- Route idle stablecoin gains into RWA yield when no trading signals active
- Separate risk model for RWA (illiquidity risk, smart contract risk, credit risk)

### Forex
- Forex pairs as perps — EUR/USD, GBP/USD, USD/JPY
- DXY-correlated trading (strong dollar = crypto headwind)

### Cross-market intelligence
- Unified signal engine across all asset classes
- Cross-asset momentum (if equities + commodities + crypto all dumping = systemic risk-off)
- Asset rotation strategy — auto-shift capital to strongest asset class

## Phase 8: Scale + Harden

- Prove consistent profitability on $1,000
- Scale to $5,000 → $10,000 → $50,000
- Hardware wallet for exchange API keys
- Mac Mini M4 Pro as dedicated always-on server (Phase 0 hardware)
- systemd watchdog + auto-restart + UPS ($50 CyberPower EC450G)
- Monitoring + alerting (Telegram alerts on trades, drawdown warnings, model disagreements)
- Multi-exchange execution (reduce single-exchange risk)
- Per-model accuracy dashboard — track which brain is best over time
- Quarterly strategy review: kill underperformers, double down on winners

## Principles

- Start small, prove it, scale up
- More data = better, but needs fact-checking first
- Conservative with user's money, aggressive with bot's money
- Perfect the core before expanding
- Every trade logged with full reasoning
- Multi-model consensus > single model confidence
- Learn from the best, build what they don't have
- No single point of failure — not one model, not one exchange, not one strategy, not one API
- Permissionless first — DEX > CEX, no KYC, no bank accounts, composable DeFi stack
