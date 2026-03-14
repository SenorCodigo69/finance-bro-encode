"""
Autonomous Crypto Trading Agent

Usage:
    python -m src.main                  # Run agent (paper mode by default)
    python -m src.main --once           # Single cycle then exit
    python -m src.main --dry-run        # Log decisions without executing
    python -m src.main --live           # Live trading (requires exchange API keys)
"""

from __future__ import annotations

import argparse
import asyncio
import math
import signal
import sys
import time
from dataclasses import asdict

from rich.panel import Panel
from rich.table import Table

from src.brain import Brain
from src.config import load_config
from src.data_fetcher import DataFetcher
from src.data_integrity import DataIntegrityChecker
from src.database import Database
from src.dex_scanner import DexScanner
from src.exchange import Exchange
from src.execution_router import ExecutionRouter
from src.executor import Executor
from src.journal import Journal
from src.macro_analyst import MacroAnalyst
from src.multi_brain import MultiBrain
from src.negotiation import NegotiationEngine
from src.portfolio import Portfolio
from src.providers import build_providers
from src.risk_manager import RiskManager
from src.strategy import StrategyEngine
from src.strategy_evolver import StrategyEvolver
from src.tradfi_intel import TradFiIntel
from src.trigger_orders import TriggerOrderManager
from src.utils import console, log


_shutdown = False


def handle_shutdown(signum, frame):
    global _shutdown
    _shutdown = True
    log.info("Shutdown signal received — finishing current cycle...")


async def cycle(
    fetcher: DataFetcher,
    strategy_engine: StrategyEngine,
    brain: Brain | MultiBrain | None,
    risk_manager: RiskManager,
    executor: Executor,
    portfolio: Portfolio,
    journal: Journal,
    db: Database,
    macro_analyst: MacroAnalyst | None = None,
    evolver: StrategyEvolver | None = None,
    integrity_checker: DataIntegrityChecker | None = None,
    tradfi_intel: TradFiIntel | None = None,
    dex_scanner: DexScanner | None = None,
    trigger_manager: TriggerOrderManager | None = None,
    negotiation_engine: NegotiationEngine | None = None,
    dry_run: bool = False,
    review_every_n_trades: int = 10,
) -> dict:
    """Run one trading cycle. Returns a summary dict."""
    summary = {"signals": 0, "approved": 0, "trades": 0, "closed": 0}

    # Step 1: Fetch market data
    log.info("Fetching market data...")
    market_data = await fetcher.fetch_all_pairs()
    current_prices = fetcher.get_latest_prices()

    if not current_prices:
        log.warning("No price data available — skipping cycle")
        return summary

    # Step 1b: Kick off external data fetches in parallel (non-blocking)
    # These are independent and would otherwise run sequentially (~20s → ~5s)
    async def _fetch_dex_scan():
        if not dex_scanner:
            return {}
        try:
            results = await dex_scanner.scan_all(fetcher.config.pairs)
            divergent = [
                (p, r.max_price_divergence_pct)
                for p, r in results.items()
                if r.max_price_divergence_pct > 0.5
            ]
            for pair, div in divergent:
                log.info(f"DEX divergence: {pair} {div:.2f}% across venues")
            return results
        except Exception as e:
            log.debug(f"DEX scan failed (non-critical): {e}")
            return {}

    async def _fetch_tradfi():
        if not tradfi_intel:
            return {}
        try:
            ohlcv_1h = {}
            for pair, tf_data in market_data.items():
                if "1h" in tf_data and not tf_data["1h"].empty:
                    ohlcv_1h[pair] = tf_data["1h"]
            return await tradfi_intel.get_full_context(
                pairs=fetcher.config.pairs,
                ohlcv_data=ohlcv_1h,
            )
        except Exception as e:
            log.warning(f"TradFi intelligence failed (non-critical): {e}")
            return {}

    async def _fetch_macro():
        if not macro_analyst:
            return None
        try:
            return await macro_analyst.get_macro_context()
        except Exception as e:
            log.warning(f"Macro context failed (non-critical): {e}")
            return None

    # Fire all three in parallel
    dex_task = asyncio.create_task(_fetch_dex_scan())
    tradfi_task = asyncio.create_task(_fetch_tradfi())
    macro_task = asyncio.create_task(_fetch_macro())

    # Step 2: Get portfolio snapshot (runs while external fetches are in flight)
    snap = await portfolio.get_snapshot(current_prices)
    log.info(
        f"Portfolio: ${snap.total_value:.2f} | "
        f"Cash: ${snap.cash:.2f} | "
        f"Drawdown: {snap.drawdown_pct:.1%} | "
        f"P&L: ${snap.total_pnl:+.2f} | "
        f"Bot: ${snap.bot_balance:.2f} | "
        f"User: ${snap.user_balance:.2f}"
    )

    # Step 3: Check drawdown circuit breaker
    if not risk_manager.check_drawdown(snap):
        log.critical("CIRCUIT BREAKER TRIGGERED — closing all positions")
        await executor.emergency_close_all("Max drawdown reached")
        db.set_state("agent_status", "PAUSED")
        return summary

    # Check if agent is paused
    status = db.get_state("agent_status")
    if status == "PAUSED":
        log.warning("Agent is PAUSED (drawdown limit hit). Reset manually to resume.")
        return summary

    # Step 4: Check stop losses and take profits
    closed = await executor.check_and_close_stops(current_prices)
    summary["closed"] = len(closed)
    if closed:
        log.info(f"Closed {len(closed)} trades (stop/TP hit)")
        # Feed closed-trade outcomes to per-model accuracy tracker
        if brain and hasattr(brain, "record_trade_outcome"):
            for trade in closed:
                if trade.pnl is not None:
                    brain.record_trade_outcome(
                        signal_id=trade.entry_time or str(trade.id),
                        pair=trade.pair,
                        direction=trade.direction,
                        was_profitable=trade.pnl > 0,
                    )

    # Step 4b: Sync on-chain trigger orders (live mode only)
    if trigger_manager:
        try:
            filled_triggers = await trigger_manager.sync_trigger_status()
            if filled_triggers:
                log.info(f"On-chain triggers filled: {len(filled_triggers)}")
        except Exception as e:
            log.debug(f"Trigger sync failed (non-critical): {e}")

    # Step 5: Update trailing stops
    open_trades = db.get_open_trades()
    if open_trades:
        # [OPT-3] Get ATR values using strategy engine's enriched cache
        atr_values = {}
        for pair, tf_data in market_data.items():
            tf = "1h" if "1h" in tf_data else list(tf_data.keys())[0] if tf_data else None
            if tf and len(tf_data[tf]) > 14:
                enriched = strategy_engine.get_enriched(tf_data[tf])
                last_atr = enriched["atr"].iloc[-1]
                if not math.isnan(last_atr):
                    atr_values[pair] = last_atr
        risk_manager.update_trailing_stops(open_trades, current_prices, atr_values)

        # Update on-chain trailing stop triggers
        if trigger_manager:
            for trade in open_trades:
                if trade.stop_loss and trade.trigger_orders_placed:
                    try:
                        await trigger_manager.update_trailing_triggers(trade, trade.stop_loss)
                    except Exception as e:
                        log.debug(f"Trailing trigger update failed for trade {trade.id}: {e}")

    # Step 5b: Await parallel external data fetches (launched in Step 1b)
    dex_scan_results = await dex_task
    tradfi_context = await tradfi_task
    macro_context = await macro_task

    # Inject TradFi context into risk manager
    if tradfi_context and tradfi_intel:
        try:
            correlated_map = {}
            for pair in fetcher.config.pairs:
                corr = tradfi_intel.correlations.get_correlated_pairs(pair)
                if corr:
                    correlated_map[pair] = corr

            iv_map = tradfi_context.get("options_iv", {})

            from src.config import EventBlockingConfig
            risk_manager.set_tradfi_context(
                event_blocking=EventBlockingConfig(),
                earnings_blackout_fn=tradfi_intel.earnings.is_earnings_blackout,
                correlated_pairs=correlated_map,
                iv_context=iv_map,
            )

            source_health = fetcher.get_source_health()
            if source_health.get("sources"):
                for src_name, stats in source_health["sources"].items():
                    log.debug(
                        f"Source {src_name}: avg={stats['avg_latency_ms']:.0f}ms, "
                        f"calls={stats['calls']}"
                    )

            if tradfi_context.get("earnings_blackouts"):
                log.info(f"Earnings blackouts: {tradfi_context['earnings_blackouts']}")
            if tradfi_context.get("correlation_warnings"):
                log.info(f"Correlation warnings: {len(tradfi_context['correlation_warnings'])} pair(s)")
        except Exception as e:
            log.warning(f"TradFi injection failed (non-critical): {e}")

    # Inject funding/OI from macro context into strategy engine
    if macro_context:
        try:
            macro_dict = macro_context.to_dict()
            raw_funding = macro_dict.get("funding_rates") or {}
            funding_rates: dict[str, float] = {
                coin: float(data.get("funding_rate", 0.0))
                for coin, data in raw_funding.items()
                if isinstance(data, dict) and "funding_rate" in data
            }

            open_interest: dict[str, dict] = {
                coin: {
                    "oi": float(data.get("open_interest", 0.0)),
                    "mark_price": float(data.get("mark_price", 0.0)),
                }
                for coin, data in raw_funding.items()
                if isinstance(data, dict) and "open_interest" in data
            }

            strategy_engine.set_market_signals(funding_rates, open_interest)
        except Exception as e:
            log.warning(f"Failed to inject funding/OI into strategy engine (non-critical): {e}")

    # Step 6b: Generate signals (regime detection runs inside generate_signals)
    signals = strategy_engine.generate_signals(market_data)
    summary["signals"] = len(signals)

    # [S7-01] Filter out signals for pairs with failed price validation or anomaly
    invalid_pairs = fetcher.get_invalid_pairs()
    if invalid_pairs and signals:
        blocked = [s for s in signals if s.pair in invalid_pairs]
        signals = [s for s in signals if s.pair not in invalid_pairs]
        for s in blocked:
            log.warning(f"Signal BLOCKED (price validation failed): {s.pair} {s.direction}")

    # Log current regime (set by generate_signals via RegimeDetector)
    if strategy_engine.current_regime is not None:
        log.info(f"Market regime: {strategy_engine.current_regime}")

    if signals:
        log.info(f"Generated {len(signals)} signal(s)")
        for sig in signals:
            log.info(f"  {sig.direction.upper()} {sig.pair} [{sig.strategy_name}] conf={sig.confidence:.2f}")

    # Step 7: AI vetting
    approved_signals: list[tuple] = []
    if signals and brain:
        market_context = _build_market_context(market_data, current_prices)

        # Pass regime info to the AI brain so it can factor it into decisions
        if strategy_engine.current_regime is not None:
            r = strategy_engine.current_regime
            market_context["market_regime"] = {
                "regime": r.regime,
                "confidence": r.confidence,
                "adx": r.adx,
                "volatility_pct": r.volatility_pct,
                "trend_direction": r.trend_direction,
            }

        # Enrich market context with macro analysis if available
        if macro_analyst:
            try:
                macro_analysis = await macro_analyst.get_ai_analysis()
                if macro_analysis:
                    market_context["macro_analysis"] = macro_analysis.to_dict()
                    log.info(
                        f"Macro outlook: {macro_analysis.outlook.upper()} "
                        f"(conf={macro_analysis.confidence:.0%})"
                    )
            except Exception as e:
                log.warning(f"Macro analysis failed (non-critical): {e}")

        # Enrich with TradFi intelligence
        if tradfi_context:
            if tradfi_context.get("macro_indicators"):
                market_context["fred_macro"] = tradfi_context["macro_indicators"]
            if tradfi_context.get("yield_spread_10y_2y") is not None:
                market_context["yield_spread_10y_2y"] = tradfi_context["yield_spread_10y_2y"]
            if tradfi_context.get("options_iv"):
                market_context["options_iv"] = tradfi_context["options_iv"]
            if tradfi_context.get("correlation_warnings"):
                market_context["correlation_warnings"] = tradfi_context["correlation_warnings"]
            if tradfi_context.get("upcoming_earnings"):
                market_context["upcoming_earnings"] = tradfi_context["upcoming_earnings"]

        # Enrich with source health
        source_health = fetcher.get_source_health()
        if source_health.get("per_pair"):
            market_context["source_health"] = source_health["per_pair"]

        # Enrich with DEX scan results
        if dex_scan_results:
            # Hardcoded venue allowlist to prevent injection via venue names
            _KNOWN_VENUES = {"hyperliquid", "dydx", "gmx"}
            dex_context = {}
            for pair, scan in dex_scan_results.items():
                if scan.max_price_divergence_pct > 0.1:
                    dex_context[str(pair)] = {
                        "divergence_pct": float(round(scan.max_price_divergence_pct, 3)),
                        "venues": [
                            str(v) for v in scan.venue_snapshots.keys()
                            if str(v) in _KNOWN_VENUES
                        ],
                        "best_bid_venue": str(scan.best_bid_venue) if scan.best_bid_venue in _KNOWN_VENUES else None,
                        "best_ask_venue": str(scan.best_ask_venue) if scan.best_ask_venue in _KNOWN_VENUES else None,
                        "arb_bps": float(round(scan.arb_opportunity_bps, 1)),
                        "funding": {
                            str(v): float(round(r, 6))
                            for v, r in scan.funding_divergence.items()
                            if str(v) in _KNOWN_VENUES
                        },
                    }
            if dex_context:
                market_context["dex_scan"] = dex_context

        # Run data integrity check before the brain sees the data
        if integrity_checker and market_context.get("macro_analysis"):
            integrity_report = integrity_checker.check(
                market_context.get("macro_analysis", {})
            )
            market_context["data_integrity"] = integrity_report.to_dict()

        # Use negotiation engine if available, otherwise fall back to MultiBrain
        if negotiation_engine:
            log.info("Routing signals through Agent Negotiation Engine...")
            vetted = await negotiation_engine.negotiate_signals(
                signals, snap, market_context
            )
        else:
            vetted = await brain.vet_signals(signals, snap, market_context)
        for sig, reasoning, ok in vetted:
            if ok:
                approved_signals.append((sig, reasoning))
            journal.log_signal(sig, acted_on=ok, reason="" if ok else reasoning)
        summary["approved"] = len(approved_signals)
    elif signals:
        # No brain available — approve all mechanical signals
        approved_signals = [(sig, "No AI vetting — mechanical approval") for sig in signals]
        summary["approved"] = len(approved_signals)

    # Step 8: Execute approved signals
    for signal_obj, ai_reasoning in approved_signals:
        pair = signal_obj.pair
        can_trade, block_reason = risk_manager.check_can_trade(snap, pair=pair)
        if not can_trade:
            journal.log_decision("Trade blocked", block_reason)
            continue

        # [OPT-3] Get ATR for position sizing using cached enriched data
        tf = signal_obj.timeframe
        atr_val = 0.0
        if pair in market_data and tf in market_data[pair]:
            enriched = strategy_engine.get_enriched(market_data[pair][tf])
            if len(enriched) > 0 and "atr" in enriched.columns:
                last_atr = enriched["atr"].iloc[-1]
                if not math.isnan(last_atr):
                    atr_val = last_atr

        if atr_val == 0:
            # Fallback: use default stop loss percentage
            price = current_prices.get(pair, 0)
            atr_val = price * 0.015  # 1.5% as ATR fallback

        trade = await executor.execute_signal(signal_obj, snap, ai_reasoning, atr_val, dry_run)
        if trade:
            summary["trades"] += 1
            snap = await portfolio.get_snapshot(current_prices)  # Refresh

    # Step 9: Save portfolio snapshot
    db.save_snapshot(snap)

    # Step 10: Check if AI review is due
    if brain:
        count_str = db.get_state("trades_since_review") or "0"
        # [OPT-11] review_every_n_trades is passed in; no need to reload config from disk
        if int(count_str) >= review_every_n_trades:
            recent = db.get_recent_trades(config.agent.review_every_n_trades)
            await brain.review_trades(recent, snap)
            db.set_state("trades_since_review", "0")

    # Step 11: Strategy evolution (every 20 cycles)
    if evolver:
        cycle_count = int(db.get_state("cycle_count") or "0") + 1
        db.set_state("cycle_count", str(cycle_count))

        if cycle_count % 20 == 0:
            try:
                log.info("Running strategy evolution check...")
                # Evaluate current performance
                performance = evolver.evaluate_strategies()
                if performance:
                    # Update stats for evolved strategies
                    recent_trades = db.get_recent_trades(50)
                    for t in recent_trades:
                        if t.pnl is not None:
                            strat_name = t.signal_data.get("strategy", "")
                            evolver.update_strategy_stats(strat_name, t.pnl)

                    # Try to generate a new strategy if we have enough data
                    total_trades = sum(p.total_trades for p in performance.values())
                    if total_trades >= 10:
                        regime = (
                            strategy_engine.current_regime.regime
                            if strategy_engine.current_regime is not None
                            else "unknown"
                        )
                        new_strat = await asyncio.to_thread(
                            evolver.generate_strategy, performance, regime
                        )
                        if new_strat:
                            log.info(f"New evolved strategy: {new_strat.name}")
                            # Reload evolved strategies into engine
                            active = evolver.get_active_strategies()
                            strategy_engine.load_evolved_strategies(active)
            except Exception as e:
                log.warning(f"Strategy evolution failed (non-critical): {e}")

    # Step 12: Log model accuracy report (every 20 cycles, if MultiBrain)
    if brain and hasattr(brain, "get_accuracy_report"):
        cycle_count_str = db.get_state("cycle_count") or "0"
        if int(cycle_count_str) % 20 == 0:
            try:
                report = brain.get_accuracy_report()
                if report.get("total_models_tracked", 0) > 0:
                    log.info(
                        f"Model accuracy report ({report['total_models_tracked']} models tracked):"
                    )
                    for model_name, stats in report.get("models", {}).items():
                        weight = report.get("weights", {}).get(model_name, 1.0)
                        log.info(
                            f"  [{model_name}] accuracy={stats['accuracy']:.1%} "
                            f"votes={stats['total_votes']} "
                            f"solo_pnl={stats['solo_pnl_estimate']:+d} "
                            f"contrarian={stats['contrarian_accuracy']:.1%} "
                            f"weight={weight:.2f}x"
                        )
                    if report.get("best_model"):
                        log.info(f"  Best model: {report['best_model']}")
            except Exception as e:
                log.debug(f"Accuracy report failed (non-critical): {e}")

    return summary


def _build_market_context(
    market_data: dict, current_prices: dict[str, float]
) -> dict:
    """Build a summary of current market conditions for the AI brain."""
    context = {}
    for pair, tf_data in market_data.items():
        # Use 1h candles for context
        tf = "1h" if "1h" in tf_data else list(tf_data.keys())[0] if tf_data else None
        if not tf or len(tf_data[tf]) < 24:
            continue
        df = tf_data[tf]
        last_24h = df.iloc[-24:]
        current = current_prices.get(pair)
        if current is None:
            continue
        open_24h = last_24h.iloc[0]["open"]
        context[pair] = {
            "price": current,
            "change_24h_pct": (current - open_24h) / open_24h * 100,
            "high_24h": last_24h["high"].max(),
            "low_24h": last_24h["low"].min(),
            "volume_24h": last_24h["volume"].sum(),
        }
    return context


def print_banner(config):
    """Print startup banner."""
    mode = config.agent.mode.upper()
    mode_color = "red" if mode == "LIVE" else "yellow"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    ps = config.risk.profit_split
    table.add_row("Mode", f"[{mode_color}]{mode}[/{mode_color}]")
    table.add_row("Capital", f"${config.risk.starting_capital:.2f}")
    table.add_row("Max Drawdown", f"{config.risk.max_drawdown_pct:.0%}")
    table.add_row("Profit Split", f"Bot {ps.bot_pct:.0%} / User {ps.user_pct:.0%}")
    table.add_row("Take Profit", f"Bot {ps.bot_take_profit_pct:.0%} / User {ps.user_take_profit_pct:.0%}")
    table.add_row("Trail Stop", f"Bot {ps.bot_trailing_stop_pct:.1%} / User {ps.user_trailing_stop_pct:.1%}")
    if config.risk.max_hold_hours > 0:
        table.add_row("Max Hold", f"{config.risk.max_hold_hours:.0f}h (auto-close)")
    table.add_row("Exchange", config.exchange.name)
    table.add_row("Pairs", ", ".join(config.agent.pairs))
    table.add_row("Cycle", f"{config.agent.cycle_interval_sec}s")
    enabled = [p["name"] for p in config.agent.providers if p.get("enabled", True)]
    table.add_row("AI Models", ", ".join(enabled) if enabled else config.agent.claude_model)
    table.add_row("Consensus", f"{config.agent.consensus_threshold:.0%} majority")

    console.print(Panel(table, title="[bold]FINANCE AGENT[/bold]", border_style="cyan"))


async def run():
    parser = argparse.ArgumentParser(description="Autonomous Crypto Trading Agent")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--dry-run", action="store_true", help="Log decisions without executing")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: config/settings.yaml)")
    args = parser.parse_args()

    config = load_config(config_path=args.config)

    if args.live:
        config.agent.mode = "live"
    if args.dry_run:
        config.agent.dry_run = True

    print_banner(config)

    # Safety check for live mode
    if config.agent.mode == "live":
        if not config.exchange_api_key or not config.exchange_api_secret:
            log.error("Live mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET in .env")
            sys.exit(1)
        console.print("[bold red]WARNING: LIVE TRADING MODE[/bold red]")
        console.print("Real money will be used. Press Ctrl+C within 5 seconds to abort.")
        await asyncio.sleep(5)

    # Initialize components
    db = Database()

    # Shared Hyperliquid native client — one client for Exchange + DataFetcher
    from src.data_sources import HyperliquidNativeClient
    hl_client = HyperliquidNativeClient()

    exchange = Exchange(
        config.exchange, config.agent.mode, config.risk.starting_capital,
        hl_client=hl_client,
        api_key=config.exchange_api_key.get_secret_value(),
        api_secret=config.exchange_api_secret.get_secret_value(),
    )
    # Reconcile paper balance with any open trades persisted in DB
    exchange.sync_paper_balance_from_trades(db.get_open_trades())

    fetcher = DataFetcher(
        exchange, db, config.agent,
        data_source_config=config.data_sources,
        alpha_vantage_key=config.alpha_vantage_api_key.get_secret_value(),
        coingecko_key="",  # Free tier, no key needed
        hl_client=hl_client,
    )
    strategy_engine = StrategyEngine(config.strategy)
    risk_manager = RiskManager(config.risk, db)
    portfolio_mgr = Portfolio(exchange, db, config.risk)
    journal = Journal(db)

    # Phase 5: DEX scanner, execution router, trigger orders
    dex_scanner_inst = None
    trigger_manager = None
    router = None

    if config.dex_scanner.enabled:
        dex_scanner_inst = DexScanner(exchange, config.dex_scanner)

    if config.trigger_orders.enabled:
        trigger_manager = TriggerOrderManager(exchange, db, config.trigger_orders)
        log.info("Trigger order manager initialized (on-chain SL/TP)")

    router = ExecutionRouter(
        exchange, config.execution_router,
        dex_scanner=dex_scanner_inst,
    )
    log.info(
        f"Execution router initialized "
        f"(primary: {config.execution_router.primary_venue}, "
        f"multi-venue: {'enabled' if config.execution_router.enable_multi_venue else 'disabled'}, "
        f"slippage tracking: {'on' if config.execution_router.slippage_tracking else 'off'})"
    )

    executor = Executor(
        exchange, risk_manager, db, journal,
        portfolio=portfolio_mgr,
        router=router,
        trigger_manager=trigger_manager,
    )

    # All data + execution uses Hyperliquid native REST API — no ccxt dependency

    # Initialize brain — prefer MultiBrain (multi-model), fall back to single Brain
    brain = None
    macro_analyst = None
    evolver = None

    log.info("Initializing AI providers...")
    providers = build_providers(config)

    negotiation_engine = None

    if providers:
        brain = MultiBrain(
            providers,
            db,
            config.agent.consensus_threshold,
            escalation_config=config.agent.escalation,
        )
        # Initialize negotiation engine alongside MultiBrain
        negotiation_engine = NegotiationEngine(providers, db)
        log.info(
            f"MultiBrain initialized: {len(providers)} provider(s), "
            f"consensus threshold {config.agent.consensus_threshold:.0%}, "
            f"escalation {'enabled' if config.agent.escalation.enabled else 'disabled'}"
        )
        log.info(
            f"NegotiationEngine initialized: 4 agents (Alpha, Beta, Gamma, Delta) "
            f"with {len(providers)} provider(s)"
        )
    elif config.anthropic_api_key:
        # Fallback: single-model Brain (backward compatible)
        brain = Brain(config.anthropic_api_key.get_secret_value(), config.agent.claude_model, db)
        log.info("Single-model brain initialized (Claude only)")
    else:
        log.warning("No AI providers configured — running without AI vetting")

    # Macro analyst + strategy evolver still use Claude (if available)
    if config.anthropic_api_key:
        macro_analyst = MacroAnalyst(config.anthropic_api_key.get_secret_value(), config.agent.claude_model)
        evolver = StrategyEvolver(config.anthropic_api_key.get_secret_value(), config.agent.claude_model, db)
        active_evolved = evolver.get_active_strategies()
        if active_evolved:
            strategy_engine.load_evolved_strategies(active_evolved)
        log.info(f"Macro analyst + strategy evolver initialized ({len(active_evolved)} evolved strategies)")
    else:
        log.info("No ANTHROPIC_API_KEY — macro analyst + evolver disabled")

    # TradFi intelligence layer
    tradfi_intel = TradFiIntel(
        fred_api_key=config.fred_api_key.get_secret_value(),
        earnings_hours_before=config.event_blocking.stocks_hours_before,
        earnings_hours_after=config.event_blocking.stocks_hours_after,
    )
    log.info(
        f"TradFi intel initialized "
        f"(FRED: {'enabled' if config.fred_api_key else 'disabled'}, "
        f"event blocking: {'enabled' if config.event_blocking.enabled else 'disabled'})"
    )

    # Data integrity checker (always enabled)
    integrity_checker = DataIntegrityChecker()

    # Initialize agent state
    if not db.get_state("trades_since_review"):
        db.set_state("trades_since_review", "0")

    # Register shutdown handler
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    cycle_num = 0
    try:
        while not _shutdown:
            cycle_num += 1
            log.info(f"--- Cycle {cycle_num} ---")

            try:
                _cycle_start = time.monotonic()

                summary = await cycle(
                    fetcher, strategy_engine, brain, risk_manager,
                    executor, portfolio_mgr, journal, db,
                    macro_analyst=macro_analyst,
                    evolver=evolver,
                    integrity_checker=integrity_checker,
                    tradfi_intel=tradfi_intel,
                    dex_scanner=dex_scanner_inst,
                    trigger_manager=trigger_manager,
                    negotiation_engine=negotiation_engine,
                    dry_run=config.agent.dry_run,
                    review_every_n_trades=config.agent.review_every_n_trades,
                )

                _cycle_duration = time.monotonic() - _cycle_start

                # [OPT-2/7] Periodic maintenance every 100 cycles
                if cycle_num % 100 == 0:
                    db.cleanup_cache()
                    db.prune_vote_records()
                log.info(
                    f"Cycle {cycle_num} complete in {_cycle_duration:.1f}s: "
                    f"{summary['signals']} signals, "
                    f"{summary['approved']} approved, "
                    f"{summary['trades']} trades, "
                    f"{summary['closed']} closed"
                )
            except Exception as e:
                log.error(f"Cycle {cycle_num} failed: {e}", exc_info=True)

            if args.once:
                break

            # Sleep in chunks for responsive shutdown
            for _ in range(config.agent.cycle_interval_sec):
                if _shutdown:
                    break
                await asyncio.sleep(1)

    finally:
        log.info("Shutting down...")
        await fetcher.close()
        await exchange.close()
        # [OPT-8] Close provider sessions
        if providers:
            for p in providers:
                if hasattr(p, "close"):
                    try:
                        await p.close()
                    except Exception:
                        pass
        db.close()
        total_tokens = 0
        if brain and hasattr(brain, "total_tokens_used"):
            total_tokens += brain.total_tokens_used
        if macro_analyst and hasattr(macro_analyst, "total_tokens_used"):
            total_tokens += macro_analyst.total_tokens_used
        if total_tokens:
            log.info(f"Total AI tokens used: {total_tokens:,}")
        log.info("Agent stopped.")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
