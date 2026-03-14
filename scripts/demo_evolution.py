"""Strategy Evolution Demo — showcases the self-improving strategy evolver.

Feeds mock performance data, triggers Claude to generate a new strategy,
validates it in the sandbox, activates it, and runs it against live OHLCV data.

Usage:
    python scripts/demo_evolution.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from src.config import load_config
from src.data_fetcher import DataFetcher
from src.database import Database
from src.exchange import Exchange
from src.strategy import StrategyEngine
from src.strategy_evolver import (
    EVOLVED_DIR,
    FORBIDDEN_TOKENS,
    StrategyEvolver,
    StrategyPerformance,
)

console = Console()


def print_step(n: int, title: str):
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]  STEP {n}: {title}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


async def main():
    config = load_config()
    db = Database()

    # Check for Anthropic API key
    if not config.anthropic_api_key:
        console.print("[red]ERROR: ANTHROPIC_API_KEY required for strategy evolution[/red]")
        return

    api_key = config.anthropic_api_key.get_secret_value()
    evolver = StrategyEvolver(api_key, config.agent.claude_model, db)
    strategy_engine = StrategyEngine(config.strategy)

    # ── Step 1: Show mock performance data ──
    print_step(1, "MOCK PERFORMANCE DATA")
    console.print("[dim]Simulating 20 trades across existing strategies...[/dim]\n")

    mock_performance = {
        "momentum": StrategyPerformance(
            strategy_name="momentum",
            total_trades=8,
            wins=5,
            losses=3,
            total_pnl=142.50,
            avg_pnl=17.81,
            win_rate=0.625,
            max_win=68.30,
            max_loss=-31.20,
            avg_confidence=0.72,
            recent_streak=2,
        ),
        "trend_following": StrategyPerformance(
            strategy_name="trend_following",
            total_trades=6,
            wins=3,
            losses=3,
            total_pnl=-22.40,
            avg_pnl=-3.73,
            win_rate=0.50,
            max_win=45.10,
            max_loss=-42.80,
            avg_confidence=0.65,
            recent_streak=-1,
        ),
        "mean_reversion": StrategyPerformance(
            strategy_name="mean_reversion",
            total_trades=4,
            wins=3,
            losses=1,
            total_pnl=89.20,
            avg_pnl=22.30,
            win_rate=0.75,
            max_win=52.40,
            max_loss=-15.60,
            avg_confidence=0.68,
            recent_streak=3,
        ),
        "breakout": StrategyPerformance(
            strategy_name="breakout",
            total_trades=2,
            wins=0,
            losses=2,
            total_pnl=-78.10,
            avg_pnl=-39.05,
            win_rate=0.0,
            max_win=0.0,
            max_loss=-45.30,
            avg_confidence=0.58,
            recent_streak=-2,
        ),
    }

    table = Table(title="Strategy Performance (Mock Data)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Avg P&L", justify="right")
    table.add_column("Streak", justify="right")

    for name, p in mock_performance.items():
        pnl_style = "green" if p.total_pnl > 0 else "red"
        streak_style = "green" if p.recent_streak > 0 else "red"
        table.add_row(
            name,
            str(p.total_trades),
            f"{p.win_rate:.0%}",
            f"[{pnl_style}]${p.total_pnl:+.2f}[/{pnl_style}]",
            f"[{pnl_style}]${p.avg_pnl:+.2f}[/{pnl_style}]",
            f"[{streak_style}]{p.recent_streak:+d}[/{streak_style}]",
        )

    console.print(table)

    # ── Step 2: Generate a new strategy ──
    print_step(2, "CLAUDE GENERATES NEW STRATEGY")
    console.print("[dim]Asking Claude to analyze performance gaps and generate a new strategy...[/dim]\n")

    new_strat = evolver.generate_strategy(
        mock_performance,
        market_regime="bull",
        hint="RSI divergence — look for price making new highs/lows while RSI diverges",
    )

    if not new_strat:
        console.print("[red]Strategy generation failed. Check API key and try again.[/red]")
        return

    console.print(f"[green]Generated:[/green] {new_strat.name}")
    console.print(f"[dim]Generation: {new_strat.generation} | Hash: {new_strat.code_hash}[/dim]")
    console.print(f"[dim]Rationale: {new_strat.rationale}[/dim]\n")

    # Show the generated code
    strat_path = Path(new_strat.file_path)
    if not strat_path.exists():
        console.print(f"[red]Strategy file not found: {strat_path}[/red]")
        return
    code = strat_path.read_text()
    console.print(Panel(
        Syntax(code, "python", theme="monokai", line_numbers=True),
        title=f"[bold]Generated Strategy: {new_strat.name}[/bold]",
        border_style="green",
    ))

    # ── Step 3: Sandbox validation ──
    print_step(3, "SANDBOX VALIDATION")

    console.print("[bold]Security checks:[/bold]")
    # Run the real sandbox validation
    valid, reason = evolver.validate_strategy(new_strat)
    if valid:
        console.print(f"  [green]PASS[/green] — No forbidden tokens ({len(FORBIDDEN_TOKENS)} checked)")
        console.print(f"  [green]PASS[/green] — Import allowlist validated")
        console.print(f"  [green]PASS[/green] — Sandbox module loaded successfully")
        console.print(f"  [green]PASS[/green] — Synthetic data test returned Signal | None")
        console.print(f"  [green]PASS[/green] — BaseStrategy subclass confirmed\n")
    else:
        console.print(f"  [red]FAIL[/red] — Validation failed: {reason}\n")

    # ── Step 4: Activate the strategy ──
    print_step(4, "ACTIVATE STRATEGY")

    activated = evolver.activate_strategy(new_strat)
    if activated:
        console.print(f"[green]Strategy '{new_strat.name}' is now ACTIVE[/green]")
        console.print("[dim]Loading into StrategyEngine...[/dim]")

        active = evolver.get_active_strategies()
        strategy_engine.load_evolved_strategies(active)
        console.print(f"[green]StrategyEngine now has {4 + len(active)} strategies[/green]")
    else:
        console.print("[yellow]Activation skipped (max active reached or validation failed)[/yellow]")

    # ── Step 5: Run against live data ──
    print_step(5, "LIVE DATA TEST")
    console.print("[dim]Fetching real OHLCV data from Hyperliquid...[/dim]\n")

    try:
        from src.data_sources import HyperliquidNativeClient
        hl_client = HyperliquidNativeClient()
        exchange = Exchange(
            config.exchange, "paper", config.risk.starting_capital,
            hl_client=hl_client,
            api_key=config.exchange_api_key.get_secret_value(),
            api_secret=config.exchange_api_secret.get_secret_value(),
        )
        fetcher = DataFetcher(
            exchange, db, config.agent,
            data_source_config=config.data_sources,
            alpha_vantage_key=config.alpha_vantage_api_key.get_secret_value(),
            coingecko_key="",
            hl_client=hl_client,
        )

        # Fetch data for BTC
        pair = "BTC/USDC:USDC"
        console.print(f"[dim]Fetching {pair} OHLCV data...[/dim]")
        market_data = await fetcher.fetch_pair(pair)

        if market_data:
            signals = strategy_engine.generate_signals({pair: market_data})
            if signals:
                sig_table = Table(title="Signals Generated (Live Data)")
                sig_table.add_column("Pair")
                sig_table.add_column("Direction")
                sig_table.add_column("Strategy")
                sig_table.add_column("Confidence")

                for sig in signals:
                    dir_style = "green" if sig.direction == "long" else "red"
                    sig_table.add_row(
                        sig.pair,
                        f"[{dir_style}]{sig.direction.upper()}[/{dir_style}]",
                        sig.strategy_name,
                        f"{sig.confidence:.2f}",
                    )
                console.print(sig_table)
            else:
                console.print("[yellow]No signals generated (market may be sideways)[/yellow]")
        else:
            console.print("[yellow]No market data fetched — agent may need exchange keys[/yellow]")

        await fetcher.close()
        await exchange.close()

    except Exception as e:
        console.print(f"[yellow]Live data test skipped: {e}[/yellow]")

    # ── Step 6: Show leaderboard ──
    print_step(6, "STRATEGY LEADERBOARD")

    leaderboard = evolver.get_leaderboard()
    if leaderboard:
        lb_table = Table(title="Evolved Strategy Leaderboard")
        lb_table.add_column("Name", style="cyan")
        lb_table.add_column("Gen", justify="right")
        lb_table.add_column("Status")
        lb_table.add_column("Trades", justify="right")
        lb_table.add_column("Win Rate", justify="right")
        lb_table.add_column("P&L", justify="right")

        for m in leaderboard:
            status_style = "green" if m.status == "active" else "yellow" if m.status == "candidate" else "red"
            lb_table.add_row(
                m.name,
                str(m.generation),
                f"[{status_style}]{m.status.upper()}[/{status_style}]",
                str(m.total_trades),
                f"{m.win_rate:.0%}" if m.total_trades > 0 else "-",
                f"${m.total_pnl:+.2f}" if m.total_trades > 0 else "-",
            )
        console.print(lb_table)
    else:
        console.print("[dim]No evolved strategies in leaderboard yet[/dim]")

    console.print("\n[bold green]Demo complete![/bold green]")
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
