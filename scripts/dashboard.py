"""Terminal dashboard — shows portfolio state, recent trades, equity curve."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from src.database import Database
from src.config import load_config


def build_dashboard(db: Database, config) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    # Header
    status = db.get_state("agent_status") or "RUNNING"
    mode = config.agent.mode.upper()
    layout["header"].update(
        Panel(f"[bold]FINANCE AGENT[/bold] | Mode: {mode} | Status: {status}", style="cyan")
    )

    # Portfolio
    curve = db.get_equity_curve(days=1)
    if curve:
        snap = curve[-1]
        port_table = Table(show_header=False, box=None)
        port_table.add_column(style="bold")
        port_table.add_column(justify="right")
        port_table.add_row("Total Value", f"${snap.total_value:.2f}")
        port_table.add_row("Cash", f"${snap.cash:.2f}")
        port_table.add_row("Positions", f"${snap.positions_value:.2f}")
        port_table.add_row("Drawdown", f"{snap.drawdown_pct:.1%}")
        port_table.add_row("Daily P&L", f"${snap.daily_pnl:+.2f}")
        port_table.add_row("Total P&L", f"${snap.total_pnl:+.2f} ({snap.total_pnl_pct:+.1%})")
        layout["left"].update(Panel(port_table, title="Portfolio"))
    else:
        layout["left"].update(Panel("No data yet", title="Portfolio"))

    # Recent trades
    trades = db.get_recent_trades(10)
    trade_table = Table(box=None)
    trade_table.add_column("Pair", style="bold")
    trade_table.add_column("Dir")
    trade_table.add_column("Entry", justify="right")
    trade_table.add_column("Exit", justify="right")
    trade_table.add_column("P&L", justify="right")
    trade_table.add_column("Status")

    for t in trades:
        pnl = f"${t.pnl:+.2f}" if t.pnl is not None else "-"
        pnl_style = "green" if t.pnl and t.pnl > 0 else "red" if t.pnl and t.pnl < 0 else ""
        trade_table.add_row(
            t.pair, t.direction.upper(),
            f"{t.entry_price:.2f}", f"{t.exit_price:.2f}" if t.exit_price else "-",
            f"[{pnl_style}]{pnl}[/{pnl_style}]",
            t.status,
        )

    layout["right"].update(Panel(trade_table, title="Recent Trades"))

    # Footer
    layout["footer"].update(
        Panel("Press Ctrl+C to exit | Refreshes every 5s", style="dim")
    )

    return layout


def main():
    config = load_config()
    db = Database()
    console = Console()

    try:
        with Live(build_dashboard(db, config), console=console, refresh_per_second=0.2) as live:
            while True:
                time.sleep(5)
                live.update(build_dashboard(db, config))
    except KeyboardInterrupt:
        pass
    finally:
        db.close()


if __name__ == "__main__":
    main()
