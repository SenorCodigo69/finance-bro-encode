"""Export trade journal and performance reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.database import Database
from src.journal import Journal


def main():
    parser = argparse.ArgumentParser(description="Export trade journal")
    parser.add_argument("--days", type=int, default=30, help="Number of days to export")
    parser.add_argument("--csv", type=str, help="Export to CSV file")
    parser.add_argument("--summary", action="store_true", help="Show performance summary")
    args = parser.parse_args()

    db = Database()
    journal = Journal(db)
    console = Console()

    if args.summary:
        perf = journal.get_performance_summary(args.days)
        if perf.get("total_trades", 0) == 0:
            console.print("[yellow]No closed trades in the last {} days[/yellow]".format(args.days))
        else:
            table = Table(title=f"Performance Summary ({args.days}d)")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")
            table.add_row("Total Trades", str(perf["total_trades"]))
            table.add_row("Wins", str(perf["wins"]))
            table.add_row("Losses", str(perf["losses"]))

            wr = perf["win_rate"]
            wr_style = "green" if wr > 0.5 else "red"
            table.add_row("Win Rate", f"[{wr_style}]{wr:.1%}[/{wr_style}]")

            pnl = perf["total_pnl"]
            pnl_style = "green" if pnl > 0 else "red"
            table.add_row("Total P&L", f"[{pnl_style}]${pnl:+.2f}[/{pnl_style}]")
            table.add_row("Avg Win", f"[green]${perf['avg_win']:+.2f}[/green]")
            table.add_row("Avg Loss", f"[red]${perf['avg_loss']:+.2f}[/red]")
            table.add_row("Profit Factor", f"{perf['profit_factor']:.2f}")
            table.add_row("Best Trade", f"${perf['best_trade']:+.2f}")
            table.add_row("Worst Trade", f"${perf['worst_trade']:+.2f}")
            console.print(table)
    elif args.csv:
        journal.export_csv(args.csv, args.days)
        console.print(f"[green]Exported to {args.csv}[/green]")
    else:
        # Default: print recent trades table
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        trades = db.get_trades_since(since)

        if not trades:
            console.print(f"[yellow]No trades in the last {args.days} days[/yellow]")
        else:
            table = Table(title=f"Trade Journal ({args.days}d)")
            table.add_column("ID")
            table.add_column("Pair")
            table.add_column("Dir")
            table.add_column("Entry", justify="right")
            table.add_column("Exit", justify="right")
            table.add_column("P&L", justify="right")
            table.add_column("Status")
            table.add_column("Strategy")
            table.add_column("Time")

            for t in trades:
                pnl = f"${t.pnl:+.2f}" if t.pnl is not None else "-"
                style = "green" if t.pnl and t.pnl > 0 else "red" if t.pnl and t.pnl < 0 else ""
                table.add_row(
                    str(t.id), t.pair, t.direction.upper(),
                    f"{t.entry_price:.2f}",
                    f"{t.exit_price:.2f}" if t.exit_price else "-",
                    f"[{style}]{pnl}[/{style}]",
                    t.status,
                    t.signal_data.get("strategy", ""),
                    t.entry_time[:16] if t.entry_time else "",
                )
            console.print(table)

    db.close()


if __name__ == "__main__":
    main()
