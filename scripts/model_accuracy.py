"""Print per-model accuracy stats from the model_accuracy table.

Usage:
    python scripts/model_accuracy.py
    python scripts/model_accuracy.py --limit 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.database import Database
from src.multi_brain import ModelAccuracyTracker


def main():
    parser = argparse.ArgumentParser(description="Per-model accuracy report")
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Max outcome rows to load (default: 10000)",
    )
    args = parser.parse_args()

    db = Database()
    tracker = ModelAccuracyTracker(db)
    console = Console()

    rows = db.get_model_outcomes(limit=args.limit)
    if not rows:
        console.print("[yellow]No model outcome data yet. Outcomes are recorded as trades close.[/yellow]")
        db.close()
        return

    stats = tracker.get_model_stats()
    weights = tracker.get_model_weights()
    best = tracker.get_best_model()

    # Main accuracy table
    table = Table(title=f"Model Accuracy Report ({len(rows)} outcome rows)", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Votes", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Solo P&L", justify="right")
    table.add_column("Agree Rate", justify="right")
    table.add_column("Contrarian Acc", justify="right")
    table.add_column("Weight", justify="right")

    for model_name, s in sorted(stats.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        accuracy = s["accuracy"]
        acc_style = "green" if accuracy >= 0.55 else "red" if accuracy < 0.45 else "yellow"

        solo_pnl = s["solo_pnl_estimate"]
        pnl_style = "green" if solo_pnl > 0 else "red"

        weight = weights.get(model_name, 1.0)
        weight_style = "green" if weight > 1.0 else "red" if weight < 1.0 else "white"

        name_display = f"[bold cyan]{model_name}[/bold cyan]" if model_name == best else model_name

        table.add_row(
            name_display,
            str(s["total_votes"]),
            str(s["correct_votes"]),
            f"[{acc_style}]{accuracy:.1%}[/{acc_style}]",
            f"[{pnl_style}]{solo_pnl:+d}[/{pnl_style}]",
            f"{s['agree_rate']:.1%}",
            f"{s['contrarian_accuracy']:.1%}",
            f"[{weight_style}]{weight:.2f}x[/{weight_style}]",
        )

    console.print(table)

    if best:
        console.print(f"\n[bold green]Best model:[/bold green] {best} "
                      f"(accuracy={stats[best]['accuracy']:.1%}, "
                      f"weight={weights.get(best, 1.0):.2f}x)")

    # Per-pair breakdown
    pair_model_stats: dict[str, dict[str, dict]] = {}
    for row in rows:
        pair = row["pair"]
        model = row["model_name"]
        if pair not in pair_model_stats:
            pair_model_stats[pair] = {}
        if model not in pair_model_stats[pair]:
            pair_model_stats[pair][model] = {"total": 0, "correct": 0}
        pair_model_stats[pair][model]["total"] += 1
        if row["was_correct"]:
            pair_model_stats[pair][model]["correct"] += 1

    if len(pair_model_stats) > 1:
        pair_table = Table(title="Accuracy by Pair", show_lines=True)
        pair_table.add_column("Pair", style="bold")
        pair_table.add_column("Model")
        pair_table.add_column("Votes", justify="right")
        pair_table.add_column("Accuracy", justify="right")

        for pair in sorted(pair_model_stats):
            first = True
            for model_name in sorted(pair_model_stats[pair]):
                ms = pair_model_stats[pair][model_name]
                acc = ms["correct"] / ms["total"] if ms["total"] else 0.0
                acc_style = "green" if acc >= 0.55 else "red" if acc < 0.45 else "yellow"
                pair_table.add_row(
                    pair if first else "",
                    model_name,
                    str(ms["total"]),
                    f"[{acc_style}]{acc:.1%}[/{acc_style}]",
                )
                first = False

        console.print(pair_table)

    db.close()


if __name__ == "__main__":
    main()
