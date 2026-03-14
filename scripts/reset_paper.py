"""Reset paper trading state — use after drawdown circuit breaker triggers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.database import Database


def main():
    config = load_config()
    db = Database()

    # Reset agent status
    db.set_state("agent_status", "RUNNING")
    db.set_state("high_water_mark", str(config.risk.starting_capital))
    db.set_state("trades_since_review", "0")

    # Close any open trades in DB (mark as cancelled)
    open_trades = db.get_open_trades()
    for trade in open_trades:
        trade.status = "cancelled"
        trade.review_notes = "Cancelled by paper reset"
        db.update_trade(trade)

    print(f"Paper trading reset complete.")
    print(f"  Starting capital: ${config.risk.starting_capital}")
    print(f"  Open trades cancelled: {len(open_trades)}")
    print(f"  Agent status: RUNNING")

    db.close()


if __name__ == "__main__":
    main()
