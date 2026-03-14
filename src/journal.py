"""Trade journal — logs every decision with reasoning for learning."""

from __future__ import annotations

import csv
from pathlib import Path

from src.database import Database
from src.models import Signal, Trade
from src.utils import log, now_iso


class Journal:
    def __init__(self, db: Database):
        self.db = db

    def log_trade(self, trade: Trade):
        """Log a trade (new or updated)."""
        if trade.exit_price is None:
            # Opening or updating an open trade
            if trade.id is None:
                self.db.save_trade(trade)
            else:
                self.db.update_trade(trade)
            log.info(
                f"TRADE OPENED: {trade.direction.upper()} {trade.pair} "
                f"@ {trade.entry_price:.2f} | qty: {trade.quantity:.6f} | "
                f"SL: {trade.stop_loss:.2f} | TP: {trade.take_profit or 'none'}"
            )
        else:
            self.db.update_trade(trade)
            status = trade.status.upper()
            pnl = f"${trade.pnl:+.2f}" if trade.pnl is not None else "n/a"
            log.info(
                f"TRADE {status}: {trade.pair} | exit: {trade.exit_price:.2f} | P&L: {pnl}"
            )

    def log_signal(self, signal: Signal, acted_on: bool, reason: str = ""):
        """Log a signal, whether it was acted on or not."""
        self.db.save_signal(signal, acted_on, reason)
        action = "ACTED ON" if acted_on else f"SKIPPED ({reason})"
        log.debug(
            f"SIGNAL {action}: {signal.direction.upper()} {signal.pair} "
            f"[{signal.strategy_name}] conf={signal.confidence:.2f}"
        )

    def log_decision(self, description: str, reasoning: str):
        """Log a non-trade decision (e.g., skipping a cycle, pausing)."""
        log.info(f"DECISION: {description} | {reasoning}")

    def get_performance_summary(self, days: int = 30) -> dict:
        """Compute performance metrics over the last N days."""
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        trades = self.db.get_trades_since(since)

        closed = [t for t in trades if t.status in ("closed", "stopped_out") and t.pnl is not None]
        if not closed:
            return {"total_trades": 0, "message": "No closed trades in period"}

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in closed)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed),
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float("inf"),
            "best_trade": max(t.pnl for t in closed),
            "worst_trade": min(t.pnl for t in closed),
            "avg_pnl_pct": sum(t.pnl_pct for t in closed if t.pnl_pct) / len(closed),
        }

    def export_csv(self, path: str, days: int = 30):
        """Export trade journal to CSV."""
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        trades = self.db.get_trades_since(since)

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "pair", "direction", "entry_price", "exit_price",
                "quantity", "stop_loss", "take_profit", "status", "pnl",
                "pnl_pct", "fees", "entry_time", "exit_time", "ai_reasoning",
                "profit_split_bot", "profit_split_user",
                "user_take_profit", "bot_take_profit",
            ])
            for t in trades:
                writer.writerow([
                    t.id, t.pair, t.direction, t.entry_price, t.exit_price,
                    t.quantity, t.stop_loss, t.take_profit, t.status, t.pnl,
                    t.pnl_pct, t.fees, t.entry_time, t.exit_time, t.ai_reasoning,
                    t.profit_split_bot, t.profit_split_user,
                    t.user_take_profit, t.bot_take_profit,
                ])
        log.info(f"Journal exported to {filepath}")
