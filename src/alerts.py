"""Alert system — notifications for important events via terminal + optional webhooks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import aiohttp

from src.models import PortfolioSnapshot, Trade
from src.utils import log


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertManager:
    """Manages alerts and notifications for the trading agent."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url
        self._alert_log: list[dict] = []
        self._log_path = Path(__file__).resolve().parent.parent / "data" / "alerts.json"
        self._load_log()

    def _load_log(self):
        if self._log_path.exists():
            try:
                self._alert_log = json.loads(self._log_path.read_text())[-100:]  # Keep last 100
            except Exception:
                self._alert_log = []

    def _save_log(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(json.dumps(self._alert_log[-100:], indent=2))

    async def alert(self, level: AlertLevel, title: str, message: str, data: dict | None = None):
        """Send an alert."""
        alert_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "title": title,
            "message": message,
            "data": data or {},
        }

        # Log to console with appropriate level
        if level == AlertLevel.CRITICAL:
            log.critical(f"ALERT: {title} — {message}")
        elif level == AlertLevel.WARNING:
            log.warning(f"ALERT: {title} — {message}")
        else:
            log.info(f"ALERT: {title} — {message}")

        # Save to file
        self._alert_log.append(alert_entry)
        self._save_log()

        # Send webhook if configured
        if self.webhook_url:
            await self._send_webhook(alert_entry)

    async def _send_webhook(self, alert: dict):
        """Send alert to a webhook (Discord/Slack compatible)."""
        emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(alert["level"], "")

        payload = {
            "content": f"{emoji} **{alert['title']}**\n{alert['message']}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status >= 400:
                        log.warning(f"Webhook failed: {resp.status}")
        except Exception as e:
            log.warning(f"Webhook error: {e}")

    # --- Convenience methods for common alerts ---

    async def trade_opened(self, trade: Trade):
        await self.alert(
            AlertLevel.INFO,
            f"Trade Opened: {trade.direction.upper()} {trade.pair}",
            f"Entry: ${trade.entry_price:.2f} | Qty: {trade.quantity:.6f} | "
            f"SL: ${trade.stop_loss:.2f} | TP: ${trade.take_profit or 'none'}",
            {"trade_id": trade.id, "pair": trade.pair},
        )

    async def trade_closed(self, trade: Trade):
        pnl = trade.pnl or 0
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        await self.alert(
            level,
            f"Trade Closed: {trade.pair} ({trade.status})",
            f"P&L: ${pnl:+.2f} ({trade.pnl_pct:+.1%})" if trade.pnl_pct else f"P&L: ${pnl:+.2f}",
            {"trade_id": trade.id, "pair": trade.pair, "pnl": pnl},
        )

    async def drawdown_warning(self, portfolio: PortfolioSnapshot):
        await self.alert(
            AlertLevel.WARNING,
            "Drawdown Warning",
            f"Current drawdown: {portfolio.drawdown_pct:.1%} | "
            f"Portfolio: ${portfolio.total_value:.2f}",
        )

    async def circuit_breaker(self, portfolio: PortfolioSnapshot):
        await self.alert(
            AlertLevel.CRITICAL,
            "CIRCUIT BREAKER TRIGGERED",
            f"Max drawdown reached: {portfolio.drawdown_pct:.1%} | "
            f"Portfolio: ${portfolio.total_value:.2f} | All positions closed.",
        )

    async def agent_started(self, mode: str, capital: float):
        await self.alert(
            AlertLevel.INFO,
            "Agent Started",
            f"Mode: {mode} | Capital: ${capital:.2f}",
        )

    async def ai_review_complete(self, summary: str):
        await self.alert(
            AlertLevel.INFO,
            "AI Review Complete",
            summary[:200],
        )

    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        return self._alert_log[-n:]
