"""Portfolio state — tracks balance, positions, P&L, drawdown.

Maintains two separate profit pools:
  - bot_balance: Aggressive reinvestment pool (~70% of profits). Used for position sizing.
  - user_balance: Conservative withdrawal pool (~30% of profits). Never risked.
"""

from __future__ import annotations

from src.config import ProfitSplitConfig, RiskConfig
from src.database import Database
from src.exchange import Exchange
from src.models import PortfolioSnapshot
from src.utils import log, now_iso


class Portfolio:
    def __init__(self, exchange: Exchange, db: Database, config: RiskConfig):
        self.exchange = exchange
        self.db = db
        self.config = config
        self._high_water_mark: float = config.starting_capital

        # Load persisted high water mark
        stored = db.get_state("high_water_mark")
        if stored:
            self._high_water_mark = float(stored)

        # Dual profit pools — bot gets the starting capital, user starts at 0
        self.bot_balance: float = config.starting_capital
        self.user_balance: float = 0.0

        # Load persisted pool balances
        stored_bot = db.get_state("bot_balance")
        stored_user = db.get_state("user_balance")
        if stored_bot is not None:
            self.bot_balance = float(stored_bot)
        if stored_user is not None:
            self.user_balance = float(stored_user)

    async def get_snapshot(self, current_prices: dict[str, float] | None = None) -> PortfolioSnapshot:
        """Build a current portfolio snapshot."""
        balance = await self.exchange.fetch_balance()
        qc = self.exchange.quote_currency
        cash = balance.get("free", {}).get(qc, 0.0)

        # Calculate positions value
        positions_value = 0.0
        if self.exchange.mode == "paper":
            if current_prices:
                positions_value = self.exchange.get_paper_portfolio_value(current_prices) - cash
        else:
            from src.exchange import parse_pair
            total_bal = balance.get("total", {})
            for asset, amount in total_bal.items():
                if asset == qc or amount <= 0:
                    continue
                # Find a matching price entry for this asset
                for price_pair, price_val in (current_prices or {}).items():
                    base, _ = parse_pair(price_pair)
                    if base == asset:
                        positions_value += amount * price_val
                        break

        total_value = cash + positions_value

        # Update high water mark
        if total_value > self._high_water_mark:
            self._high_water_mark = total_value
            self.db.set_state("high_water_mark", str(self._high_water_mark))

        # Calculate drawdown from high water mark
        drawdown_pct = 0.0
        if self._high_water_mark > 0:
            drawdown_pct = (self._high_water_mark - total_value) / self._high_water_mark

        # P&L
        total_pnl = total_value - self.config.starting_capital
        total_pnl_pct = total_pnl / self.config.starting_capital if self.config.starting_capital > 0 else 0

        # Daily P&L (compare with earliest snapshot today)
        daily_pnl = self._calc_daily_pnl(total_value)

        # Count open positions
        open_trades = self.db.get_open_trades()
        open_count = len(open_trades)

        return PortfolioSnapshot(
            timestamp=now_iso(),
            total_value=total_value,
            cash=cash,
            positions_value=positions_value,
            open_positions=open_count,
            drawdown_pct=drawdown_pct,
            high_water_mark=self._high_water_mark,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            bot_balance=self.bot_balance,
            user_balance=self.user_balance,
        )

    def _calc_daily_pnl(self, current_value: float) -> float:
        """Calculate P&L since start of today."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
        curve = self.db.get_equity_curve(days=1, limit=1)
        if curve:
            return current_value - curve[0].total_value
        return current_value - self.config.starting_capital

    # --- Dual profit pool management ---

    def apply_profit_split(self, pnl: float) -> tuple[float, float]:
        """Split a trade's P&L into bot and user pools.

        For profitable trades: split according to profit_split config.
        For losing trades: loss comes entirely from bot_balance (user pool is protected).

        Returns (bot_share, user_share).
        """
        split = self.config.profit_split

        if pnl > 0:
            bot_share = pnl * split.bot_pct
            user_share = pnl * split.user_pct
            self.bot_balance += bot_share
            self.user_balance += user_share
            log.info(
                f"PROFIT SPLIT: ${pnl:+.2f} -> "
                f"Bot +${bot_share:.2f} (${self.bot_balance:.2f}) | "
                f"User +${user_share:.2f} (${self.user_balance:.2f})"
            )
        else:
            # Losses come only from bot pool — user pool is protected
            bot_share = pnl  # negative
            user_share = 0.0
            self.bot_balance += bot_share
            log.info(
                f"LOSS ABSORBED BY BOT: ${pnl:+.2f} -> "
                f"Bot ${bot_share:.2f} (${self.bot_balance:.2f}) | "
                f"User unchanged (${self.user_balance:.2f})"
            )

        self._persist_balances()
        return bot_share, user_share

    def _persist_balances(self):
        """Save dual pool balances to agent_state for crash recovery."""
        self.db.set_state("bot_balance", f"{self.bot_balance:.6f}")
        self.db.set_state("user_balance", f"{self.user_balance:.6f}")
