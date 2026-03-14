"""Order execution — places trades via exchange, tracks fills.

Supports dual profit-taking: each trade gets user and bot TP/trailing levels
set at entry. On close, profit is split between bot and user pools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.database import Database
from src.exchange import Exchange
from src.execution_router import ExecutionRouter
from src.journal import Journal
from src.models import PortfolioSnapshot, Signal, Trade
from src.risk_manager import RiskManager
from src.trigger_orders import TriggerOrderManager
from src.utils import log, now_iso

if TYPE_CHECKING:
    from src.portfolio import Portfolio


class Executor:
    def __init__(
        self,
        exchange: Exchange,
        risk_manager: RiskManager,
        db: Database,
        journal: Journal,
        portfolio: "Portfolio | None" = None,
        router: ExecutionRouter | None = None,
        trigger_manager: TriggerOrderManager | None = None,
    ):
        self.exchange = exchange
        self.risk = risk_manager
        self.db = db
        self.journal = journal
        self.portfolio = portfolio  # Needed for profit splitting on close
        self.router = router
        self.trigger_manager = trigger_manager

    async def execute_signal(
        self,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        ai_reasoning: str,
        atr_value: float,
        dry_run: bool = False,
    ) -> Trade | None:
        """Full execution flow: risk check -> size -> place order -> log."""

        # Risk check
        can_trade, block_reason = self.risk.check_can_trade(portfolio)
        if not can_trade:
            self.journal.log_signal(signal, acted_on=False, reason=block_reason)
            return None

        # Get current price for entry
        ticker = await self.exchange.fetch_ticker(signal.pair)
        entry_price = ticker["last"]
        if not entry_price:
            log.warning(f"No price available for {signal.pair}")
            return None

        # Calculate stop loss and take profit
        stop_loss = self.risk.calculate_stop_loss(signal.direction, entry_price, atr_value)
        take_profit = self.risk.calculate_take_profit(entry_price, stop_loss, signal.direction)

        # Calculate dual take-profit levels for user/bot pools
        user_tp, bot_tp, user_trail, bot_trail = self.risk.calculate_dual_take_profits(
            entry_price, signal.direction
        )

        # Size position (uses bot_balance only)
        quantity = self.risk.size_position(entry_price, stop_loss, portfolio, signal.confidence)
        if quantity <= 0:
            self.journal.log_signal(signal, acted_on=False, reason="Position size too small")
            return None

        # Minimum order check ($5 minimum)
        if quantity * entry_price < 5.0:
            self.journal.log_signal(signal, acted_on=False, reason=f"Below minimum order (${quantity * entry_price:.2f})")
            return None

        trade = Trade(
            id=None,
            pair=signal.pair,
            direction=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="open",
            entry_time=now_iso(),
            signal_data={
                "strategy": signal.strategy_name,
                "confidence": signal.confidence,
                "indicators": signal.indicators,
                "reasoning": signal.reasoning,
            },
            ai_reasoning=ai_reasoning,
            # Dual profit-taking levels
            user_take_profit=user_tp,
            bot_take_profit=bot_tp,
            user_trailing_stop=user_trail,
            bot_trailing_stop=bot_trail,
        )

        log.info(
            f"Dual TP set: User TP={user_tp:.2f} (trail {user_trail:.2f}) | "
            f"Bot TP={bot_tp:.2f} (trail {bot_trail:.2f})"
        )

        if dry_run:
            log.info(f"[DRY RUN] Would open: {signal.direction.upper()} {signal.pair} "
                     f"qty={quantity:.6f} @ {entry_price:.2f}")
            self.journal.log_signal(signal, acted_on=False, reason="dry_run")
            return None

        # Place the order (via router if available, else direct)
        try:
            side = "buy" if signal.direction == "long" else "sell"

            if self.router:
                plan = await self.router.plan_execution(signal.pair, side, quantity)
                order = await self.router.execute(plan)
                trade.execution_venue = order.get("execution_venue", "hyperliquid")
                trade.slippage_actual_bps = order.get("slippage_actual_bps")
            else:
                order = await self.exchange.create_market_order(signal.pair, side, quantity)

            trade.entry_price = order.get("price", entry_price)
            trade.fees = order.get("fee", 0)

            # Explicitly save trade to DB to ensure trade.id is set
            if trade.id is None:
                trade.id = self.db.save_trade(trade)
            self.journal.log_trade(trade)
            self.journal.log_signal(signal, acted_on=True)

            # Place on-chain trigger orders (SL/TP)
            if self.trigger_manager and trade.id is not None:
                try:
                    await self.trigger_manager.place_triggers_for_trade(trade)
                    self.db.update_trade(trade)
                except Exception as e:
                    log.warning(f"Trigger order placement failed (non-critical): {e}")

            return trade

        except Exception as e:
            log.error(f"Order execution failed for {signal.pair}: {e}")
            self.journal.log_signal(signal, acted_on=False, reason=f"Execution error: {e}")
            return None

    async def close_trade(self, trade: Trade, reason: str, current_price: float | None = None) -> Trade:
        """Close an open trade and split profit between bot/user pools."""
        if current_price is None:
            ticker = await self.exchange.fetch_ticker(trade.pair)
            current_price = ticker["last"]

        # Place closing order
        side = "sell" if trade.direction == "long" else "buy"
        try:
            order = await self.exchange.create_market_order(trade.pair, side, trade.quantity)
            exit_price = order.get("price", current_price)
            fee = order.get("fee", 0)
        except Exception as e:
            log.error(f"Failed to close trade {trade.id} ({trade.pair}): {e}")
            return trade

        # Calculate P&L
        if trade.direction == "long":
            pnl = (exit_price - trade.entry_price) * trade.quantity - trade.fees - fee
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity - trade.fees - fee

        pnl_pct = pnl / (trade.entry_price * trade.quantity) if trade.entry_price * trade.quantity > 0 else 0

        trade.exit_price = exit_price
        trade.exit_time = now_iso()
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.fees += fee
        trade.status = reason  # "closed", "stopped_out", "take_profit"

        # --- Dual profit split ---
        if self.portfolio is not None and pnl is not None:
            bot_share, user_share = self.portfolio.apply_profit_split(pnl)
            trade.profit_split_bot = bot_share
            trade.profit_split_user = user_share
            log.info(
                f"Trade {trade.id} P&L ${pnl:+.2f} split: "
                f"Bot=${bot_share:+.2f} User=${user_share:+.2f}"
            )
        else:
            trade.profit_split_bot = pnl
            trade.profit_split_user = 0.0

        self.journal.log_trade(trade)

        # Cancel any remaining trigger orders for this trade
        if self.trigger_manager and trade.id is not None:
            try:
                await self.trigger_manager.cancel_triggers_for_trade(trade.id)
            except Exception as e:
                log.warning(f"Trigger cancellation failed (non-critical): {e}")

        # Update trade count for review trigger
        count = self.db.get_state("trades_since_review") or "0"
        self.db.set_state("trades_since_review", str(int(count) + 1))

        return trade

    async def check_and_close_stops(self, current_prices: dict[str, float]) -> list[Trade]:
        """Check all open trades against stop losses and take profits."""
        open_trades = self.db.get_open_trades()
        to_close = self.risk.check_stop_losses(open_trades, current_prices)

        closed: list[Trade] = []
        for trade, reason in to_close:
            price = current_prices.get(trade.pair)
            result = await self.close_trade(trade, reason, price)
            if result.status != "open":
                closed.append(result)

        return closed

    async def emergency_close_all(self, reason: str) -> list[Trade]:
        """Close all open positions immediately — circuit breaker."""
        log.critical(f"EMERGENCY CLOSE ALL: {reason}")
        open_trades = self.db.get_open_trades()
        closed: list[Trade] = []

        for trade in open_trades:
            result = await self.close_trade(trade, "emergency_close")
            closed.append(result)

        return closed
