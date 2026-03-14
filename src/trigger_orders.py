"""On-chain trigger orders — stop losses and take profits on Hyperliquid.

Places SL/TP orders directly on Hyperliquid's order book via native REST API
so they execute even if the bot is offline. Paper mode tracks triggers
in-memory without hitting the API.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.config import TriggerOrderConfig
from src.database import Database
from src.exchange import Exchange
from src.models import Trade
from src.utils import log, now_iso


def _sanitize_error(e: Exception) -> str:
    """Strip potential secrets from error messages before storage."""
    msg = str(e)[:500]
    # Redact anything that looks like an API key or secret (32+ alphanumeric chars)
    msg = re.sub(r'[A-Za-z0-9]{32,}', '[REDACTED]', msg)
    return msg


@dataclass
class TriggerOrder:
    id: int | None
    trade_id: int
    exchange_order_id: str | None
    pair: str
    side: str
    trigger_price: float
    quantity: float
    order_type: str  # "stop_loss" | "take_profit" | "user_take_profit" | "bot_take_profit"
    status: str  # "pending" | "placed" | "triggered" | "cancelled" | "failed"
    placed_time: str | None = None
    triggered_time: str | None = None
    error: str | None = None


class TriggerOrderManager:
    """Manages on-chain trigger orders for stop losses and take profits."""

    def __init__(self, exchange: Exchange, db: Database, config: TriggerOrderConfig):
        self._exchange = exchange
        self._db = db
        self._config = config
        self._last_trail_update: dict[int, float] = {}  # trade_id -> monotonic timestamp
        self._last_trail_price: dict[int, float] = {}   # trade_id -> last trigger price

    async def place_triggers_for_trade(self, trade: Trade) -> list[TriggerOrder]:
        """Place SL and TP trigger orders for a newly opened trade.

        In paper mode: tracks triggers in DB but doesn't call exchange API.
        In live mode: places actual on-chain trigger orders via native HL API.
        """
        if trade.id is None:
            log.warning("Cannot place triggers for unsaved trade")
            return []

        triggers: list[TriggerOrder] = []

        # 1. Stop loss (entire position)
        if trade.stop_loss and trade.stop_loss > 0:
            sl_side = "sell" if trade.direction == "long" else "buy"
            sl_trigger = await self._place_single_trigger(
                trade_id=trade.id,
                pair=trade.pair,
                side=sl_side,
                trigger_price=trade.stop_loss,
                quantity=trade.quantity,
                order_type="stop_loss",
            )
            if sl_trigger:
                triggers.append(sl_trigger)

        # 2. User take profit (conservative, partial close — 50% of position)
        if trade.user_take_profit and trade.user_take_profit > 0:
            tp_side = "sell" if trade.direction == "long" else "buy"
            user_qty = trade.quantity * 0.5  # User portion
            tp_trigger = await self._place_single_trigger(
                trade_id=trade.id,
                pair=trade.pair,
                side=tp_side,
                trigger_price=trade.user_take_profit,
                quantity=user_qty,
                order_type="user_take_profit",
            )
            if tp_trigger:
                triggers.append(tp_trigger)

        # 3. Bot take profit (aggressive, remaining 50%)
        if trade.bot_take_profit and trade.bot_take_profit > 0:
            tp_side = "sell" if trade.direction == "long" else "buy"
            bot_qty = trade.quantity * 0.5  # Bot portion
            tp_trigger = await self._place_single_trigger(
                trade_id=trade.id,
                pair=trade.pair,
                side=tp_side,
                trigger_price=trade.bot_take_profit,
                quantity=bot_qty,
                order_type="bot_take_profit",
            )
            if tp_trigger:
                triggers.append(tp_trigger)

        if triggers:
            trade.trigger_orders_placed = True
            log.info(
                f"Placed {len(triggers)} trigger order(s) for trade {trade.id} "
                f"({trade.pair} {trade.direction})"
            )

        return triggers

    async def cancel_triggers_for_trade(self, trade_id: int) -> int:
        """Cancel all active trigger orders for a trade. Returns count cancelled."""
        active = self._db.get_trigger_orders_for_trade(trade_id)
        cancelled = 0

        for trigger in active:
            if trigger["status"] not in ("pending", "placed"):
                continue

            if self._exchange.mode == "live" and trigger.get("exchange_order_id"):
                try:
                    await self._exchange.cancel_order(
                        trigger["exchange_order_id"], trigger["pair"]
                    )
                except Exception as e:
                    log.warning(f"Failed to cancel trigger {trigger['id']}: {e}")
                    continue

            self._db.update_trigger_order(
                trigger["id"], status="cancelled"
            )
            cancelled += 1

        if cancelled:
            log.info(f"Cancelled {cancelled} trigger order(s) for trade {trade_id}")

        # Clean up tracking state
        self._last_trail_update.pop(trade_id, None)
        self._last_trail_price.pop(trade_id, None)

        return cancelled

    async def sync_trigger_status(self) -> list[TriggerOrder]:
        """Check which trigger orders have been filled on-chain.

        In paper mode: no-op (paper stops are handled by RiskManager).
        In live mode: queries exchange for order status updates.
        Returns list of triggers that were filled since last check.
        """
        if self._exchange.mode != "live":
            return []

        filled: list[TriggerOrder] = []
        active = self._db.get_active_trigger_orders()

        for trigger in active:
            if not trigger.get("exchange_order_id"):
                continue

            try:
                # Check order status on exchange
                order = await self._exchange.fetch_order(
                    trigger["exchange_order_id"], trigger["pair"]
                )
                if order.get("status") == "closed":
                    self._db.update_trigger_order(
                        trigger["id"],
                        status="triggered",
                        triggered_time=now_iso(),
                    )
                    filled.append(self._dict_to_trigger(trigger))
                    log.info(
                        f"Trigger order filled: {trigger['order_type']} for trade "
                        f"{trigger['trade_id']} at {trigger['trigger_price']}"
                    )
            except Exception as e:
                log.debug(f"Failed to check trigger {trigger['id']}: {e}")

        # Clean up trailing stop tracking for filled triggers (S8-M4)
        for t in filled:
            self._last_trail_update.pop(t.trade_id, None)
            self._last_trail_price.pop(t.trade_id, None)

        return filled

    async def update_trailing_triggers(self, trade: Trade, new_stop: float) -> bool:
        """Update stop loss trigger to new trailing stop level.

        Rate-limited: only updates if enough time has passed AND the price
        has moved enough to justify cancelling and replacing the trigger.
        """
        if trade.id is None:
            return False

        # Rate limit: minimum interval
        now = time.monotonic()
        last_update = self._last_trail_update.get(trade.id, 0)
        if now - last_update < self._config.trailing_update_min_interval_sec:
            return False

        # Rate limit: minimum price movement
        last_price = self._last_trail_price.get(trade.id)
        if last_price and last_price > 0:
            move_pct = abs(new_stop - last_price) / last_price * 100
            if move_pct < self._config.trailing_update_min_move_pct:
                return False

        # Find and cancel existing SL trigger
        triggers = self._db.get_trigger_orders_for_trade(trade.id)
        for trigger in triggers:
            if trigger["order_type"] == "stop_loss" and trigger["status"] in ("pending", "placed"):
                if self._exchange.mode == "live" and trigger.get("exchange_order_id"):
                    try:
                        await self._exchange.cancel_order(
                            trigger["exchange_order_id"], trigger["pair"]
                        )
                    except Exception as e:
                        log.warning(f"Failed to cancel old SL trigger: {e}")
                        return False
                self._db.update_trigger_order(trigger["id"], status="cancelled")

        # Place new SL trigger at updated trailing stop
        sl_side = "sell" if trade.direction == "long" else "buy"
        new_trigger = await self._place_single_trigger(
            trade_id=trade.id,
            pair=trade.pair,
            side=sl_side,
            trigger_price=new_stop,
            quantity=trade.quantity,
            order_type="stop_loss",
        )

        if new_trigger:
            self._last_trail_update[trade.id] = now
            self._last_trail_price[trade.id] = new_stop
            log.info(
                f"Updated trailing SL trigger for trade {trade.id}: "
                f"new stop={new_stop:.2f}"
            )
            return True

        return False

    async def _place_single_trigger(
        self,
        trade_id: int,
        pair: str,
        side: str,
        trigger_price: float,
        quantity: float,
        order_type: str,
    ) -> TriggerOrder | None:
        """Place a single trigger order on-chain or track in paper mode."""
        trigger_dict = {
            "trade_id": trade_id,
            "exchange_order_id": None,
            "pair": pair,
            "side": side,
            "trigger_price": trigger_price,
            "quantity": quantity,
            "order_type": order_type,
            "status": "pending",
            "placed_time": now_iso(),
            "triggered_time": None,
            "error": None,
        }

        if self._exchange.mode == "live":
            try:
                # Place trigger order via native HL API
                order = await self._exchange.create_trigger_order(
                    pair=pair,
                    side=side,
                    amount=quantity,
                    trigger_price=trigger_price,
                    reduce_only=True,
                )
                trigger_dict["exchange_order_id"] = order.get("id")
                trigger_dict["status"] = "placed"
            except Exception as e:
                log.warning(f"Failed to place {order_type} trigger for trade {trade_id}: {e}")
                trigger_dict["status"] = "failed"
                trigger_dict["error"] = _sanitize_error(e)
        else:
            # Paper mode: track but don't hit API
            trigger_dict["status"] = "placed"
            trigger_dict["exchange_order_id"] = f"paper-trigger-{trade_id}-{order_type}-{int(time.time())}"

        # Save to DB
        trigger_id = self._db.save_trigger_order(trigger_dict)

        trigger = TriggerOrder(
            id=trigger_id,
            trade_id=trade_id,
            exchange_order_id=trigger_dict["exchange_order_id"],
            pair=pair,
            side=side,
            trigger_price=trigger_price,
            quantity=quantity,
            order_type=order_type,
            status=trigger_dict["status"],
            placed_time=trigger_dict["placed_time"],
            triggered_time=None,
            error=trigger_dict.get("error"),
        )

        return trigger if trigger.status != "failed" else None

    @staticmethod
    def _dict_to_trigger(d: dict) -> TriggerOrder:
        return TriggerOrder(
            id=d.get("id"),
            trade_id=d["trade_id"],
            exchange_order_id=d.get("exchange_order_id"),
            pair=d["pair"],
            side=d["side"],
            trigger_price=d["trigger_price"],
            quantity=d["quantity"],
            order_type=d["order_type"],
            status=d["status"],
            placed_time=d.get("placed_time"),
            triggered_time=d.get("triggered_time"),
            error=d.get("error"),
        )
