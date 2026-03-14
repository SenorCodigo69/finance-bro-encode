"""Risk management — position sizing, stop losses, drawdown circuit breaker.

The 30% drawdown hard stop is non-negotiable. This module cannot be bypassed.

Dual profit-taking:
  - Each trade gets TWO virtual take-profit levels and trailing stops.
  - User portion (30%): conservative TP + tight trailing stop.
  - Bot portion (70%): aggressive TP + wide trailing stop, lets winners run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import EventBlockingConfig, ProfitSplitConfig, RiskConfig
from src.database import Database
from src.models import PortfolioSnapshot, Trade
from src.utils import log


# Hardcoded absolute maximums — config cannot exceed these
_ABSOLUTE_MAX_DRAWDOWN = 0.30
_ABSOLUTE_MAX_POSITION_PCT = 0.25  # Never risk more than 25% per trade
_ABSOLUTE_MAX_AGGREGATE_EXPOSURE = 0.30  # Never expose more than 30% of capital across all positions


class RiskManager:
    def __init__(self, config: RiskConfig, db: Database):
        self.config = config
        self.db = db
        self.profit_split = config.profit_split
        # Enforce hard caps regardless of config
        self.max_drawdown = min(config.max_drawdown_pct, _ABSOLUTE_MAX_DRAWDOWN)
        self.max_position_pct = min(config.max_position_pct, _ABSOLUTE_MAX_POSITION_PCT)

        # TradFi intelligence (set per cycle via set_tradfi_context)
        self._event_blocking: EventBlockingConfig | None = None
        self._earnings_blackout_fn = None  # callable: (pair) -> bool
        self._correlated_pairs: dict[str, list[str]] = {}
        self._iv_context: dict[str, dict] = {}

    def set_tradfi_context(
        self,
        event_blocking: EventBlockingConfig | None = None,
        earnings_blackout_fn=None,
        correlated_pairs: dict[str, list[str]] | None = None,
        iv_context: dict[str, dict] | None = None,
    ) -> None:
        """Inject TradFi intelligence data for current cycle."""
        self._event_blocking = event_blocking
        self._earnings_blackout_fn = earnings_blackout_fn
        self._correlated_pairs = correlated_pairs or {}
        self._iv_context = iv_context or {}

    def check_drawdown(self, portfolio: PortfolioSnapshot) -> bool:
        """Returns True if within limits, False if circuit breaker should trigger."""
        if portfolio.drawdown_pct >= self.max_drawdown:
            log.critical(
                f"CIRCUIT BREAKER: Drawdown {portfolio.drawdown_pct:.1%} >= "
                f"limit {self.max_drawdown:.1%}. Portfolio: ${portfolio.total_value:.2f}"
            )
            return False
        if portfolio.drawdown_pct >= self.max_drawdown * 0.8:
            log.warning(
                f"Drawdown warning: {portfolio.drawdown_pct:.1%} "
                f"(limit: {self.max_drawdown:.1%})"
            )
        return True

    def check_can_trade(
        self,
        portfolio: PortfolioSnapshot,
        pair: str | None = None,
    ) -> tuple[bool, str]:
        """Pre-trade checks. Returns (allowed, reason_if_blocked)."""
        # Drawdown check
        if not self.check_drawdown(portfolio):
            return False, "Max drawdown reached"

        # Max open positions
        if portfolio.open_positions >= self.config.max_open_positions:
            return False, f"Max open positions ({self.config.max_open_positions}) reached"

        # Aggregate exposure cap — hardcoded, cannot be bypassed
        if portfolio.total_value > 0:
            exposure_pct = portfolio.positions_value / portfolio.total_value
            if exposure_pct >= _ABSOLUTE_MAX_AGGREGATE_EXPOSURE:
                return False, (
                    f"Aggregate exposure limit: {exposure_pct:.0%} >= "
                    f"{_ABSOLUTE_MAX_AGGREGATE_EXPOSURE:.0%} cap"
                )

        # Daily trade limit
        daily_count = self.db.get_trade_count_today()
        if daily_count >= self.config.max_daily_trades:
            return False, f"Daily trade limit ({self.config.max_daily_trades}) reached"

        # Cooldown after loss
        last_loss = self.db.get_last_loss_time()
        if last_loss:
            cooldown_until = datetime.fromisoformat(last_loss) + timedelta(minutes=self.config.cooldown_after_loss_min)
            if datetime.now(timezone.utc) < cooldown_until:
                remaining = (cooldown_until - datetime.now(timezone.utc)).seconds // 60
                return False, f"Loss cooldown: {remaining}min remaining"

        # Event blocking (earnings + macro events) — hard gate
        if pair and self._event_blocking and self._event_blocking.enabled:
            if self._earnings_blackout_fn and self._earnings_blackout_fn(pair):
                return False, f"Earnings blackout for {pair}"

        # Correlated positions check — treat highly correlated open positions as single risk unit
        if pair and self._correlated_pairs:
            correlated = self._correlated_pairs.get(pair, [])
            open_trades = self.db.get_open_trades()
            open_pairs = {t.pair for t in open_trades}
            correlated_open = [p for p in correlated if p in open_pairs]
            if correlated_open:
                # Count correlated open positions as if they're the same position
                effective_positions = portfolio.open_positions + len(correlated_open)
                if effective_positions >= self.config.max_open_positions:
                    return False, (
                        f"Correlated positions limit: {pair} correlates with "
                        f"{', '.join(correlated_open)}"
                    )

        return True, ""

    def size_position(
        self,
        entry_price: float,
        stop_loss_price: float,
        portfolio: PortfolioSnapshot,
        confidence: float,
        pair: str | None = None,
    ) -> float:
        """Calculate position size based on risk per trade.

        Uses fixed-fractional risk: risk a percentage of BOT BALANCE per trade,
        scaled by signal confidence. User balance is never risked.
        """
        # Use bot_balance for sizing — user_balance is off-limits
        sizing_balance = portfolio.bot_balance if portfolio.bot_balance > 0 else portfolio.total_value

        # Base risk: max_position_pct of bot's pool, scaled by confidence
        risk_amount = sizing_balance * self.max_position_pct * confidence

        # IV-based adjustment: reduce size by 50% when IV is elevated (>50%)
        if pair and pair in self._iv_context:
            avg_iv = self._iv_context[pair].get("avg_iv", 0)
            if avg_iv > 0.5:
                risk_amount *= 0.5
                log.info(f"IV adjustment: {pair} avg_iv={avg_iv:.2f} — sizing reduced 50%")

        # Validate prices
        if entry_price <= 0 or stop_loss_price <= 0:
            log.warning(f"Invalid prices: entry={entry_price}, stop={stop_loss_price}")
            return 0.0

        # Distance to stop loss (in price)
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            log.warning("Stop distance is 0 — cannot size position")
            return 0.0

        # Position size = risk amount / stop distance
        quantity = risk_amount / stop_distance

        # Cap at max_position_pct of bot's pool value
        max_value = sizing_balance * self.max_position_pct
        max_quantity = max_value / entry_price
        quantity = min(quantity, max_quantity)

        # Ensure we don't spend more than available cash
        max_from_cash = (portfolio.cash * 0.95) / entry_price  # Keep 5% buffer
        quantity = min(quantity, max_from_cash)

        return max(0.0, quantity)

    def calculate_stop_loss(
        self, direction: str, entry_price: float, atr_value: float
    ) -> float:
        """ATR-based stop loss. Minimum: default_stop_loss_pct from config."""
        atr_stop_distance = 2.0 * atr_value
        min_stop_distance = entry_price * self.config.default_stop_loss_pct

        stop_distance = max(atr_stop_distance, min_stop_distance)

        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self, entry_price: float, stop_loss: float, direction: str, rr_ratio: float = 2.0
    ) -> float:
        """Calculate take profit based on risk-reward ratio."""
        stop_distance = abs(entry_price - stop_loss)
        tp_distance = stop_distance * rr_ratio

        if direction == "long":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    def calculate_dual_take_profits(
        self, entry_price: float, direction: str
    ) -> tuple[float, float, float, float]:
        """Calculate separate take-profit and trailing-stop levels for bot and user portions.

        Returns (user_take_profit, bot_take_profit, user_trailing_stop, bot_trailing_stop).
        The trailing stops are returned as absolute price levels based on entry.
        """
        ps = self.profit_split

        if direction == "long":
            user_tp = entry_price * (1 + ps.user_take_profit_pct)
            bot_tp = entry_price * (1 + ps.bot_take_profit_pct)
            user_trail = entry_price * (1 - ps.user_trailing_stop_pct)
            bot_trail = entry_price * (1 - ps.bot_trailing_stop_pct)
        else:  # short
            user_tp = entry_price * (1 - ps.user_take_profit_pct)
            bot_tp = entry_price * (1 - ps.bot_take_profit_pct)
            user_trail = entry_price * (1 + ps.user_trailing_stop_pct)
            bot_trail = entry_price * (1 + ps.bot_trailing_stop_pct)

        return user_tp, bot_tp, user_trail, bot_trail

    def check_stop_losses(
        self, open_trades: list[Trade], current_prices: dict[str, float]
    ) -> list[tuple[Trade, str]]:
        """Check which trades hit stop loss, take profit, or max hold time.

        Supports dual profit-taking: user portion can close at its conservative TP
        while bot portion keeps running. A trade fully closes only when:
          - Stop loss is hit (both portions close), OR
          - Bot's take profit / trailing stop is hit (final close), OR
          - Both portions have already been closed, OR
          - Max hold time exceeded (auto-close to prevent stale positions).

        Returns list of (trade, reason) tuples for trades that should fully close.
        """
        to_close: list[tuple[Trade, str]] = []

        for trade in open_trades:
            # --- Check max hold time (if configured) ---
            if self.config.max_hold_hours > 0 and trade.entry_time:
                try:
                    entry_dt = datetime.fromisoformat(trade.entry_time)
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    hold_duration = datetime.now(timezone.utc) - entry_dt
                    max_hold = timedelta(hours=self.config.max_hold_hours)
                    if hold_duration >= max_hold:
                        hours_held = hold_duration.total_seconds() / 3600
                        log.info(
                            f"MAX HOLD TIME: {trade.pair} {trade.direction.upper()} "
                            f"held {hours_held:.1f}h >= {self.config.max_hold_hours}h limit"
                        )
                        to_close.append((trade, "max_hold_time"))
                        continue
                except (ValueError, TypeError):
                    pass  # Malformed entry_time — skip hold check
            price = current_prices.get(trade.pair)
            if price is None:
                continue

            # --- Check hard stop loss (applies to ENTIRE position) ---
            if trade.direction == "long" and price <= trade.stop_loss:
                to_close.append((trade, "stop_loss"))
                continue
            elif trade.direction == "short" and price >= trade.stop_loss:
                to_close.append((trade, "stop_loss"))
                continue

            # --- Dual TP checks (only if dual TPs are set) ---
            has_dual = trade.user_take_profit is not None and trade.bot_take_profit is not None

            if has_dual:
                # Check user portion (conservative, closes first)
                if not trade.user_portion_closed:
                    user_hit = (
                        (trade.direction == "long" and price >= trade.user_take_profit)
                        or (trade.direction == "short" and price <= trade.user_take_profit)
                    )
                    # Also check user trailing stop
                    user_trail_hit = False
                    if trade.user_trailing_stop is not None:
                        user_trail_hit = (
                            (trade.direction == "long" and price <= trade.user_trailing_stop)
                            or (trade.direction == "short" and price >= trade.user_trailing_stop)
                        )
                    if user_hit or user_trail_hit:
                        trade.user_portion_closed = True
                        reason = "user_take_profit" if user_hit else "user_trailing_stop"
                        log.info(
                            f"USER PORTION CLOSED ({reason}): {trade.pair} "
                            f"@ {price:.2f} (entry: {trade.entry_price:.2f})"
                        )

                # Check bot portion (aggressive, stays open longer)
                if not trade.bot_portion_closed:
                    bot_hit = (
                        (trade.direction == "long" and price >= trade.bot_take_profit)
                        or (trade.direction == "short" and price <= trade.bot_take_profit)
                    )
                    bot_trail_hit = False
                    if trade.bot_trailing_stop is not None:
                        bot_trail_hit = (
                            (trade.direction == "long" and price <= trade.bot_trailing_stop)
                            or (trade.direction == "short" and price >= trade.bot_trailing_stop)
                        )
                    if bot_hit or bot_trail_hit:
                        trade.bot_portion_closed = True
                        reason = "bot_take_profit" if bot_hit else "bot_trailing_stop"
                        log.info(
                            f"BOT PORTION CLOSED ({reason}): {trade.pair} "
                            f"@ {price:.2f} (entry: {trade.entry_price:.2f})"
                        )

                # Only fully close when both portions are done
                if trade.user_portion_closed and trade.bot_portion_closed:
                    to_close.append((trade, "take_profit"))

            else:
                # Legacy single take-profit behavior
                if trade.direction == "long":
                    if trade.take_profit and price >= trade.take_profit:
                        to_close.append((trade, "take_profit"))
                else:
                    if trade.take_profit and price <= trade.take_profit:
                        to_close.append((trade, "take_profit"))

        return to_close

    def update_trailing_stops(
        self, open_trades: list[Trade], current_prices: dict[str, float], atr_values: dict[str, float]
    ) -> list[Trade]:
        """Trail stops for profitable trades. Returns trades with updated stop losses.

        Updates three trailing stops:
          1. The main stop_loss (hard stop for entire position, ATR-based).
          2. user_trailing_stop (tighter, percentage-based).
          3. bot_trailing_stop (wider, percentage-based).
        """
        updated = []
        ps = self.profit_split

        for trade in open_trades:
            price = current_prices.get(trade.pair)
            atr_val = atr_values.get(trade.pair)
            if price is None or atr_val is None:
                continue

            changed = False
            trail_distance = 1.5 * atr_val

            if trade.direction == "long" and price > trade.entry_price:
                # Main ATR-based trailing stop
                new_stop = price - trail_distance
                if new_stop > trade.stop_loss:
                    trade.stop_loss = new_stop
                    changed = True
                    log.info(f"Trailing stop updated: {trade.pair} LONG stop -> {new_stop:.2f}")

                # User trailing stop (tighter — 2% below current price)
                if not trade.user_portion_closed and trade.user_trailing_stop is not None:
                    new_user_trail = price * (1 - ps.user_trailing_stop_pct)
                    if new_user_trail > trade.user_trailing_stop:
                        trade.user_trailing_stop = new_user_trail
                        changed = True
                        log.debug(f"User trailing stop: {trade.pair} -> {new_user_trail:.2f}")

                # Bot trailing stop (wider — 3.5% below current price)
                if not trade.bot_portion_closed and trade.bot_trailing_stop is not None:
                    new_bot_trail = price * (1 - ps.bot_trailing_stop_pct)
                    if new_bot_trail > trade.bot_trailing_stop:
                        trade.bot_trailing_stop = new_bot_trail
                        changed = True
                        log.debug(f"Bot trailing stop: {trade.pair} -> {new_bot_trail:.2f}")

            elif trade.direction == "short" and price < trade.entry_price:
                # Main ATR-based trailing stop
                new_stop = price + trail_distance
                if new_stop < trade.stop_loss:
                    trade.stop_loss = new_stop
                    changed = True
                    log.info(f"Trailing stop updated: {trade.pair} SHORT stop -> {new_stop:.2f}")

                # User trailing stop (tighter — 2% above current price)
                if not trade.user_portion_closed and trade.user_trailing_stop is not None:
                    new_user_trail = price * (1 + ps.user_trailing_stop_pct)
                    if new_user_trail < trade.user_trailing_stop:
                        trade.user_trailing_stop = new_user_trail
                        changed = True
                        log.debug(f"User trailing stop: {trade.pair} -> {new_user_trail:.2f}")

                # Bot trailing stop (wider — 3.5% above current price)
                if not trade.bot_portion_closed and trade.bot_trailing_stop is not None:
                    new_bot_trail = price * (1 + ps.bot_trailing_stop_pct)
                    if new_bot_trail < trade.bot_trailing_stop:
                        trade.bot_trailing_stop = new_bot_trail
                        changed = True
                        log.debug(f"Bot trailing stop: {trade.pair} -> {new_bot_trail:.2f}")

            if changed:
                updated.append(trade)

        return updated
