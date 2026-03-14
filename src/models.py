"""Data models for the trading agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OHLCV:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    pair: str
    timeframe: str
    direction: str  # "long" | "short" | "hold"
    confidence: float  # 0.0 - 1.0
    strategy_name: str
    indicators: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    timestamp: str = ""


@dataclass
class Trade:
    id: int | None
    pair: str
    direction: str  # "long" | "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float | None = None
    exit_price: float | None = None
    status: str = "open"  # "open" | "closed" | "stopped_out" | "cancelled"
    pnl: float | None = None
    pnl_pct: float | None = None
    entry_time: str = ""
    exit_time: str | None = None
    signal_data: dict[str, Any] = field(default_factory=dict)
    ai_reasoning: str = ""
    review_notes: str | None = None
    fees: float = 0.0

    # Dual profit-taking fields
    bot_take_profit: float | None = None   # Aggressive TP for bot's portion
    user_take_profit: float | None = None  # Conservative TP for user's portion
    bot_trailing_stop: float | None = None   # Wider trailing stop for bot
    user_trailing_stop: float | None = None  # Tighter trailing stop for user
    user_portion_closed: bool = False  # True once user's portion has been TP'd
    bot_portion_closed: bool = False   # True once bot's portion has been TP'd
    profit_split_bot: float | None = None   # Bot's share of profit ($)
    profit_split_user: float | None = None  # User's share of profit ($)

    # Phase 5: Execution tracking
    execution_venue: str = "hyperliquid"
    slippage_actual_bps: float | None = None
    trigger_orders_placed: bool = False


@dataclass
class PortfolioSnapshot:
    timestamp: str
    total_value: float
    cash: float
    positions_value: float
    open_positions: int
    drawdown_pct: float
    high_water_mark: float
    daily_pnl: float
    total_pnl: float
    total_pnl_pct: float

    # Dual pool balances
    bot_balance: float = 0.0   # Bot's reinvestment pool (aggressive)
    user_balance: float = 0.0  # User's conservative withdrawal pool
