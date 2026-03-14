"""Backtesting engine — tests strategies against historical OHLCV data.

Uses the SAME strategy and risk management code as live trading.
Simulates realistic fees (0.045%) and slippage (0.05-0.15%).

Usage:
    python scripts/backtest.py --pair BTC/USDC:USDC --days 30 --strategy momentum
    python scripts/backtest.py --pair ETH/USDC:USDC --days 90 --all-strategies
    python scripts/backtest.py --all-pairs --days 30 --all-strategies
    python scripts/backtest.py --all-pairs --days 30 --all-strategies --export results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ccxt
except ImportError:
    raise ImportError("Backtest script requires ccxt: pip install ccxt")
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import RiskConfig, StrategyConfig, load_config
from src.indicators import atr as compute_atr, compute_all
from src.models import PortfolioSnapshot, Signal, Trade
from src.strategy import (
    BaseStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEE_RATE = 0.00045        # 0.045% per trade (Hyperliquid taker fee)
SLIPPAGE_MIN = 0.0005     # 0.05% minimum slippage
SLIPPAGE_MAX = 0.0015     # 0.15% maximum slippage

# All 13 configured pairs
ALL_PAIRS = [
    "BTC/USDC:USDC",
    "ETH/USDC:USDC",
    "AAVE/USDC:USDC",
    "XYZ-AAPL/USDC:USDC",
    "XYZ-TSLA/USDC:USDC",
    "XYZ-NVDA/USDC:USDC",
    "XYZ-MSFT/USDC:USDC",
    "XYZ-GOLD/USDC:USDC",
    "XYZ-BRENTOIL/USDC:USDC",
    "XYZ-SILVER/USDC:USDC",
    "XYZ-EURUSD/USDC:USDC",
    "XYZ-GBPUSD/USDC:USDC",
    "XYZ-USDJPY/USDC:USDC",
]

# Map strategy names -> classes
STRATEGY_MAP: dict[str, type[BaseStrategy]] = {
    "momentum": MomentumStrategy,
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
}


# ---------------------------------------------------------------------------
# Simulated risk manager (no database dependency)
# ---------------------------------------------------------------------------

class BacktestRiskManager:
    """Mirrors src.risk_manager.RiskManager but without database dependencies.

    Uses the same position sizing, stop-loss, and take-profit formulas.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.max_drawdown = min(config.max_drawdown_pct, 0.30)

    def size_position(
        self,
        entry_price: float,
        stop_loss_price: float,
        portfolio: PortfolioSnapshot,
        confidence: float,
    ) -> float:
        risk_amount = portfolio.total_value * self.config.max_position_pct * confidence
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            return 0.0

        quantity = risk_amount / stop_distance
        max_value = portfolio.total_value * self.config.max_position_pct
        max_quantity = max_value / entry_price
        quantity = min(quantity, max_quantity)

        max_from_cash = (portfolio.cash * 0.95) / entry_price
        quantity = min(quantity, max_from_cash)
        return max(0.0, quantity)

    def calculate_stop_loss(
        self, direction: str, entry_price: float, atr_value: float
    ) -> float:
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
        stop_distance = abs(entry_price - stop_loss)
        tp_distance = stop_distance * rr_ratio

        if direction == "long":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    def check_drawdown(self, portfolio: PortfolioSnapshot) -> bool:
        return portfolio.drawdown_pct < self.max_drawdown


# ---------------------------------------------------------------------------
# Data models for backtest results
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    pair: str
    direction: str
    strategy: str
    entry_price: float
    exit_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    exit_time: str
    exit_reason: str  # "stop_loss" | "take_profit" | "end_of_data"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    pair: str
    days: int
    starting_capital: float
    final_capital: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_value: float
    avg_trade_duration_candles: int
    total_fees: float
    total_slippage: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(pair: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch historical OHLCV data from Hyperliquid, with Bybit fallback."""

    exchanges_to_try = [
        ("Hyperliquid", ccxt.hyperliquid({"enableRateLimit": True})),
        ("Bybit", ccxt.bybit({"enableRateLimit": True})),
    ]

    # Build pair variants for fallback matching
    base = pair.split("/")[0]
    pair_variants = [pair]
    if ":" in pair:
        pair_variants.append(pair.split(":")[0])  # BTC/USDC
    pair_variants.append(f"{base}/USDC")
    pair_variants.append(f"{base}/USDT")
    # Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for p in pair_variants:
        if p not in seen:
            seen.add(p)
            unique_variants.append(p)
    pair_variants = unique_variants

    for exch_name, exchange in exchanges_to_try:
        for try_pair in pair_variants:
            try:
                console.print(f"[cyan]Fetching {days}d of {timeframe} data for {try_pair} from {exch_name}...[/cyan]")

                since = exchange.parse8601(
                    (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat()
                )

                all_candles: list = []
                limit = 1000

                while True:
                    candles = exchange.fetch_ohlcv(try_pair, timeframe, since=since, limit=limit)
                    if not candles:
                        break
                    all_candles.extend(candles)
                    since = candles[-1][0] + 1
                    if len(candles) < limit:
                        break

                if not all_candles or len(all_candles) < 30:
                    console.print(f"[yellow]Only {len(all_candles)} candles from {exch_name} for {try_pair}, trying next...[/yellow]")
                    continue

                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                console.print(
                    f"[green]Fetched {len(df)} candles from {exch_name} "
                    f"({df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to "
                    f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d')})[/green]"
                )
                return df

            except Exception as e:
                console.print(f"[yellow]{exch_name} failed for {try_pair}: {e}[/yellow]")
                continue

    console.print(f"[red]No data available for {pair} from any exchange[/red]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Slippage simulation
# ---------------------------------------------------------------------------

def apply_slippage(price: float, direction: str, is_entry: bool) -> tuple[float, float]:
    """Apply random slippage to a price.

    Returns (adjusted_price, slippage_amount).
    Entry buys slip up, entry sells slip down, etc.
    """
    slip_pct = random.uniform(SLIPPAGE_MIN, SLIPPAGE_MAX)
    slip_amount = price * slip_pct

    # Entry long / exit short: price slips up (worse for buyer)
    # Entry short / exit long: price slips down (worse for seller)
    if (direction == "long" and is_entry) or (direction == "short" and not is_entry):
        return price + slip_amount, slip_amount
    else:
        return price - slip_amount, slip_amount


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    pair: str,
    strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    starting_capital: float,
    days: int,
) -> BacktestResult:
    """Run a strategy against historical data, candle by candle."""

    risk_mgr = BacktestRiskManager(risk_config)

    # State
    cash = starting_capital
    high_water_mark = starting_capital
    open_trade: BacktestTrade | None = None
    closed_trades: list[BacktestTrade] = []
    equity_curve: list[float] = []
    max_drawdown_pct = 0.0
    max_drawdown_value = 0.0
    trade_durations: list[int] = []
    candle_since_entry = 0

    # Strategy preferred timeframe for sizing the analysis window
    # We need at least 30 candles for indicators to warm up
    warmup = 35

    for i in range(warmup, len(df)):
        # Current candle (the one we're "on" — we see its OHLC at close)
        current = df.iloc[i]
        current_price = current["close"]
        current_time = str(current["timestamp"])

        # Portfolio value at this candle
        position_value = 0.0
        if open_trade is not None:
            if open_trade.direction == "long":
                position_value = open_trade.quantity * current_price
            else:
                # Short P&L: profit when price drops
                unrealised = (open_trade.entry_price - current_price) * open_trade.quantity
                position_value = (open_trade.entry_price * open_trade.quantity) + unrealised

        total_value = cash + position_value
        equity_curve.append(total_value)

        # Track drawdown
        if total_value > high_water_mark:
            high_water_mark = total_value
        dd_pct = (high_water_mark - total_value) / high_water_mark if high_water_mark > 0 else 0.0
        dd_value = high_water_mark - total_value
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)
        max_drawdown_value = max(max_drawdown_value, dd_value)

        # ---- Check open trade for stop-loss / take-profit ----
        if open_trade is not None:
            candle_since_entry += 1
            hit_sl = False
            hit_tp = False

            # Check against the candle's high and low (intra-bar)
            candle_high = current["high"]
            candle_low = current["low"]

            if open_trade.direction == "long":
                if candle_low <= open_trade.stop_loss:
                    hit_sl = True
                elif candle_high >= open_trade.take_profit:
                    hit_tp = True
            else:  # short
                if candle_high >= open_trade.stop_loss:
                    hit_sl = True
                elif candle_low <= open_trade.take_profit:
                    hit_tp = True

            if hit_sl or hit_tp:
                if hit_sl:
                    raw_exit = open_trade.stop_loss
                    reason = "stop_loss"
                else:
                    raw_exit = open_trade.take_profit
                    reason = "take_profit"

                exit_price, exit_slip = apply_slippage(raw_exit, open_trade.direction, is_entry=False)
                exit_fee = exit_price * open_trade.quantity * FEE_RATE

                if open_trade.direction == "long":
                    gross_pnl = (exit_price - open_trade.entry_price) * open_trade.quantity
                else:
                    gross_pnl = (open_trade.entry_price - exit_price) * open_trade.quantity

                net_pnl = gross_pnl - open_trade.fees - exit_fee
                pnl_pct = net_pnl / (open_trade.entry_price * open_trade.quantity) if open_trade.quantity > 0 else 0.0

                open_trade.exit_price = exit_price
                open_trade.exit_time = current_time
                open_trade.exit_reason = reason
                open_trade.pnl = net_pnl
                open_trade.pnl_pct = pnl_pct
                open_trade.fees += exit_fee
                open_trade.slippage += exit_slip

                # Return capital
                if open_trade.direction == "long":
                    cash += exit_price * open_trade.quantity - exit_fee
                else:
                    # Close short: return margin + P&L
                    cash += (open_trade.entry_price * open_trade.quantity) + gross_pnl - exit_fee

                trade_durations.append(candle_since_entry)
                closed_trades.append(open_trade)
                open_trade = None
                candle_since_entry = 0
                continue  # Don't open a new trade on the same candle we closed

        # ---- Circuit breaker check ----
        portfolio = PortfolioSnapshot(
            timestamp=current_time,
            total_value=total_value,
            cash=cash,
            positions_value=position_value,
            open_positions=1 if open_trade else 0,
            drawdown_pct=dd_pct,
            high_water_mark=high_water_mark,
            daily_pnl=0.0,
            total_pnl=total_value - starting_capital,
            total_pnl_pct=(total_value - starting_capital) / starting_capital,
        )

        if not risk_mgr.check_drawdown(portfolio):
            continue  # Skip trading while circuit breaker is active

        # ---- Generate signal (only if no open position) ----
        if open_trade is not None:
            continue

        # Build the data window the strategy expects: dict of {timeframe: DataFrame}
        window = df.iloc[max(0, i - 100) : i + 1].copy().reset_index(drop=True)
        if len(window) < 30:
            continue

        # Determine which timeframe key the strategy looks for
        preferred_tf = _strategy_preferred_tf(strategy)
        data_dict = {preferred_tf: window}

        try:
            signal = strategy.analyze(data_dict, pair, strategy_config)
        except Exception:
            continue

        if signal is None or signal.confidence < 0.5:
            continue

        # ---- Compute stops from ATR (same as live risk manager) ----
        enriched = compute_all(window, strategy_config.__dict__)
        atr_val = enriched["atr"].iloc[-2] if len(enriched) > 1 and not pd.isna(enriched["atr"].iloc[-2]) else None
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            continue

        stop_loss = risk_mgr.calculate_stop_loss(signal.direction, current_price, atr_val)
        take_profit = risk_mgr.calculate_take_profit(current_price, stop_loss, signal.direction)

        # ---- Position sizing (same formula as live) ----
        quantity = risk_mgr.size_position(current_price, stop_loss, portfolio, signal.confidence)
        if quantity <= 0:
            continue

        # ---- Apply entry slippage + fees ----
        entry_price, entry_slip = apply_slippage(current_price, signal.direction, is_entry=True)
        entry_fee = entry_price * quantity * FEE_RATE

        # Check we can afford the entry
        cost = entry_price * quantity + entry_fee
        if cost > cash:
            # Reduce quantity to fit budget
            quantity = (cash * 0.95) / (entry_price * (1 + FEE_RATE))
            if quantity <= 0:
                continue
            entry_fee = entry_price * quantity * FEE_RATE
            cost = entry_price * quantity + entry_fee

        # Deduct capital
        if signal.direction == "long":
            cash -= cost
        else:
            # Short: lock up the notional as margin
            cash -= entry_price * quantity + entry_fee

        open_trade = BacktestTrade(
            pair=pair,
            direction=signal.direction,
            strategy=strategy.name,
            entry_price=entry_price,
            exit_price=0.0,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=current_time,
            exit_time="",
            exit_reason="",
            fees=entry_fee,
            slippage=entry_slip,
        )
        candle_since_entry = 0

    # ---- Close any remaining open trade at last price ----
    if open_trade is not None:
        last = df.iloc[-1]
        exit_price, exit_slip = apply_slippage(last["close"], open_trade.direction, is_entry=False)
        exit_fee = exit_price * open_trade.quantity * FEE_RATE

        if open_trade.direction == "long":
            gross_pnl = (exit_price - open_trade.entry_price) * open_trade.quantity
        else:
            gross_pnl = (open_trade.entry_price - exit_price) * open_trade.quantity

        net_pnl = gross_pnl - open_trade.fees - exit_fee
        pnl_pct = net_pnl / (open_trade.entry_price * open_trade.quantity) if open_trade.quantity > 0 else 0.0

        open_trade.exit_price = exit_price
        open_trade.exit_time = str(last["timestamp"])
        open_trade.exit_reason = "end_of_data"
        open_trade.pnl = net_pnl
        open_trade.pnl_pct = pnl_pct
        open_trade.fees += exit_fee
        open_trade.slippage += exit_slip

        if open_trade.direction == "long":
            cash += exit_price * open_trade.quantity - exit_fee
        else:
            cash += (open_trade.entry_price * open_trade.quantity) + gross_pnl - exit_fee

        trade_durations.append(candle_since_entry)
        closed_trades.append(open_trade)

    # ---- Compute metrics ----
    final_capital = cash
    total_pnl = final_capital - starting_capital
    total_pnl_pct = total_pnl / starting_capital if starting_capital > 0 else 0.0

    wins = [t for t in closed_trades if t.pnl > 0]
    losses = [t for t in closed_trades if t.pnl <= 0]
    total_trades = len(closed_trades)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    largest_win = max((t.pnl for t in wins), default=0.0)
    largest_loss = min((t.pnl for t in losses), default=0.0)
    gross_wins = sum(t.pnl for t in wins)
    gross_losses = abs(sum(t.pnl for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0
    total_fees = sum(t.fees for t in closed_trades)
    total_slippage = sum(t.slippage for t in closed_trades)

    # Sharpe ratio (annualised, from per-trade returns)
    if len(closed_trades) >= 2:
        returns = [t.pnl_pct for t in closed_trades]
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = math.sqrt(var) if var > 0 else 0.0
        # Annualise assuming ~252 trading days, rough estimate of trades/year
        trades_per_year = max(1, total_trades * (365 / max(days, 1)))
        sharpe = (mean_ret / std_ret) * math.sqrt(trades_per_year) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    avg_duration = int(sum(trade_durations) / len(trade_durations)) if trade_durations else 0

    return BacktestResult(
        strategy_name=strategy.name,
        pair=pair,
        days=days,
        starting_capital=starting_capital,
        final_capital=final_capital,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_value=max_drawdown_value,
        avg_trade_duration_candles=avg_duration,
        total_fees=total_fees,
        total_slippage=total_slippage,
        trades=closed_trades,
        equity_curve=equity_curve,
    )


def _strategy_preferred_tf(strategy: BaseStrategy) -> str:
    """Return the timeframe each strategy prefers."""
    if isinstance(strategy, (TrendFollowingStrategy, BreakoutStrategy)):
        return "1h"
    return "15m"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results: list[BacktestResult]):
    """Print a rich summary table for all backtest results."""
    if not results:
        console.print("[red]No results to display.[/red]")
        return

    # Header
    r0 = results[0]
    console.print()
    console.print(
        Panel(
            f"[bold]Backtest Results[/bold]  |  {r0.pair}  |  {r0.days} days  |  "
            f"Capital: ${r0.starting_capital:,.2f}  |  "
            f"Fee: {FEE_RATE:.2%}  |  Slippage: {SLIPPAGE_MIN:.2%}-{SLIPPAGE_MAX:.2%}",
            style="cyan",
        )
    )

    # Summary table
    table = Table(title="Strategy Performance", show_lines=True)
    table.add_column("Metric", style="bold", min_width=24)
    for r in results:
        table.add_column(r.strategy_name, justify="right", min_width=16)

    # Rows
    metrics = [
        ("Total Trades", lambda r: str(r.total_trades)),
        ("Win / Loss", lambda r: f"{r.winning_trades} / {r.losing_trades}"),
        ("Win Rate", lambda r: f"{r.win_rate:.1%}"),
        ("Total P&L", lambda r: _colored_money(r.total_pnl)),
        ("Total Return", lambda r: _colored_pct(r.total_pnl_pct)),
        ("Avg Win", lambda r: f"${r.avg_win:+.2f}" if r.avg_win else "-"),
        ("Avg Loss", lambda r: f"${r.avg_loss:+.2f}" if r.avg_loss else "-"),
        ("Largest Win", lambda r: f"${r.largest_win:+.2f}" if r.largest_win else "-"),
        ("Largest Loss", lambda r: f"${r.largest_loss:+.2f}" if r.largest_loss else "-"),
        ("Profit Factor", lambda r: f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "INF"),
        ("Sharpe Ratio", lambda r: f"{r.sharpe_ratio:.2f}"),
        ("Max Drawdown", lambda r: f"{r.max_drawdown_pct:.1%} (${r.max_drawdown_value:.2f})"),
        ("Avg Duration", lambda r: f"{r.avg_trade_duration_candles} candles"),
        ("Total Fees", lambda r: f"${r.total_fees:.2f}"),
        ("Total Slippage", lambda r: f"${r.total_slippage:.2f}"),
        ("Final Capital", lambda r: f"${r.final_capital:,.2f}"),
    ]

    for label, fn in metrics:
        row_values = []
        for r in results:
            row_values.append(fn(r))
        table.add_row(label, *row_values)

    console.print(table)

    # Per-strategy trade lists (condensed)
    for r in results:
        if not r.trades:
            continue
        trade_table = Table(title=f"{r.strategy_name} — Trade Log ({len(r.trades)} trades)", show_lines=False)
        trade_table.add_column("#", justify="right", style="dim")
        trade_table.add_column("Dir", min_width=5)
        trade_table.add_column("Entry", justify="right")
        trade_table.add_column("Exit", justify="right")
        trade_table.add_column("Qty", justify="right")
        trade_table.add_column("P&L", justify="right")
        trade_table.add_column("P&L %", justify="right")
        trade_table.add_column("Reason")
        trade_table.add_column("Entry Time")

        for idx, t in enumerate(r.trades, 1):
            pnl_style = "green" if t.pnl > 0 else "red"
            trade_table.add_row(
                str(idx),
                t.direction.upper(),
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}",
                f"{t.quantity:.6f}",
                f"[{pnl_style}]${t.pnl:+.2f}[/{pnl_style}]",
                f"[{pnl_style}]{t.pnl_pct:+.2%}[/{pnl_style}]",
                t.exit_reason,
                t.entry_time[:19] if t.entry_time else "-",
            )

        console.print(trade_table)


def _colored_money(value: float) -> str:
    color = "green" if value >= 0 else "red"
    return f"[{color}]${value:+,.2f}[/{color}]"


def _colored_pct(value: float) -> str:
    color = "green" if value >= 0 else "red"
    return f"[{color}]{value:+.2%}[/{color}]"


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(results: list[BacktestResult], filepath: str):
    """Export all trades across strategies to a CSV file."""
    path = Path(filepath)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "pair", "direction", "entry_price", "exit_price",
            "quantity", "stop_loss", "take_profit", "pnl", "pnl_pct",
            "fees", "slippage", "exit_reason", "entry_time", "exit_time",
        ])
        for r in results:
            for t in r.trades:
                writer.writerow([
                    t.strategy, t.pair, t.direction, f"{t.entry_price:.8f}",
                    f"{t.exit_price:.8f}", f"{t.quantity:.8f}",
                    f"{t.stop_loss:.8f}", f"{t.take_profit:.8f}",
                    f"{t.pnl:.4f}", f"{t.pnl_pct:.6f}",
                    f"{t.fees:.4f}", f"{t.slippage:.4f}",
                    t.exit_reason, t.entry_time, t.exit_time,
                ])
    console.print(f"[green]Exported {sum(len(r.trades) for r in results)} trades to {path.resolve()}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest trading strategies against historical OHLCV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backtest.py --pair BTC/USDC:USDC --days 30 --strategy momentum
  python scripts/backtest.py --pair ETH/USDC:USDC --days 90 --all-strategies
  python scripts/backtest.py --all-pairs --days 30 --all-strategies
  python scripts/backtest.py --all-pairs --days 30 --all-strategies --export results.csv
        """,
    )
    parser.add_argument("--pair", type=str, default="BTC/USDC:USDC", help="Trading pair (default: BTC/USDC:USDC)")
    parser.add_argument("--all-pairs", action="store_true", help="Run all 7 configured pairs")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history (default: 30)")
    parser.add_argument("--strategy", type=str, choices=list(STRATEGY_MAP.keys()),
                        help="Run a specific strategy")
    parser.add_argument("--all-strategies", action="store_true", help="Run all strategies")
    parser.add_argument("--capital", type=float, default=None,
                        help="Starting capital in USDC (default: from config)")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Candle timeframe for data fetch (default: auto per strategy)")
    parser.add_argument("--export", type=str, default=None, metavar="FILE",
                        help="Export trades to CSV file")
    return parser.parse_args()


def print_multi_pair_summary(all_results: dict[str, list[BacktestResult]]):
    """Print a summary table across all pairs and strategies."""
    console.print()
    table = Table(title="Multi-Pair Strategy Summary", show_lines=True)
    table.add_column("Pair", style="bold", min_width=20)
    for name in STRATEGY_MAP:
        table.add_column(name, justify="right", min_width=14)
    table.add_column("Best", justify="center", min_width=16, style="bold green")

    weight_recommendations: dict[str, dict[str, float]] = {}

    for pair, results in all_results.items():
        row = []
        best_name = ""
        best_pnl = -float("inf")
        pair_weights: dict[str, float] = {}

        for r in results:
            pnl_str = _colored_money(r.total_pnl)
            wr = f"{r.win_rate:.0%}"
            row.append(f"{pnl_str}\n{r.total_trades}t / {wr}")
            if r.total_pnl > best_pnl:
                best_pnl = r.total_pnl
                best_name = r.strategy_name

        # Calculate weight recommendations based on performance
        total_pnl_all = sum(abs(r.total_pnl) for r in results) or 1.0
        for r in results:
            if r.total_trades == 0:
                pair_weights[r.strategy_name] = 0.5
            elif r.total_pnl > 0:
                # Winners get boosted proportionally
                pair_weights[r.strategy_name] = min(1.5, 1.0 + (r.total_pnl / total_pnl_all))
            else:
                # Losers get penalised
                pair_weights[r.strategy_name] = max(0.3, 1.0 - (abs(r.total_pnl) / total_pnl_all))

        # Normalize pair label
        short_pair = pair.split("/")[0].replace("XYZ-", "")
        weight_recommendations[pair] = pair_weights
        table.add_row(short_pair, *row, best_name)

    console.print(table)

    # Print recommended weights
    console.print()
    console.print(Panel("[bold]Recommended pair_weights for settings.yaml[/bold]", style="cyan"))
    for pair, weights in weight_recommendations.items():
        short = pair.split("/")[0].replace("XYZ-", "")
        console.print(f"  [bold]{short}[/bold]:")
        for strat, w in weights.items():
            marker = " [green]*[/green]" if w > 1.0 else " [red]v[/red]" if w < 0.8 else ""
            console.print(f"    {strat}: {w:.2f}{marker}")

    return weight_recommendations


def main():
    args = parse_args()

    if not args.strategy and not args.all_strategies:
        console.print("[red]Specify --strategy <name> or --all-strategies[/red]")
        sys.exit(1)

    # Load project config
    config = load_config()
    strategy_config = config.strategy
    risk_config = config.risk
    starting_capital = args.capital if args.capital is not None else risk_config.starting_capital

    # Determine which strategies to run
    if args.all_strategies:
        strategies = [cls() for cls in STRATEGY_MAP.values()]
    else:
        strategies = [STRATEGY_MAP[args.strategy]()]

    # Determine which pairs to run
    pairs = ALL_PAIRS if args.all_pairs else [args.pair]

    all_results: dict[str, list[BacktestResult]] = {}

    for pair in pairs:
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]  Backtesting: {pair}[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")

        pair_results: list[BacktestResult] = []

        # Fetch data per timeframe needed (strategies use different TFs)
        data_cache: dict[str, pd.DataFrame] = {}

        for strat in strategies:
            # Determine timeframe for this strategy
            tf = args.timeframe if args.timeframe else _strategy_preferred_tf(strat)
            if tf not in data_cache:
                data_cache[tf] = fetch_ohlcv(pair, tf, args.days)

            df = data_cache[tf]
            console.print(f"\n[bold cyan]Running {strat.name} on {pair} ({tf})...[/bold cyan]")
            result = run_backtest(
                df=df,
                strategy=strat,
                pair=pair,
                strategy_config=strategy_config,
                risk_config=risk_config,
                starting_capital=starting_capital,
                days=args.days,
            )
            pair_results.append(result)
            console.print(
                f"  [dim]{strat.name}: {result.total_trades} trades, "
                f"P&L ${result.total_pnl:+.2f} ({result.total_pnl_pct:+.2%})[/dim]"
            )

        all_results[pair] = pair_results
        print_results(pair_results)

    # Multi-pair summary
    if len(pairs) > 1:
        weight_recs = print_multi_pair_summary(all_results)

    # Export
    if args.export:
        flat = [r for results in all_results.values() for r in results]
        export_csv(flat, args.export)


if __name__ == "__main__":
    main()
