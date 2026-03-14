"""Exchange connectivity — Hyperliquid native API with transparent paper/live mode.

Paper mode: simulates fills using allMids prices, tracks positions in-memory.
Live mode: places real orders via Hyperliquid native REST API (EIP-712 signed).

No ccxt dependency — all API calls go through Hyperliquid's native REST endpoints.
"""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_sources import HyperliquidNativeClient

from eth_account import Account

from src.config import ExchangeConfig
from src.utils import log, now_iso


def parse_pair(pair: str) -> tuple[str, str]:
    """Parse pair format into (base, quote).

    Handles both simple ('BTC/USDC') and contract ('BTC/USDC:USDC') formats.
    Raises ValueError on malformed input.
    """
    if not pair or "/" not in pair:
        raise ValueError(f"Invalid pair format: {pair!r}")
    parts = pair.split("/")
    base = parts[0].strip()
    quote = parts[1].split(":")[0].strip() if len(parts) > 1 else ""
    if not base or not quote:
        raise ValueError(f"Empty base or quote in pair: {pair!r}")
    return base, quote


def _pair_to_hl_coin(pair: str) -> str:
    """Convert pair format to Hyperliquid native API coin name.

    Examples:
        'BTC/USDC:USDC' -> 'BTC'
        'XYZ-NVDA/USDC:USDC' -> 'xyz:NVDA'
    """
    base = parse_pair(pair)[0]
    if base.startswith("XYZ-"):
        return "xyz:" + base[4:]
    return base


class Exchange:
    """Unified exchange interface using Hyperliquid native REST API.

    Paper mode simulates fills locally using allMids prices.
    Live mode will use native HL exchange API (EIP-712 signed requests).
    """

    def __init__(
        self,
        config: ExchangeConfig,
        mode: str = "paper",
        starting_capital: float = 200.0,
        hl_client: "HyperliquidNativeClient | None" = None,
        api_key: str = "",
        api_secret: str = "",
    ):
        self.config = config
        self.mode = mode
        self.quote_currency = config.quote_currency

        # Native Hyperliquid client for price fetching (shared with DataFetcher)
        self._hl_client = hl_client
        self._mid_prices: dict[str, float] = {}
        self._mid_prices_ts: float = 0.0
        self._CACHE_TTL = 3.0  # seconds
        self._price_lock = asyncio.Lock()  # serialize allMids refreshes

        # Paper trading state
        self.paper_balance: dict[str, float] = {self.quote_currency: starting_capital}
        self.paper_orders: list[dict] = []
        self._order_counter = 0
        self._fee_rate = 0.00045  # 0.045% taker fee (Hyperliquid)

        # Live mode credentials (for native HL exchange API)
        self._api_key = api_key
        self._api_secret = api_secret

        self._hl_exchange = None
        self._hl_info = None
        self._wallet_address = ""

        if mode == "live":
            if not api_key or not api_secret:
                raise ValueError("Live mode requires exchange API credentials")
            self._init_hl_sdk(api_key, api_secret)

    def _init_hl_sdk(self, api_key: str, api_secret: str):
        """Initialize Hyperliquid SDK for live trading."""
        from hyperliquid.info import Info
        from hyperliquid.exchange import Exchange as HLExchange
        from hyperliquid.utils import constants

        # api_key is the private key, api_secret is the same for HL
        account = Account.from_key(api_key)
        self._wallet_address = account.address
        self._hl_info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self._hl_exchange = HLExchange(account, constants.MAINNET_API_URL)
        log.info(f"Hyperliquid SDK initialized — wallet: {self._wallet_address[:10]}...")

    def sync_paper_balance_from_trades(self, open_trades: list) -> None:
        """Reconcile paper_balance with open trades loaded from DB.

        After a restart, paper_balance resets to starting capital only.
        This method reconstructs asset balances from persistent open trades
        so that positions can be closed properly.
        """
        if self.mode != "paper" or not open_trades:
            return

        for trade in open_trades:
            base, quote = parse_pair(trade.pair)
            cost = trade.entry_price * trade.quantity
            fee = trade.fees if trade.fees else 0

            if trade.direction == "long":
                # We bought base with quote — deduct quote, add base
                self.paper_balance[base] = self.paper_balance.get(base, 0) + trade.quantity
                self.paper_balance[quote] = self.paper_balance.get(quote, 0) - cost - fee
            else:
                # We sold/shorted base — deduct base, add quote
                self.paper_balance[base] = self.paper_balance.get(base, 0) - trade.quantity
                self.paper_balance[quote] = self.paper_balance.get(quote, 0) + cost - fee

        synced = [(t.pair, t.direction, t.quantity) for t in open_trades]
        log.info(f"Paper balance synced with {len(open_trades)} open trade(s): {synced}")

    async def _refresh_prices(self) -> None:
        """Refresh allMids price cache if stale (2 API calls cover ALL pairs)."""
        if self._hl_client is None:
            return
        now = time.monotonic()
        if now - self._mid_prices_ts < self._CACHE_TTL and self._mid_prices:
            return
        async with self._price_lock:
            # Re-check after acquiring lock (another coroutine may have refreshed)
            if time.monotonic() - self._mid_prices_ts < self._CACHE_TTL and self._mid_prices:
                return
            perp_mids, xyz_mids = await asyncio.gather(
                self._hl_client.fetch_all_mids(),
                self._hl_client.fetch_all_mids_xyz(),
            )
            self._mid_prices = {**perp_mids, **xyz_mids}
            self._mid_prices_ts = time.monotonic()

    async def get_price(self, pair: str) -> float | None:
        """Get current mid price via native allMids."""
        await self._refresh_prices()
        coin = _pair_to_hl_coin(pair)
        return self._mid_prices.get(coin)

    async def close(self):
        """Cleanup. Client lifecycle managed externally (DataFetcher)."""
        pass

    # --- Market data ---

    async def fetch_ticker(self, pair: str) -> dict:
        """Fetch current price via native allMids. Returns dict with 'last'."""
        price = await self.get_price(pair)
        if price is None:
            raise ValueError(f"No price available for {pair}")
        return {"last": price, "bid": price, "ask": price}

    async def fetch_order_book(self, pair: str, limit: int = 10) -> dict:
        """Fetch order book. Paper mode returns synthetic book from mid price."""
        if self.mode == "paper":
            price = await self.get_price(pair) or 0
            spread = price * 0.0001  # 1 bps spread
            return {
                "bids": [[price - spread, 1.0]],
                "asks": [[price + spread, 1.0]],
            }
        raise NotImplementedError("Live order book requires HL SDK")

    async def fetch_tickers(self, pairs: list[str]) -> dict[str, dict]:
        """Fetch tickers for multiple pairs in one batch (single allMids refresh)."""
        await self._refresh_prices()
        result = {}
        for pair in pairs:
            coin = _pair_to_hl_coin(pair)
            price = self._mid_prices.get(coin)
            if price is not None:
                result[pair] = {"last": price, "bid": price, "ask": price}
        return result

    # --- Order execution ---

    @staticmethod
    def _validate_order(side: str, amount: float, price: float | None = None) -> None:
        """Validate order parameters. Raises ValueError on invalid input."""
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side {side!r}, must be 'buy' or 'sell'")
        if amount <= 0:
            raise ValueError(f"Amount must be positive, got {amount}")
        if price is not None and price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

    async def create_market_order(self, pair: str, side: str, amount: float) -> dict:
        self._validate_order(side, amount)
        if self.mode == "paper":
            return await self._paper_market_order(pair, side, amount)

        # Live: place market order via HL SDK
        coin = _pair_to_hl_coin(pair)
        is_buy = side == "buy"

        # Get current price for slippage limit (HL requires a limit price even for "market")
        price = await self.get_price(pair)
        if price is None:
            raise ValueError(f"Cannot get price for {pair}")
        # Set limit 0.5% away to act as market order with slippage protection
        limit_price = price * (1.005 if is_buy else 0.995)

        log.info(f"[LIVE] Placing {side.upper()} {amount} {coin} @ ~${price:.2f} (limit ${limit_price:.2f})")

        result = await asyncio.to_thread(
            self._hl_exchange.market_open,
            coin,
            is_buy,
            amount,
            None,  # px=None for market
            0.005,  # 0.5% slippage
        )

        status = result.get("status", "unknown")
        if status == "ok":
            response = result.get("response", {})
            data = response.get("data", {})
            statuses = data.get("statuses", [{}])
            fill = statuses[0] if statuses else {}
            filled_info = fill.get("filled", fill.get("resting", {}))
            fill_price = float(filled_info.get("avgPx", price))
            oid = filled_info.get("oid", str(self._order_counter))
            self._order_counter += 1

            log.info(f"[LIVE] Order filled: {side.upper()} {amount} {coin} @ ${fill_price:.2f}")
            return {
                "id": str(oid),
                "pair": pair,
                "side": side,
                "amount": amount,
                "price": fill_price,
                "cost": fill_price * amount,
                "fee": fill_price * amount * self._fee_rate,
                "timestamp": now_iso(),
                "status": "filled",
            }
        else:
            error_msg = result.get("response", str(result))
            raise Exception(f"HL order failed: {error_msg}")

    async def create_limit_order(self, pair: str, side: str, amount: float, price: float) -> dict:
        self._validate_order(side, amount, price)
        if self.mode == "paper":
            return self._paper_limit_order(pair, side, amount, price)
        raise NotImplementedError("Live limit orders require Hyperliquid SDK")

    async def cancel_order(self, order_id: str, pair: str) -> dict:
        if self.mode == "paper":
            return self._paper_cancel_order(order_id)
        raise NotImplementedError("Live order cancellation requires Hyperliquid SDK")

    async def fetch_order(self, order_id: str, pair: str) -> dict:
        """Fetch order status. Paper mode checks in-memory orders."""
        if self.mode == "paper":
            for order in self.paper_orders:
                if order["id"] == order_id:
                    return order
            raise Exception(f"Order {order_id} not found")
        raise NotImplementedError("Live order fetch requires Hyperliquid SDK")

    async def create_trigger_order(
        self, pair: str, side: str, amount: float, trigger_price: float, reduce_only: bool = True
    ) -> dict:
        """Place a trigger (stop) order. Paper mode returns a synthetic order ID."""
        self._validate_order(side, amount, trigger_price)
        if self.mode == "paper":
            self._order_counter += 1
            return {
                "id": f"paper-trigger-{self._order_counter}",
                "status": "open",
                "pair": pair,
                "side": side,
                "amount": amount,
                "triggerPrice": trigger_price,
            }
        raise NotImplementedError("Live trigger orders require Hyperliquid SDK")

    async def fetch_balance(self) -> dict:
        if self.mode == "paper":
            return {"free": deepcopy(self.paper_balance), "total": deepcopy(self.paper_balance)}
        # Live: query HL API for account state (perps + spot for unified accounts)
        state, spot_state = await asyncio.gather(
            asyncio.to_thread(self._hl_info.user_state, self._wallet_address),
            asyncio.to_thread(self._hl_info.spot_user_state, self._wallet_address),
        )
        # Perp margin value
        margin = state.get("marginSummary", {})
        perp_value = float(margin.get("accountValue", 0))
        margin_used = float(margin.get("totalMarginUsed", 0))

        # Spot USDC balance (unified accounts use spot as perp collateral)
        spot_usdc = 0.0
        for bal in spot_state.get("balances", []):
            if bal.get("coin") == "USDC":
                spot_usdc = float(bal.get("total", 0))
                break

        total = perp_value + spot_usdc
        available = total - margin_used
        return {
            "free": {self.quote_currency: available},
            "total": {self.quote_currency: total},
        }

    # --- Paper trading simulation ---

    async def _paper_market_order(self, pair: str, side: str, amount: float) -> dict:
        """Simulate a market order fill at current price + slippage."""
        ticker = await self.fetch_ticker(pair)
        price = ticker["last"]

        # Simulate slippage (0.05-0.15%)
        import random
        slippage = random.uniform(0.0005, 0.0015)
        if side == "buy":
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)

        cost = fill_price * amount
        fee = cost * self._fee_rate
        base, quote = parse_pair(pair)

        if side == "buy":
            required = cost + fee
            if self.paper_balance.get(quote, 0) < required:
                raise Exception(f"Insufficient {quote} balance: need {required:.2f}, have {self.paper_balance.get(quote, 0):.2f}")
            self.paper_balance[quote] = self.paper_balance.get(quote, 0) - required
            self.paper_balance[base] = self.paper_balance.get(base, 0) + amount
        else:
            if self.paper_balance.get(base, 0) < amount:
                raise Exception(f"Insufficient {base} balance: need {amount}, have {self.paper_balance.get(base, 0)}")
            self.paper_balance[base] = self.paper_balance.get(base, 0) - amount
            self.paper_balance[quote] = self.paper_balance.get(quote, 0) + (cost - fee)

        self._order_counter += 1
        order = {
            "id": f"paper-{self._order_counter}",
            "pair": pair,
            "side": side,
            "type": "market",
            "amount": amount,
            "price": fill_price,
            "cost": cost,
            "fee": fee,
            "status": "closed",
            "timestamp": now_iso(),
        }
        self.paper_orders.append(order)
        # Cap paper_orders to last 500 to prevent unbounded growth
        if len(self.paper_orders) > 500:
            self.paper_orders = self.paper_orders[-500:]
        log.info(f"[PAPER] {side.upper()} {amount:.6f} {base} @ {fill_price:.2f} (fee: {fee:.4f} {quote})")
        return order

    def _paper_limit_order(self, pair: str, side: str, amount: float, price: float) -> dict:
        self._order_counter += 1
        order = {
            "id": f"paper-{self._order_counter}",
            "pair": pair,
            "side": side,
            "type": "limit",
            "amount": amount,
            "price": price,
            "cost": price * amount,
            "fee": 0,
            "status": "open",
            "timestamp": now_iso(),
        }
        self.paper_orders.append(order)
        log.info(f"[PAPER] LIMIT {side.upper()} {amount:.6f} {parse_pair(pair)[0]} @ {price:.2f}")
        return order

    def _paper_cancel_order(self, order_id: str) -> dict:
        for order in self.paper_orders:
            if order["id"] == order_id and order["status"] == "open":
                order["status"] = "cancelled"
                log.info(f"[PAPER] Cancelled order {order_id}")
                return order
        raise Exception(f"Order {order_id} not found or not open")

    def get_paper_portfolio_value(self, prices: dict[str, float]) -> float:
        """Calculate total paper portfolio value in quote currency."""
        qc = self.quote_currency
        total = self.paper_balance.get(qc, 0)
        for asset, amount in self.paper_balance.items():
            if asset == qc or amount <= 0:
                continue
            matched = False
            for price_pair, price_val in prices.items():
                base, _ = parse_pair(price_pair)
                if base == asset:
                    total += amount * price_val
                    matched = True
                    break
            if not matched:
                log.debug(f"No price found for {asset} — skipping in portfolio value")
        return total
