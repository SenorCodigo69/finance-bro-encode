"""Cross-DEX price scanner — compares perp prices across DEX venues.

Scans Hyperliquid vs dYdX (and optionally GMX) for price divergence,
funding rate differences, and spread comparison. Used for oracle manipulation
detection and future multi-venue routing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import json

import aiohttp

from src.config import DexScannerConfig
from src.exchange import Exchange, parse_pair
from src.utils import log, now_iso

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB max per API response


def _safe_float(val, default: float = 0.0) -> float:
    """Parse float from API data, returning default on failure."""
    try:
        result = float(val)
        if not __import__("math").isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


async def _read_json(resp: aiohttp.ClientResponse, max_bytes: int = _MAX_RESPONSE_BYTES):
    """Read and parse JSON response with size limit."""
    raw = await resp.read()
    if len(raw) > max_bytes:
        log.warning(f"Response too large ({len(raw)} bytes), discarding")
        return None
    return json.loads(raw)


# --- Data models ---

@dataclass
class DexVenueSnapshot:
    venue: str
    pair: str
    mark_price: float | None = None
    index_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    spread_bps: float | None = None
    funding_rate: float | None = None
    open_interest: float | None = None
    taker_fee_bps: float = 0.0
    timestamp: str = ""
    latency_ms: float = 0.0


@dataclass
class DexScanResult:
    pair: str
    venue_snapshots: dict[str, DexVenueSnapshot] = field(default_factory=dict)
    best_bid_venue: str | None = None
    best_ask_venue: str | None = None
    max_price_divergence_pct: float = 0.0
    funding_divergence: dict[str, float] = field(default_factory=dict)
    arb_opportunity_bps: float = 0.0
    scan_time: str = ""


# --- Venue adapters ---

# Map our pair symbols to dYdX market tickers
_DYDX_PAIR_MAP = {
    "BTC/USDC:USDC": "BTC-USD",
    "ETH/USDC:USDC": "ETH-USD",
    "AAVE/USDC:USDC": "AAVE-USD",
}

# Map our pair symbols to GMX index token symbols (Arbitrum)
_GMX_PAIR_MAP = {
    "BTC/USDC:USDC": "BTC",
    "ETH/USDC:USDC": "ETH",
}

_DYDX_BASE_URL = "https://indexer.dydx.trade/v4"
_DYDX_TIMEOUT = 10
_GMX_STATS_URL = "https://arbitrum-api.gmxinfra.io"
_GMX_TIMEOUT = 10


class DexVenueAdapter(Protocol):
    """Protocol for DEX venue data adapters."""

    @property
    def venue_name(self) -> str: ...

    def supports_pair(self, pair: str) -> bool: ...

    async def fetch_snapshot(self, pair: str, session: aiohttp.ClientSession) -> DexVenueSnapshot | None: ...


class HyperliquidAdapter:
    """Fetch perp data from Hyperliquid info API."""

    venue_name = "hyperliquid"

    def supports_pair(self, pair: str) -> bool:
        return True  # Hyperliquid is our primary, supports all configured pairs

    async def fetch_snapshot(
        self, pair: str, session: aiohttp.ClientSession, exchange: Exchange | None = None,
    ) -> DexVenueSnapshot | None:
        start = time.monotonic()
        try:
            base, _ = parse_pair(pair)
            # Use Hyperliquid info API for meta + asset contexts
            async with session.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "metaAndAssetCtxs"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await _read_json(resp)

            if not data or not isinstance(data, list) or len(data) < 2:
                return None

            meta = data[0]
            asset_ctxs = data[1]
            universe = meta.get("universe", [])

            # Find our asset
            for i, asset_info in enumerate(universe):
                if asset_info.get("name") == base and i < len(asset_ctxs):
                    ctx = asset_ctxs[i]
                    mark = _safe_float(ctx.get("markPx", 0))
                    mid = _safe_float(ctx.get("midPx", 0)) if ctx.get("midPx") else mark
                    funding = _safe_float(ctx.get("funding", 0))
                    oi = _safe_float(ctx.get("openInterest", 0))

                    latency = (time.monotonic() - start) * 1000
                    return DexVenueSnapshot(
                        venue="hyperliquid",
                        pair=pair,
                        mark_price=mark if mark > 0 else None,
                        index_price=None,  # HL doesn't expose index separately here
                        best_bid=mid * 0.9999 if mid > 0 else None,  # Approximate
                        best_ask=mid * 1.0001 if mid > 0 else None,
                        spread_bps=2.0 if mid > 0 else None,  # HL typically ~2bps
                        funding_rate=funding,
                        open_interest=oi if oi > 0 else None,
                        taker_fee_bps=4.5,
                        timestamp=now_iso(),
                        latency_ms=latency,
                    )
            return None
        except Exception as e:
            log.debug(f"Hyperliquid scan failed for {pair}: {e}")
            return None


class DydxAdapter:
    """Fetch perp data from dYdX v4 indexer (free, no auth)."""

    venue_name = "dydx"

    def supports_pair(self, pair: str) -> bool:
        return pair in _DYDX_PAIR_MAP

    async def fetch_snapshot(self, pair: str, session: aiohttp.ClientSession) -> DexVenueSnapshot | None:
        dydx_ticker = _DYDX_PAIR_MAP.get(pair)
        if not dydx_ticker:
            return None

        start = time.monotonic()
        try:
            # Fetch market data
            url = f"{_DYDX_BASE_URL}/perpetualMarkets"
            async with session.get(
                url,
                params={"ticker": dydx_ticker},
                timeout=aiohttp.ClientTimeout(total=_DYDX_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await _read_json(resp)

            if not data:
                return None
            markets = data.get("markets", {})
            market = markets.get(dydx_ticker)
            if not market:
                return None

            oracle_price = _safe_float(market.get("oraclePrice", 0))
            next_funding = _safe_float(market.get("nextFundingRate", 0))
            oi = _safe_float(market.get("openInterest", 0))

            # Fetch order book for spread
            best_bid = None
            best_ask = None
            spread_bps = None
            try:
                ob_url = f"{_DYDX_BASE_URL}/orderbooks/perpetualMarket/{dydx_ticker}"
                async with session.get(
                    ob_url,
                    timeout=aiohttp.ClientTimeout(total=_DYDX_TIMEOUT),
                ) as ob_resp:
                    if ob_resp.status == 200:
                        ob_data = await _read_json(ob_resp)
                        if ob_data:
                            bids = ob_data.get("bids", [])[:50]
                            asks = ob_data.get("asks", [])[:50]
                            if bids:
                                best_bid = _safe_float(bids[0].get("price", 0))
                            if asks:
                                best_ask = _safe_float(asks[0].get("price", 0))
                            if best_bid and best_ask and best_bid > 0:
                                spread_bps = (best_ask - best_bid) / best_bid * 10000
            except Exception:
                pass  # Order book is nice-to-have

            latency = (time.monotonic() - start) * 1000
            return DexVenueSnapshot(
                venue="dydx",
                pair=pair,
                mark_price=oracle_price if oracle_price > 0 else None,
                index_price=oracle_price if oracle_price > 0 else None,
                best_bid=best_bid,
                best_ask=best_ask,
                spread_bps=spread_bps,
                funding_rate=next_funding,
                open_interest=oi if oi > 0 else None,
                taker_fee_bps=5.0,  # dYdX v4 taker fee
                timestamp=now_iso(),
                latency_ms=latency,
            )
        except Exception as e:
            log.debug(f"dYdX scan failed for {pair}: {e}")
            return None


class GmxAdapter:
    """Fetch perp data from GMX v2 stats API (Arbitrum)."""

    venue_name = "gmx"

    def supports_pair(self, pair: str) -> bool:
        return pair in _GMX_PAIR_MAP

    async def fetch_snapshot(self, pair: str, session: aiohttp.ClientSession) -> DexVenueSnapshot | None:
        symbol = _GMX_PAIR_MAP.get(pair)
        if not symbol:
            return None

        start = time.monotonic()
        try:
            # GMX prices endpoint
            async with session.get(
                f"{_GMX_STATS_URL}/prices/tickers",
                timeout=aiohttp.ClientTimeout(total=_GMX_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                tickers = await _read_json(resp)

            if not tickers or not isinstance(tickers, list):
                return None

            # Find our token's price data (cap iteration to avoid huge lists)
            for ticker in tickers[:200]:
                token_symbol = ticker.get("tokenSymbol", "")
                if token_symbol == symbol:
                    min_price = _safe_float(ticker.get("minPrice", 0)) / 1e30 if ticker.get("minPrice") else None
                    max_price = _safe_float(ticker.get("maxPrice", 0)) / 1e30 if ticker.get("maxPrice") else None

                    if min_price and max_price:
                        mid = (min_price + max_price) / 2
                        spread = (max_price - min_price) / mid * 10000 if mid > 0 else None
                    else:
                        mid = min_price or max_price
                        spread = None

                    latency = (time.monotonic() - start) * 1000
                    return DexVenueSnapshot(
                        venue="gmx",
                        pair=pair,
                        mark_price=mid,
                        index_price=None,
                        best_bid=min_price,
                        best_ask=max_price,
                        spread_bps=spread,
                        funding_rate=None,  # GMX uses borrow fees, not funding
                        open_interest=None,
                        taker_fee_bps=7.0,  # GMX ~0.07% for perps
                        timestamp=now_iso(),
                        latency_ms=latency,
                    )
            return None
        except Exception as e:
            log.debug(f"GMX scan failed for {pair}: {e}")
            return None


# --- Scanner orchestrator ---

_MAX_CACHE_AGE_SEC = 300  # 5 min hard limit on stale cache
_MAX_CACHE_PAIRS = 50     # Prune cache if it exceeds this many pairs


_DEFAULT_FEES_BPS = {"hyperliquid": 4.5, "dydx": 5.0, "gmx": 7.0}


class DexScanner:
    """Orchestrates cross-DEX price scanning across multiple venues."""

    def __init__(self, exchange: Exchange, config: DexScannerConfig):
        self._exchange = exchange
        self._config = config
        self._scan_cache: dict[str, DexScanResult] = {}
        self._cache_timestamps: dict[str, float] = {}

        # Build fee map from config (overrides defaults)
        self._venue_fees: dict[str, float] = dict(_DEFAULT_FEES_BPS)
        for v in config.venues:
            self._venue_fees[v.name] = v.taker_fee_bps

        # Build adapter list from config
        self._adapters: list = []
        self._adapters.append(HyperliquidAdapter())

        venue_map = {v.name: v for v in config.venues}
        if venue_map.get("dydx") and venue_map["dydx"].enabled:
            self._adapters.append(DydxAdapter())
        if venue_map.get("gmx") and venue_map["gmx"].enabled:
            self._adapters.append(GmxAdapter())

        log.info(
            f"DexScanner initialized: {len(self._adapters)} venue(s) "
            f"({', '.join(a.venue_name for a in self._adapters)})"
        )

    async def scan_pair(self, pair: str) -> DexScanResult:
        """Scan all venues for a single pair. Uses cache if fresh."""
        now = time.monotonic()
        cached_ts = self._cache_timestamps.get(pair, 0)
        if (now - cached_ts < self._config.scan_interval_sec
                and now - cached_ts < _MAX_CACHE_AGE_SEC):
            return self._scan_cache[pair]

        # Prune stale entries if cache grows too large
        if len(self._scan_cache) > _MAX_CACHE_PAIRS:
            oldest = sorted(self._cache_timestamps, key=self._cache_timestamps.get)
            for old_pair in oldest[:len(self._scan_cache) - _MAX_CACHE_PAIRS]:
                self._scan_cache.pop(old_pair, None)
                self._cache_timestamps.pop(old_pair, None)

        snapshots: dict[str, DexVenueSnapshot] = {}
        async with aiohttp.ClientSession() as session:
            for adapter in self._adapters:
                if not adapter.supports_pair(pair):
                    continue
                try:
                    if isinstance(adapter, HyperliquidAdapter):
                        snap = await adapter.fetch_snapshot(pair, session, self._exchange)
                    else:
                        snap = await adapter.fetch_snapshot(pair, session)
                    if snap:
                        # Override fee from config if available
                        cfg_fee = self._venue_fees.get(adapter.venue_name)
                        if cfg_fee is not None:
                            snap.taker_fee_bps = cfg_fee
                        snapshots[adapter.venue_name] = snap
                except Exception as e:
                    log.debug(f"Scan adapter {adapter.venue_name} failed for {pair}: {e}")

        result = self._build_result(pair, snapshots)
        self._scan_cache[pair] = result
        self._cache_timestamps[pair] = now
        return result

    async def scan_all(self, pairs: list[str]) -> dict[str, DexScanResult]:
        """Scan all configured pairs across all venues."""
        results = {}
        for pair in pairs:
            results[pair] = await self.scan_pair(pair)
        return results

    def get_best_venue(self, pair: str, side: str) -> str | None:
        """Get best venue for a trade based on cached scan. Returns venue name."""
        cached = self._scan_cache.get(pair)
        if not cached:
            return self._config.primary_venue if hasattr(self._config, 'primary_venue') else "hyperliquid"

        if side == "buy":
            return cached.best_ask_venue
        return cached.best_bid_venue

    def get_cached_results(self) -> dict[str, DexScanResult]:
        return dict(self._scan_cache)

    def _build_result(self, pair: str, snapshots: dict[str, DexVenueSnapshot]) -> DexScanResult:
        """Compute divergence, best venue, and arb opportunity from snapshots."""
        result = DexScanResult(pair=pair, venue_snapshots=snapshots, scan_time=now_iso())

        if len(snapshots) < 2:
            # Single venue — no divergence to compute
            if snapshots:
                venue_name = next(iter(snapshots))
                result.best_bid_venue = venue_name
                result.best_ask_venue = venue_name
            return result

        # Collect prices for divergence calc
        prices = {}
        for venue, snap in snapshots.items():
            if snap.mark_price and snap.mark_price > 0:
                prices[venue] = snap.mark_price

        if len(prices) >= 2:
            # Reject outlier prices (>10% from median) — oracle manipulation guard
            price_list = sorted(prices.values())
            median_p = price_list[len(price_list) // 2]
            if median_p > 0:
                for venue in list(prices.keys()):
                    if abs(prices[venue] - median_p) / median_p > 0.10:
                        log.warning(
                            f"Rejecting outlier price from {venue}: "
                            f"{prices[venue]:.2f} vs median {median_p:.2f}"
                        )
                        del snapshots[venue]
                        del prices[venue]

            if len(prices) >= 2:
                price_list = list(prices.values())
                min_p = min(price_list)
                max_p = max(price_list)
                mid_p = sum(price_list) / len(price_list)
                if mid_p > 0:
                    result.max_price_divergence_pct = (max_p - min_p) / mid_p * 100

        # Best bid (highest bid = best venue to sell)
        best_bid = 0.0
        for venue, snap in snapshots.items():
            if snap.best_bid and snap.best_bid > best_bid:
                best_bid = snap.best_bid
                result.best_bid_venue = venue

        # Best ask (lowest ask = best venue to buy)
        best_ask = float("inf")
        for venue, snap in snapshots.items():
            if snap.best_ask and snap.best_ask < best_ask:
                best_ask = snap.best_ask
                result.best_ask_venue = venue

        # Arb opportunity: buy at best ask, sell at best bid (minus fees)
        if best_bid > 0 and best_ask < float("inf") and result.best_bid_venue != result.best_ask_venue:
            buy_snap = snapshots.get(result.best_ask_venue)
            sell_snap = snapshots.get(result.best_bid_venue)
            if buy_snap and sell_snap:
                total_fee_bps = buy_snap.taker_fee_bps + sell_snap.taker_fee_bps
                gross_bps = (best_bid - best_ask) / best_ask * 10000
                result.arb_opportunity_bps = gross_bps - total_fee_bps

        # Funding divergence
        for venue, snap in snapshots.items():
            if snap.funding_rate is not None:
                result.funding_divergence[venue] = snap.funding_rate

        # Log alerts
        if result.max_price_divergence_pct > self._config.divergence_alert_pct:
            log.warning(
                f"DEX price divergence alert: {pair} divergence={result.max_price_divergence_pct:.2f}% "
                f"across {list(prices.keys())}"
            )

        return result
