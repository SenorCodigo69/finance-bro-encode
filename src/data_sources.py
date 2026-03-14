"""Multi-source data layer — 3x price sources per asset class with cross-validation.

Sources per asset class:
  Crypto:       Hyperliquid (primary) + DeFi Llama + CoinGecko
  Stocks:       Hyperliquid (primary) + yfinance + Alpha Vantage
  Commodities:  Hyperliquid (primary) + yfinance + Alpha Vantage

OHLCV and prices come from Hyperliquid's native REST API (no ccxt overhead).
Secondary sources validate price sanity.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import aiohttp

from src.config import DataSourcesConfig
from src.exchange import _pair_to_hl_coin, parse_pair
from src.utils import log


def _safe_float(v: Any, default: float | None = None) -> float | None:
    """Convert to float, returning default if result is inf/nan.

    [S7-02] Prevents inf/nan from API responses propagating through calculations.
    """
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ── Asset class classification ───────────────────────────────────────────

_CRYPTO_BASES = {"BTC", "ETH", "AAVE", "SOL", "DOGE", "LINK", "UNI", "ARB", "ONDO"}
_STOCK_PREFIXES = {"XYZ-AAPL", "XYZ-TSLA", "XYZ-NVDA", "XYZ-MSFT", "XYZ-TSM"}
_COMMODITY_PREFIXES = {"XYZ-GOLD", "XYZ-BRENTOIL", "XYZ-SILVER", "XYZ-NATGAS"}
_INDEX_PREFIXES = {"XYZ-XYZ100", "XYZ-JP225", "ABCD-USA500"}
_FOREX_PREFIXES = {"XYZ-EUR", "XYZ-GBP", "XYZ-JPY"}


def classify_pair(pair_base: str) -> str:
    """Map a pair's base asset to an asset class."""
    if pair_base in _CRYPTO_BASES:
        return "crypto"
    if pair_base in _STOCK_PREFIXES:
        return "stocks"
    if pair_base in _COMMODITY_PREFIXES:
        return "commodities"
    if pair_base in _INDEX_PREFIXES:
        return "indices"
    if pair_base in _FOREX_PREFIXES:
        return "forex"
    # Default: if starts with XYZ- assume synthetic (no secondary sources)
    if pair_base.startswith("XYZ-") or pair_base.startswith("ABCD-"):
        return "synthetics"
    return "crypto"


# ── External identifier mappings ─────────────────────────────────────────

CRYPTO_COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "AAVE": "aave",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ARB": "arbitrum",
    "ONDO": "ondo-finance",
}

CRYPTO_DEFILLAMA_IDS: dict[str, str] = {
    "BTC": "coingecko:bitcoin",
    "ETH": "coingecko:ethereum",
    "AAVE": "coingecko:aave",
    "SOL": "coingecko:solana",
    "DOGE": "coingecko:dogecoin",
    "LINK": "coingecko:chainlink",
    "UNI": "coingecko:uniswap",
    "ARB": "coingecko:arbitrum",
    "ONDO": "coingecko:ondo-finance",
}

STOCK_TICKERS: dict[str, str] = {
    "XYZ-AAPL": "AAPL",
    "XYZ-TSLA": "TSLA",
    "XYZ-NVDA": "NVDA",
    "XYZ-MSFT": "MSFT",
    "XYZ-TSM": "TSM",
}

COMMODITY_YFINANCE: dict[str, str] = {
    "XYZ-GOLD": "GC=F",
    "XYZ-BRENTOIL": "BZ=F",
    "XYZ-SILVER": "SI=F",
    "XYZ-NATGAS": "NG=F",
}

COMMODITY_ALPHAVANTAGE: dict[str, str] = {
    "XYZ-BRENTOIL": "BRENT",
    "XYZ-GOLD": "GLD",      # ETF proxy
    "XYZ-SILVER": "SLV",    # ETF proxy
    "XYZ-NATGAS": "UNG",    # ETF proxy
}

# ── Timeframe conversion ────────────────────────────────────────────────

_TF_TO_YFINANCE: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "1h", "1d": "1d",
}

# Timeframe durations in milliseconds (for native API start/end time calculation)
_TF_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


# ── Hyperliquid coin name mapping (canonical in exchange.py) ──────────────

# _pair_to_hl_coin is imported from exchange.py (single source of truth)


def _is_synthetic(pair: str) -> bool:
    """Check if a pair is a Hyperliquid synthetic (XYZ-prefixed)."""
    return parse_pair(pair)[0].startswith("XYZ-")


# ── FetchResult ──────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    """Result of a multi-source price fetch with validation."""
    candles: list[list] = field(default_factory=list)
    latest_price: float | None = None
    source_prices: dict[str, float] = field(default_factory=dict)
    divergence_pct: float = 0.0
    price_valid: bool = True
    source_latencies: dict[str, float] = field(default_factory=dict)
    anomaly_flag: bool = False


# ── DataSource protocol ─────────────────────────────────────────────────

@runtime_checkable
class DataSource(Protocol):
    name: str

    async def fetch_latest_price(self, identifier: str) -> float | None:
        """Fetch latest price. identifier format depends on adapter."""
        ...

    def supports_pair(self, pair_base: str) -> bool:
        """Check if this source can provide data for the given pair base."""
        ...


# ── Hyperliquid Native API Client ────────────────────────────────────────

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB safety limit


class HyperliquidNativeClient:
    """Direct Hyperliquid REST API client using aiohttp.

    Bypasses ccxt entirely — no rate limiter overhead, true parallel requests.
    All endpoints go through POST https://api.hyperliquid.xyz/info.
    """

    _BASE_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self, base_url: str | None = None):
        self._base_url = base_url or self._BASE_URL
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def _post(self, payload: dict) -> Any:
        """POST to the info endpoint with size-limited response parsing."""
        session = await self._ensure_session()
        async with session.post(
            self._base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            raw = await resp.read()
            if len(raw) > _MAX_RESPONSE_BYTES:
                log.warning(f"HL response too large ({len(raw)} bytes), discarding")
                return None
            if resp.status != 200:
                log.debug(f"HL API error {resp.status}: {raw[:200]}")
                return None
            return json.loads(raw)

    async def fetch_all_mids(self) -> dict[str, float]:
        """Fetch all perp mid prices in one call. Weight: 2."""
        data = await self._post({"type": "allMids"})
        if not data or not isinstance(data, dict):
            return {}
        result = {}
        for coin, price_str in data.items():
            price = _safe_float(price_str)
            if price is not None:
                result[coin] = price
        return result

    async def fetch_all_mids_xyz(self) -> dict[str, float]:
        """Fetch all xyz synthetic mid prices. Weight: 2."""
        data = await self._post({"type": "allMids", "dex": "xyz"})
        if not data or not isinstance(data, dict):
            return {}
        result = {}
        for coin, price_str in data.items():
            price = _safe_float(price_str)
            if price is not None:
                result[coin] = price
        return result

    async def fetch_candles(
        self, coin: str, interval: str, start_time: int, end_time: int
    ) -> list[list]:
        """Fetch OHLCV candles for one coin. Weight: 20 + items/60.

        Returns candles in ccxt format: [[timestamp, open, high, low, close, volume], ...]
        """
        data = await self._post({
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
            },
        })
        if not data or not isinstance(data, list):
            return []
        candles = []
        for c in data:
            try:
                candles.append([
                    c["t"],              # timestamp (open time ms)
                    float(c["o"]),       # open
                    float(c["h"]),       # high
                    float(c["l"]),       # low
                    float(c["c"]),       # close
                    float(c["v"]),       # volume
                ])
            except (KeyError, ValueError, TypeError):
                continue
        return sorted(candles, key=lambda x: x[0])

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ── Adapters ─────────────────────────────────────────────────────────────

class HyperliquidSource:
    """Primary source — Hyperliquid native REST API (no ccxt)."""
    name = "hyperliquid"

    def __init__(self, client: HyperliquidNativeClient):
        self._client = client
        # Cache allMids results briefly to avoid redundant calls within a cycle
        self._all_mids: dict[str, float] = {}
        self._all_mids_ts: float = 0.0
        self._CACHE_TTL = 5.0  # seconds

    async def _refresh_mids(self) -> None:
        """Refresh allMids cache if stale (both perp and xyz dex)."""
        now = time.monotonic()
        if now - self._all_mids_ts < self._CACHE_TTL and self._all_mids:
            return
        # Fetch perp and xyz mids in parallel
        perp_mids, xyz_mids = await asyncio.gather(
            self._client.fetch_all_mids(),
            self._client.fetch_all_mids_xyz(),
        )
        self._all_mids = {**perp_mids, **xyz_mids}
        self._all_mids_ts = now

    async def fetch_latest_price(self, pair: str) -> float | None:
        """Get price from cached allMids (2 API calls cover ALL pairs)."""
        await self._refresh_mids()
        hl_coin = _pair_to_hl_coin(pair)
        return self._all_mids.get(hl_coin)

    async def fetch_ohlcv(self, pair: str, timeframe: str, limit: int) -> list[list] | None:
        """Fetch candles via native candleSnapshot endpoint."""
        hl_coin = _pair_to_hl_coin(pair)
        tf_ms = _TF_MS.get(timeframe, 3_600_000)
        end_time = int(time.time() * 1000)
        start_time = end_time - (limit * tf_ms)
        try:
            candles = await self._client.fetch_candles(
                hl_coin, timeframe, start_time, end_time
            )
            return candles if candles else None
        except Exception:
            return None

    def supports_pair(self, pair_base: str) -> bool:
        return True


class DefiLlamaSource:
    """DeFi Llama — aggregated on-chain prices. Free, no auth."""
    name = "defillama"
    _BASE_URL = "https://coins.llama.fi"

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session

    async def fetch_latest_price(self, pair_base: str) -> float | None:
        llama_id = CRYPTO_DEFILLAMA_IDS.get(pair_base)
        if not llama_id:
            return None
        session = self._session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        try:
            url = f"{self._BASE_URL}/prices/current/{llama_id}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                coin_data = data.get("coins", {}).get(llama_id, {})
                return _safe_float(coin_data.get("price"))
        except Exception:
            return None
        finally:
            if not self._session:
                await session.close()

    def supports_pair(self, pair_base: str) -> bool:
        return pair_base in CRYPTO_DEFILLAMA_IDS


class CoinGeckoSource:
    """CoinGecko — free tier, already used in macro_analyst."""
    name = "coingecko"
    _BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: str = "", session: aiohttp.ClientSession | None = None):
        self._api_key = api_key
        self._session = session

    async def fetch_latest_price(self, pair_base: str) -> float | None:
        cg_id = CRYPTO_COINGECKO_IDS.get(pair_base)
        if not cg_id:
            return None
        session = self._session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        try:
            url = f"{self._BASE_URL}/simple/price?ids={cg_id}&vs_currencies=usd"
            headers = {}
            if self._api_key:
                headers["x-cg-demo-api-key"] = self._api_key
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return _safe_float(data.get(cg_id, {}).get("usd"))
        except Exception:
            return None
        finally:
            if not self._session:
                await session.close()

    def supports_pair(self, pair_base: str) -> bool:
        return pair_base in CRYPTO_COINGECKO_IDS


class YFinanceSource:
    """yfinance — stocks and commodities via Yahoo Finance."""
    name = "yfinance"

    async def fetch_latest_price(self, pair_base: str) -> float | None:
        ticker_symbol = STOCK_TICKERS.get(pair_base) or COMMODITY_YFINANCE.get(pair_base)
        if not ticker_symbol:
            return None
        try:
            import yfinance as yf
            ticker = yf.Ticker(ticker_symbol)
            info = await asyncio.to_thread(lambda: ticker.fast_info)
            price = getattr(info, "last_price", None)
            if price is None:
                price = getattr(info, "previous_close", None)
            return _safe_float(price) if price else None
        except Exception:
            return None

    def supports_pair(self, pair_base: str) -> bool:
        return pair_base in STOCK_TICKERS or pair_base in COMMODITY_YFINANCE


class AlphaVantageSource:
    """Alpha Vantage — free 25 req/day, REST API for stocks and commodities."""
    name = "alpha_vantage"
    _BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = "", session: aiohttp.ClientSession | None = None):
        self._api_key = api_key
        self._session = session
        self._daily_calls = 0
        self._daily_reset: float = 0.0
        self._daily_limit = 25

    def _check_rate_limit(self) -> bool:
        """[S7-07] Optimistic counting — increment before the request to avoid TOCTOU."""
        now = time.time()
        if now - self._daily_reset > 86400:
            self._daily_calls = 0
            self._daily_reset = now
        if self._daily_calls >= self._daily_limit:
            return False
        self._daily_calls += 1
        return True

    async def fetch_latest_price(self, pair_base: str) -> float | None:
        if not self._api_key:
            return None
        if not self._check_rate_limit():
            log.debug("Alpha Vantage daily limit reached")
            return None

        # Determine symbol and function
        stock_ticker = STOCK_TICKERS.get(pair_base)
        commodity_id = COMMODITY_ALPHAVANTAGE.get(pair_base)

        if stock_ticker:
            return await self._fetch_stock_price(stock_ticker)
        elif commodity_id:
            return await self._fetch_commodity_price(commodity_id, pair_base)
        return None

    async def _fetch_stock_price(self, symbol: str) -> float | None:
        session = self._session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self._api_key,
            }
            async with session.get(self._BASE_URL, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                quote = data.get("Global Quote", {})
                price_str = quote.get("05. price")
                return _safe_float(price_str) if price_str else None
        except Exception:
            return None
        finally:
            if not self._session:
                await session.close()

    async def _fetch_commodity_price(self, commodity_id: str, pair_base: str) -> float | None:
        # For ETF proxies (GLD, SLV), use GLOBAL_QUOTE
        if commodity_id in ("GLD", "SLV"):
            return await self._fetch_stock_price(commodity_id)
        # For BRENT, use commodity endpoint
        session = self._session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        try:
            params = {
                "function": "BRENT",
                "interval": "daily",
                "apikey": self._api_key,
            }
            async with session.get(self._BASE_URL, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                entries = data.get("data", [])
                if entries:
                    val = entries[0].get("value")
                    return _safe_float(val) if val and val != "." else None
                return None
        except Exception:
            return None
        finally:
            if not self._session:
                await session.close()

    def supports_pair(self, pair_base: str) -> bool:
        return pair_base in STOCK_TICKERS or pair_base in COMMODITY_ALPHAVANTAGE


# ── DataSourceManager ────────────────────────────────────────────────────

class DataSourceManager:
    """Orchestrates multi-source price fetching, cross-validation, and anomaly detection."""

    def __init__(
        self,
        config: DataSourcesConfig,
        alpha_vantage_key: str = "",
        coingecko_key: str = "",
        hl_client: HyperliquidNativeClient | None = None,
        # Deprecated: exchange param kept for backward compat (ignored)
        exchange: Any = None,
    ):
        self.config = config

        # Primary source — native Hyperliquid API (no ccxt)
        self._client = hl_client or HyperliquidNativeClient()
        self._owns_client = hl_client is None  # close if we created it
        self._primary = HyperliquidSource(self._client)

        # Secondary sources per asset class
        # [S7-03] Share single AlphaVantageSource instance to enforce 25/day limit
        _av_source = AlphaVantageSource(api_key=alpha_vantage_key)
        self._secondary: dict[str, list[DataSource]] = {
            "crypto": [DefiLlamaSource(), CoinGeckoSource(api_key=coingecko_key)],
            "stocks": [YFinanceSource(), _av_source],
            "commodities": [YFinanceSource(), _av_source],
        }

        # Rolling price history for anomaly detection (per pair, last 20 ticks)
        self._price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        # Source latency tracking (per source name, last 50 calls)
        self._latency_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    async def fetch_ohlcv(self, pair: str, timeframe: str, limit: int = 60) -> list[list]:
        """Fetch OHLCV from primary source (Hyperliquid native). Raises on failure."""
        candles = await self._primary.fetch_ohlcv(pair, timeframe, limit)
        if candles:
            return candles
        raise Exception(f"Primary OHLCV failed for {pair} {timeframe}")

    async def fetch_validated_price(self, pair: str) -> FetchResult:
        """Fetch price from all available sources, cross-validate, detect anomalies."""
        base = parse_pair(pair)[0]
        asset_class = classify_pair(base)

        # Fan out to all sources in parallel
        tasks = []

        # Primary: uses full pair
        tasks.append(self._timed_fetch_primary(pair))

        # Secondary: uses pair_base
        for source in self._secondary.get(asset_class, []):
            if source.supports_pair(base):
                tasks.append(self._timed_fetch_secondary(source, base))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        source_prices: dict[str, float] = {}
        source_latencies: dict[str, float] = {}

        for result in results:
            if isinstance(result, Exception):
                continue
            name, price, latency = result
            if price is not None:
                source_prices[name] = price
                source_latencies[name] = latency
                # Track latency
                self._latency_history[name].append(latency)

        # Cross-validate prices
        divergence_pct = 0.0
        price_valid = True
        if len(source_prices) >= 2:
            divergence_pct = self._compute_divergence(source_prices)
            price_valid = divergence_pct <= self.config.max_price_divergence_pct

        # Get primary price (or median if primary failed)
        latest_price = source_prices.get("hyperliquid")
        if latest_price is None and source_prices:
            latest_price = statistics.median(source_prices.values())

        # Anomaly detection (z-score)
        anomaly_flag = False
        if latest_price is not None:
            anomaly_flag = self._check_anomaly(pair, latest_price)

        # Log latency warnings
        self._check_latency_health(source_latencies)

        return FetchResult(
            latest_price=latest_price,
            source_prices=source_prices,
            divergence_pct=divergence_pct,
            price_valid=price_valid,
            source_latencies=source_latencies,
            anomaly_flag=anomaly_flag,
        )

    async def _timed_fetch_primary(self, pair: str) -> tuple[str, float | None, float]:
        t0 = time.monotonic()
        try:
            price = await self._primary.fetch_latest_price(pair)
            latency = (time.monotonic() - t0) * 1000
            return ("hyperliquid", price, latency)
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            log.debug(f"Primary price fetch failed: {e}")
            return ("hyperliquid", None, latency)

    async def _timed_fetch_secondary(
        self, source: DataSource, pair_base: str
    ) -> tuple[str, float | None, float]:
        t0 = time.monotonic()
        try:
            price = await source.fetch_latest_price(pair_base)
            latency = (time.monotonic() - t0) * 1000
            return (source.name, price, latency)
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            log.debug(f"Source {source.name} failed for {pair_base}: {e}")
            return (source.name, None, latency)

    @staticmethod
    def _compute_divergence(prices: dict[str, float]) -> float:
        """Compute max divergence from median as a percentage."""
        if len(prices) < 2:
            return 0.0
        values = list(prices.values())
        median = statistics.median(values)
        if median == 0:
            return 0.0
        max_div = max(abs(v - median) / median * 100 for v in values)
        return max_div

    def _check_anomaly(self, pair: str, price: float) -> bool:
        """Rolling z-score anomaly detection. Returns True if price is >Nσ from rolling mean."""
        history = self._price_history[pair]

        if len(history) < 5:
            history.append(price)
            return False

        mean = statistics.mean(history)
        std = statistics.stdev(history) if len(history) > 1 else 0.0

        is_anomaly = False
        if std > 0:
            z_score = abs(price - mean) / std
            if z_score > self.config.anomaly_zscore_threshold:
                log.warning(
                    f"ANOMALY: {pair} price {price:.2f} z-score={z_score:.1f} "
                    f"(mean={mean:.2f}, std={std:.2f})"
                )
                is_anomaly = True

        history.append(price)
        return is_anomaly

    def _check_latency_health(self, latencies: dict[str, float]) -> None:
        """Log warnings for slow sources."""
        for source_name, latency_ms in latencies.items():
            if latency_ms > 2000:
                log.warning(f"Source {source_name} slow: {latency_ms:.0f}ms")

    def get_source_health(self) -> dict[str, dict]:
        """Get latency stats for all tracked sources."""
        health = {}
        for name, latencies in self._latency_history.items():
            if latencies:
                health[name] = {
                    "avg_latency_ms": statistics.mean(latencies),
                    "max_latency_ms": max(latencies),
                    "calls": len(latencies),
                }
        return health

    async def close(self):
        """Cleanup resources."""
        if self._owns_client:
            await self._client.close()
