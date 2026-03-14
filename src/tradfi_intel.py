"""TradFi intelligence layer — earnings awareness, macro data, IV, correlations.

Provides institutional-grade context for synthetic stock/commodity trading:
  - EarningsCalendar: block trades around earnings releases
  - OptionsIntel: implied volatility context for position sizing
  - FREDClient: macro indicators (Treasury yields, CPI, DXY, fed funds)
  - CorrelationMatrix: rolling cross-asset correlations
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import math

import aiohttp
import pandas as pd

from src.data_sources import STOCK_TICKERS, COMMODITY_YFINANCE, classify_pair
from src.exchange import parse_pair
from src.utils import log


def _safe_float(v, default: float = 0.0) -> float:
    """[S7-02] Convert to float, returning default if result is inf/nan."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ── EarningsCalendar ─────────────────────────────────────────────────────

class EarningsCalendar:
    """Track upcoming earnings dates for synthetic stock pairs.

    Exposes is_earnings_blackout() — hard gate for RiskManager.
    """

    def __init__(self, hours_before: int = 24, hours_after: int = 2):
        self.hours_before = hours_before
        self.hours_after = hours_after
        self._cache: dict[str, datetime | None] = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 3600.0  # Refresh hourly

    async def refresh(self) -> None:
        """Refresh earnings dates from yfinance."""
        import yfinance as yf
        for base, ticker_symbol in STOCK_TICKERS.items():
            try:
                ticker = yf.Ticker(ticker_symbol)
                cal = await asyncio.to_thread(lambda t=ticker: t.calendar)
                if cal is not None and isinstance(cal, dict):
                    earnings_dates = cal.get("Earnings Date", [])
                    if earnings_dates:
                        ed = earnings_dates[0]
                        if isinstance(ed, pd.Timestamp):
                            self._cache[base] = ed.to_pydatetime().replace(tzinfo=timezone.utc)
                        else:
                            self._cache[base] = None
                    else:
                        self._cache[base] = None
                elif cal is not None and isinstance(cal, pd.DataFrame):
                    if "Earnings Date" in cal.columns and len(cal) > 0:
                        ed = cal["Earnings Date"].iloc[0]
                        if isinstance(ed, pd.Timestamp):
                            self._cache[base] = ed.to_pydatetime().replace(tzinfo=timezone.utc)
                        else:
                            self._cache[base] = None
                    else:
                        self._cache[base] = None
                else:
                    self._cache[base] = None
            except Exception as e:
                log.debug(f"Earnings fetch failed for {ticker_symbol}: {e}")
                self._cache[base] = None
        self._cache_time = time.time()

    async def ensure_fresh(self) -> None:
        """Refresh cache if stale."""
        if time.time() - self._cache_time > self._cache_ttl:
            await self.refresh()

    def is_earnings_blackout(self, pair: str) -> bool:
        """Check if a pair is in an earnings blackout window."""
        try:
            base = parse_pair(pair)[0]
        except ValueError:
            return False

        if classify_pair(base) != "stocks":
            return False

        earnings_date = self._cache.get(base)
        if earnings_date is None:
            return False

        now = datetime.now(timezone.utc)
        blackout_start = earnings_date - timedelta(hours=self.hours_before)
        blackout_end = earnings_date + timedelta(hours=self.hours_after)
        return blackout_start <= now <= blackout_end

    def get_upcoming_earnings(self) -> dict[str, str | None]:
        """Get cached earnings dates for display."""
        return {
            base: dt.isoformat() if dt else None
            for base, dt in self._cache.items()
        }


# ── OptionsIntel ─────────────────────────────────────────────────────────

@dataclass
class IVContext:
    """Implied volatility context for a stock pair."""
    avg_iv: float = 0.0
    put_call_ratio: float = 1.0
    calls_count: int = 0
    puts_count: int = 0

    def to_dict(self) -> dict:
        return {
            "avg_iv": round(self.avg_iv, 4),
            "put_call_ratio": round(self.put_call_ratio, 2),
            "calls_count": self.calls_count,
            "puts_count": self.puts_count,
        }


class OptionsIntel:
    """Extract implied volatility context from stock options chains."""

    def __init__(self):
        self._cache: dict[str, IVContext] = {}
        # [S7-08] Per-ticker cache timestamps instead of shared single timestamp
        self._cache_times: dict[str, float] = {}
        self._cache_ttl: float = 1800.0  # 30 min

    async def get_iv_context(self, pair: str) -> IVContext | None:
        """Get IV context for a stock pair. Returns None for non-stock pairs."""
        try:
            base = parse_pair(pair)[0]
        except ValueError:
            return None

        ticker_symbol = STOCK_TICKERS.get(base)
        if not ticker_symbol:
            return None

        # Check cache (per-ticker timestamps)
        cache_time = self._cache_times.get(base, 0.0)
        if base in self._cache and time.time() - cache_time < self._cache_ttl:
            return self._cache[base]

        try:
            import yfinance as yf
            ticker = yf.Ticker(ticker_symbol)
            expirations = await asyncio.to_thread(lambda: ticker.options)
            if not expirations:
                return None

            chain = await asyncio.to_thread(lambda: ticker.option_chain(expirations[0]))
            calls_iv = chain.calls["impliedVolatility"].dropna()
            puts_iv = chain.puts["impliedVolatility"].dropna()

            # [S7-02, S7-06] Safe float conversion + clamp to [0, 5.0]
            avg_iv = 0.0
            if len(calls_iv) > 0 and len(puts_iv) > 0:
                avg_iv = _safe_float((calls_iv.mean() + puts_iv.mean()) / 2)
            elif len(calls_iv) > 0:
                avg_iv = _safe_float(calls_iv.mean())
            elif len(puts_iv) > 0:
                avg_iv = _safe_float(puts_iv.mean())
            avg_iv = max(0.0, min(avg_iv, 5.0))

            put_call_ratio = len(chain.puts) / max(len(chain.calls), 1)

            ctx = IVContext(
                avg_iv=avg_iv,
                put_call_ratio=put_call_ratio,
                calls_count=len(chain.calls),
                puts_count=len(chain.puts),
            )
            self._cache[base] = ctx
            self._cache_times[base] = time.time()
            return ctx

        except Exception as e:
            log.debug(f"Options data unavailable for {ticker_symbol}: {e}")
            return None

    async def get_all_iv_context(self, pairs: list[str]) -> dict[str, dict]:
        """Get IV context for all stock pairs."""
        result = {}
        for pair in pairs:
            try:
                base = parse_pair(pair)[0]
            except ValueError:
                continue
            if base not in STOCK_TICKERS:
                continue
            ctx = await self.get_iv_context(pair)
            if ctx:
                result[pair] = ctx.to_dict()
        return result


# ── FREDClient ───────────────────────────────────────────────────────────

FRED_SERIES: dict[str, str] = {
    "treasury_2y": "DGS2",
    "treasury_10y": "DGS10",
    "fed_funds_rate": "FEDFUNDS",
    "cpi_yoy": "CPIAUCSL",
    "ppi_yoy": "PPIACO",
    "dxy": "DTWEXBGS",
}


class FREDClient:
    """Federal Reserve Economic Data — macro indicators."""
    _BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._cache: dict[str, dict] = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 3600.0  # 1 hour

    async def get_macro_snapshot(self) -> dict[str, dict]:
        """Fetch key macro indicators. Returns {name: {value, date}}."""
        if not self._api_key:
            return {}

        if self._cache and time.time() - self._cache_time < self._cache_ttl:
            return self._cache

        snapshot: dict[str, dict] = {}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            tasks = [
                self._fetch_series(session, name, series_id)
                for name, series_id in FRED_SERIES.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, tuple):
                    name, value, date = result
                    if value is not None:
                        snapshot[name] = {"value": value, "date": date}

        if snapshot:
            self._cache = snapshot
            self._cache_time = time.time()
        return snapshot

    async def _fetch_series(
        self, session: aiohttp.ClientSession, name: str, series_id: str
    ) -> tuple[str, float | None, str | None]:
        try:
            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": "1",
            }
            async with session.get(self._BASE_URL, params=params) as resp:
                if resp.status != 200:
                    return (name, None, None)
                data = await resp.json()
                obs = data.get("observations", [])
                if obs and obs[0].get("value", ".") != ".":
                    val = _safe_float(obs[0]["value"])
                    return (name, val if val != 0.0 else None, obs[0].get("date"))
                return (name, None, None)
        except Exception as e:
            log.debug(f"FRED fetch failed for {series_id}: {e}")
            return (name, None, None)

    def get_yield_spread(self) -> float | None:
        """Compute 10Y-2Y yield spread from cached data (recession indicator)."""
        t10 = self._cache.get("treasury_10y", {}).get("value")
        t2 = self._cache.get("treasury_2y", {}).get("value")
        if t10 is not None and t2 is not None:
            return t10 - t2
        return None


# ── CorrelationMatrix ────────────────────────────────────────────────────

class CorrelationMatrix:
    """Rolling cross-asset correlation for portfolio risk management."""

    def __init__(self, threshold: float = 0.85, window: int = 20):
        self.threshold = threshold
        self.window = window
        self._last_matrix: dict[str, dict[str, float]] = {}

    def compute(self, ohlcv_data: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
        """Compute correlation matrix from OHLCV close prices.

        Args:
            ohlcv_data: {pair: DataFrame with 'close' column}

        Returns:
            {pair: {other_pair: correlation}}
        """
        closes: dict[str, pd.Series] = {}
        for pair, df in ohlcv_data.items():
            if df is not None and not df.empty and len(df) >= self.window:
                closes[pair] = df["close"].tail(self.window).reset_index(drop=True)

        if len(closes) < 2:
            self._last_matrix = {}
            return {}

        df = pd.DataFrame(closes)
        corr = df.corr()
        self._last_matrix = corr.to_dict()
        return self._last_matrix

    def get_correlated_pairs(
        self, pair: str, ohlcv_data: dict[str, pd.DataFrame] | None = None
    ) -> list[str]:
        """Get pairs with correlation above threshold to the given pair."""
        matrix = self._last_matrix
        if ohlcv_data is not None:
            matrix = self.compute(ohlcv_data)

        if pair not in matrix:
            return []

        return [
            other
            for other, corr_val in matrix[pair].items()
            if other != pair and abs(corr_val) > self.threshold
        ]

    def get_correlation_warnings(
        self, open_pairs: list[str]
    ) -> list[dict[str, str | float]]:
        """Get warnings about highly correlated open positions."""
        warnings = []
        seen = set()
        for pair in open_pairs:
            if pair not in self._last_matrix:
                continue
            for other in open_pairs:
                if other == pair or (pair, other) in seen or (other, pair) in seen:
                    continue
                corr = self._last_matrix.get(pair, {}).get(other, 0.0)
                if abs(corr) > self.threshold:
                    warnings.append({
                        "pair_a": pair,
                        "pair_b": other,
                        "correlation": round(corr, 3),
                    })
                    seen.add((pair, other))
        return warnings


# ── TradFi Intel Aggregator ──────────────────────────────────────────────

class TradFiIntel:
    """Aggregates all TradFi intelligence sources."""

    def __init__(
        self,
        fred_api_key: str = "",
        earnings_hours_before: int = 24,
        earnings_hours_after: int = 2,
        correlation_threshold: float = 0.85,
    ):
        self.earnings = EarningsCalendar(earnings_hours_before, earnings_hours_after)
        self.options = OptionsIntel()
        self.fred = FREDClient(api_key=fred_api_key)
        self.correlations = CorrelationMatrix(threshold=correlation_threshold)

    async def refresh(self) -> None:
        """Refresh all cached data."""
        tasks = [
            self.earnings.ensure_fresh(),
            self.fred.get_macro_snapshot(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_full_context(
        self,
        pairs: list[str],
        ohlcv_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict:
        """Get full TradFi intelligence context for brain enrichment."""
        await self.refresh()

        context: dict = {}

        # Earnings
        upcoming = self.earnings.get_upcoming_earnings()
        if upcoming:
            context["upcoming_earnings"] = upcoming
            blackouts = [p for p in pairs if self.earnings.is_earnings_blackout(p)]
            if blackouts:
                context["earnings_blackouts"] = blackouts

        # FRED macro
        macro = await self.fred.get_macro_snapshot()
        if macro:
            context["macro_indicators"] = macro
            spread = self.fred.get_yield_spread()
            if spread is not None:
                context["yield_spread_10y_2y"] = round(spread, 2)

        # IV context (stock pairs only)
        iv_data = await self.options.get_all_iv_context(pairs)
        if iv_data:
            context["options_iv"] = iv_data

        # Correlations
        if ohlcv_data:
            self.correlations.compute(ohlcv_data)
            open_pair_list = [p for p in pairs if p in ohlcv_data]
            warnings = self.correlations.get_correlation_warnings(open_pair_list)
            if warnings:
                context["correlation_warnings"] = warnings

        return context
