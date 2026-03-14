"""Market data pipeline — fetches OHLCV via Hyperliquid native API, caches in SQLite, returns DataFrames.

Uses multi-source cross-validation: primary (Hyperliquid native REST) for OHLCV,
secondary sources for price sanity checks.

In-memory candle cache: skips API calls when no new candle has closed since
last fetch. On a 5-min cycle this cuts API calls from ~42 to ~8 per cycle.
"""

from __future__ import annotations

import asyncio
import time

import pandas as pd

from src.config import AgentConfig, DataSourcesConfig
from src.data_sources import (
    DataSourceManager,
    FetchResult,
    HyperliquidNativeClient,
    _pair_to_hl_coin,
)
from src.database import Database
from src.utils import log

# Timeframe durations in milliseconds
_TF_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class DataFetcher:
    def __init__(
        self,
        exchange,  # kept for backward compat (executor still uses it)
        db: Database,
        config: AgentConfig,
        data_source_config: DataSourcesConfig | None = None,
        alpha_vantage_key: str = "",
        coingecko_key: str = "",
        hl_client: HyperliquidNativeClient | None = None,
    ):
        self.exchange = exchange
        self.db = db
        self.config = config
        self._latest_prices: dict[str, float] = {}
        self._source_health: dict[str, dict] = {}

        # In-memory OHLCV cache: avoids API calls when no new candle has closed
        self._candle_cache: dict[tuple[str, str], list] = {}  # (pair, tf) -> raw candles
        self._df_cache: dict[tuple[str, str], pd.DataFrame] = {}  # (pair, tf) -> DataFrame
        self._cache_stats = {"hits": 0, "misses": 0, "incremental": 0}

        # Native Hyperliquid client — shared with Exchange for price fetching
        self._hl_client = hl_client or HyperliquidNativeClient()

        # Build DataSourceManager with native client
        ds_config = data_source_config or DataSourcesConfig()
        self._source_manager = DataSourceManager(
            config=ds_config,
            alpha_vantage_key=alpha_vantage_key,
            coingecko_key=coingecko_key,
            hl_client=self._hl_client,
        )

    async def fetch_all_pairs(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetch OHLCV for all configured pairs and timeframes.

        Fetches pairs in parallel (up to 20 concurrent) — native API has no
        ccxt rate limiter bottleneck, just the 1200 weight/min budget.
        Uses in-memory candle cache to skip API calls for unchanged timeframes.
        Returns: {pair: {timeframe: DataFrame}}
        """
        result: dict[str, dict[str, pd.DataFrame]] = {}
        # Higher concurrency is safe with native API (no ccxt throttler)
        sem = asyncio.Semaphore(20)

        # Reset per-cycle stats
        before = self._cache_stats.copy()

        # Crypto pairs get all timeframes; synthetics skip 5m (less depth, noisier)
        _CRYPTO_PREFIXES = ("BTC/", "ETH/", "AAVE/", "SOL/", "LINK/", "DOGE/")
        # Only validate pairs that have secondary sources (crypto, known stocks, commodities)
        _VALIDATE_PREFIXES = _CRYPTO_PREFIXES + ("XYZ-AAPL/", "XYZ-TSLA/", "XYZ-NVDA/", "XYZ-MSFT/", "XYZ-TSM/", "XYZ-GOLD/", "XYZ-BRENTOIL/", "XYZ-SILVER/")

        async def _fetch_pair_all_tf(pair: str) -> tuple[str, dict[str, pd.DataFrame]]:
            pair_data: dict[str, pd.DataFrame] = {}
            is_crypto = pair.startswith(_CRYPTO_PREFIXES)
            timeframes = self.config.timeframes if is_crypto else [
                tf for tf in self.config.timeframes if tf != "5m"
            ]
            async with sem:
                for tf in timeframes:
                    try:
                        df = await self.fetch_pair(pair, tf)
                        pair_data[tf] = df

                        if not df.empty and (pair not in self._latest_prices or tf == self.config.timeframes[0]):
                            self._latest_prices[pair] = float(df.iloc[-1]["close"])
                    except Exception as e:
                        log.warning(f"Failed to fetch {pair} {tf}: {e}")

            # Cross-validate price only for pairs with secondary sources
            if pair.startswith(_VALIDATE_PREFIXES):
                await self._validate_pair_price(pair)
            return pair, pair_data

        tasks = [_fetch_pair_all_tf(pair) for pair in self.config.pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                log.warning(f"Pair fetch failed: {r}")
                continue
            pair, pair_data = r
            result[pair] = pair_data

        # Log cache efficiency
        hits = self._cache_stats["hits"] - before["hits"]
        misses = self._cache_stats["misses"] - before["misses"]
        incremental = self._cache_stats["incremental"] - before["incremental"]
        total = hits + misses + incremental
        if total > 0:
            log.info(
                f"OHLCV cache: {hits} hits, {incremental} incremental, "
                f"{misses} full fetches (saved {hits}/{total} API calls)"
            )

        return result

    async def _validate_pair_price(self, pair: str) -> None:
        """Run multi-source price validation for a pair."""
        try:
            fetch_result = await self._source_manager.fetch_validated_price(pair)
            if fetch_result.source_prices:
                self._source_health[pair] = {
                    "divergence_pct": fetch_result.divergence_pct,
                    "price_valid": fetch_result.price_valid,
                    "anomaly_flag": fetch_result.anomaly_flag,
                    "source_latencies": fetch_result.source_latencies,
                    "source_count": len(fetch_result.source_prices),
                    "sources": list(fetch_result.source_prices.keys()),
                }
                if not fetch_result.price_valid:
                    log.warning(
                        f"Price validation FAILED for {pair}: "
                        f"divergence={fetch_result.divergence_pct:.2f}% "
                        f"(sources: {fetch_result.source_prices})"
                    )
                if fetch_result.anomaly_flag:
                    log.warning(f"ANOMALY detected for {pair}")
        except Exception as e:
            log.debug(f"Price validation skipped for {pair}: {e}")

    async def fetch_pair(self, pair: str, timeframe: str, limit: int = 60) -> pd.DataFrame:
        """Fetch OHLCV for a single pair+timeframe from primary source.

        Uses in-memory cache: if no new candle has closed since last fetch,
        returns cached DataFrame without making an API call.
        """
        cache_key = (pair, timeframe)
        tf_ms = _TF_MS.get(timeframe, 3_600_000)

        # Check in-memory cache
        if cache_key in self._candle_cache:
            cached = self._candle_cache[cache_key]
            if cached:
                last_ts = cached[-1][0]
                now_ms = int(time.time() * 1000)
                elapsed = now_ms - last_ts

                if elapsed < tf_ms:
                    # No new complete candle — return cached DataFrame
                    self._cache_stats["hits"] += 1
                    return self._df_cache[cache_key]

                # New candle(s) expected — fetch only what's needed
                new_count = min(int(elapsed / tf_ms) + 2, limit)
                try:
                    new_candles = await self._source_manager.fetch_ohlcv(
                        pair, timeframe, new_count
                    )
                    if new_candles:
                        merged = self._merge_candles(cached, new_candles, limit)
                        self._candle_cache[cache_key] = merged
                        df = self._candles_to_df(merged)
                        self._df_cache[cache_key] = df
                        self.db.cache_ohlcv(pair, timeframe, new_candles)
                        self._cache_stats["incremental"] += 1
                        return df
                except Exception as e:
                    log.warning(
                        f"Incremental fetch failed for {pair} {timeframe}, "
                        f"using cache: {e}"
                    )
                    self._cache_stats["hits"] += 1
                    return self._df_cache[cache_key]

        # Cold cache — full fetch
        self._cache_stats["misses"] += 1
        candles = await self._source_manager.fetch_ohlcv(pair, timeframe, limit)

        if not candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        self.db.cache_ohlcv(pair, timeframe, candles)
        self._candle_cache[cache_key] = candles
        df = self._candles_to_df(candles)
        self._df_cache[cache_key] = df
        return df

    @staticmethod
    def _merge_candles(cached: list, new: list, max_keep: int = 60) -> list:
        """Merge new candles into cached list, deduplicating by timestamp."""
        by_ts = {c[0]: c for c in cached}
        for c in new:
            by_ts[c[0]] = c  # New data overwrites (updates in-progress candle)
        merged = sorted(by_ts.values(), key=lambda c: c[0])
        return merged[-max_keep:]

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        """Convert raw candle list to a validated DataFrame."""
        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        df = df[df["high"] >= df["low"]]
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        return df

    def get_latest_prices(self) -> dict[str, float]:
        return self._latest_prices.copy()

    def get_source_health(self) -> dict[str, dict]:
        """Get per-pair source health + global source latency stats."""
        return {
            "per_pair": self._source_health.copy(),
            "sources": self._source_manager.get_source_health(),
        }

    def get_invalid_pairs(self) -> set[str]:
        """[S7-01] Return pairs where price validation failed or anomaly detected."""
        invalid = set()
        for pair, health in self._source_health.items():
            if not health.get("price_valid", True) or health.get("anomaly_flag", False):
                invalid.add(pair)
        return invalid

    async def get_single_price(self, pair: str) -> float | None:
        """Fetch the latest price for a single pair via native allMids."""
        try:
            price = await self._source_manager._primary.fetch_latest_price(pair)
            if price:
                self._latest_prices[pair] = price
            return price
        except Exception as e:
            log.warning(f"Failed to fetch price for {pair}: {e}")
            return self._latest_prices.get(pair)

    async def close(self):
        """Close data source connections."""
        await self._source_manager.close()
        await self._hl_client.close()
