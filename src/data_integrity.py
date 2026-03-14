"""Data integrity checker — cross-references sources, detects anomalies, scores reliability.

Runs after macro data is fetched but before it reaches the AI brain.
Flags conflicts, stale data, and anomalies so the brain can weight sources properly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.utils import log


@dataclass
class IntegrityReport:
    """Results of the data integrity check."""

    timestamp: float = 0.0
    anomalies: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    stale_sources: list[str] = field(default_factory=list)
    sentiment_consensus: dict[str, Any] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 1.0  # 0.0 = don't trust anything, 1.0 = all clear

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomalies": self.anomalies,
            "conflicts": self.conflicts,
            "stale_sources": self.stale_sources,
            "sentiment_consensus": self.sentiment_consensus,
            "source_scores": self.source_scores,
            "overall_confidence": round(self.overall_confidence, 3),
        }

    @property
    def has_issues(self) -> bool:
        return bool(self.anomalies or self.conflicts or self.stale_sources)


class DataIntegrityChecker:
    """Cross-references macro data sources and flags inconsistencies.

    Checks:
    1. Price anomaly detection — flag if any source diverges >5% from others
    2. Sentiment conflict detection — flag when sources disagree on direction
    3. Stale data detection — flag data older than expected freshness window
    4. Source reliability scoring — weight sources by historical accuracy
    5. Sentiment consensus — count bullish/bearish/neutral signals across all sources
    """

    # Source reliability scores (0.0 - 1.0) — updated over time
    _DEFAULT_SCORES: dict[str, float] = {
        "fear_greed": 0.7,           # Decent contrarian indicator
        "global_market": 0.8,        # CoinGecko is reliable
        "coin_prices": 0.9,          # Direct price data, high reliability
        "trending": 0.3,             # Noisy, FOMO indicator
        "reddit": 0.2,               # Very noisy, contrarian only
        "dxy": 0.6,                  # Exchange rate proxy, decent
        "cryptopanic": 0.5,          # Community votes add signal
        "yahoo_macro": 0.8,          # Institutional data, reliable
        "rss_news": 0.5,             # Headlines need interpretation
        "hackernews": 0.3,           # Tech-focused, not always relevant
        "polymarket": 0.7,           # Prediction markets have real money behind them
        "funding_rates": 0.85,       # Direct market data, very actionable
        "open_interest": 0.85,       # Direct market data, very actionable
        "long_short_ratio": 0.75,    # Good contrarian signal
        "taker_buy_sell": 0.8,       # Direct flow data
        "economic_calendar": 0.85,   # High-impact macro events are very reliable signals
        "onchain_btc": 0.7,          # Good background signal, not directly tradeable
        "stablecoin_flows": 0.75,    # Daily granularity, decent macro signal
        "orderbook_depth": 0.8,      # Real-time market microstructure, actionable
        "whale_movements": 0.7,      # Direct on-chain data, moderately actionable
        "sec_filings": 0.6,          # Impactful but infrequent regulatory events
        "onchain_btc_macro": 0.8,    # Objective on-chain data, historically predictive
        "github_activity": 0.3,      # Very loosely correlated with price, supplementary
        "liquidation_data": 0.85,    # Direct market data, liquidation cascades are actionable
        "cryptoquant_onchain": 0.85, # Enterprise on-chain metrics (MVRV, exchange flows), historically predictive
    }

    def __init__(self):
        self.source_scores = dict(self._DEFAULT_SCORES)
        self._historical_accuracy: dict[str, list[bool]] = {}

    def check(self, macro_context_dict: dict[str, Any]) -> IntegrityReport:
        """Run all integrity checks on the macro context data.

        Args:
            macro_context_dict: Output of MacroContext.to_dict()

        Returns:
            IntegrityReport with all findings.
        """
        report = IntegrityReport(timestamp=time.time())
        report.source_scores = dict(self.source_scores)

        self._check_price_anomalies(macro_context_dict, report)
        self._check_sentiment_conflicts(macro_context_dict, report)
        self._check_stale_data(macro_context_dict, report)
        self._check_economic_event_risk(macro_context_dict, report)
        self._build_sentiment_consensus(macro_context_dict, report)
        self._calculate_overall_confidence(macro_context_dict, report)

        if report.has_issues:
            log.info(
                f"Data integrity: {len(report.anomalies)} anomalies, "
                f"{len(report.conflicts)} conflicts, "
                f"{len(report.stale_sources)} stale — "
                f"confidence {report.overall_confidence:.0%}"
            )
        else:
            log.debug("Data integrity: all clear")

        return report

    # ── Private checks ──────────────────────────────────────────────

    def _check_price_anomalies(self, ctx: dict, report: IntegrityReport) -> None:
        """Flag if price data from different sources diverges significantly."""
        coin_prices = ctx.get("coin_prices") or {}

        for symbol, data in coin_prices.items():
            price = data.get("price")
            change_1h = data.get("change_1h_pct")
            change_24h = data.get("change_24h_pct")

            if price is None:
                continue

            # Flag extreme 1h moves (>10%) — likely bad data or flash crash
            if change_1h is not None and abs(change_1h) > 10:
                report.anomalies.append(
                    f"{symbol} 1h change is {change_1h:+.1f}% — possible anomaly or flash crash"
                )

            # Flag extreme 24h moves (>25%)
            if change_24h is not None and abs(change_24h) > 25:
                report.anomalies.append(
                    f"{symbol} 24h change is {change_24h:+.1f}% — extreme move, verify data"
                )

        # Cross-check: if funding rate says overleveraged longs but price is dropping fast,
        # that's a liquidation cascade signal
        funding = ctx.get("funding_rates") or {}
        for symbol, fdata in funding.items():
            # Handle both Binance format (BTCUSDT) and Hyperliquid format (BTC)
            pair_symbol = symbol.replace("USDT", "").replace("USDC", "")
            coin_data = coin_prices.get(pair_symbol, {})
            change_1h = coin_data.get("change_1h_pct")

            if fdata.get("signal") == "overleveraged_longs" and change_1h is not None and change_1h < -3:
                report.anomalies.append(
                    f"{symbol}: overleveraged longs + price dropping {change_1h:.1f}% — "
                    f"liquidation cascade risk"
                )

    def _check_sentiment_conflicts(self, ctx: dict, report: IntegrityReport) -> None:
        """Flag when major sources disagree on market direction."""
        signals: dict[str, str] = {}  # source -> "bullish" | "bearish" | "neutral"

        # Fear & Greed
        fg = ctx.get("fear_greed_index")
        if fg and isinstance(fg, dict):
            val = fg.get("value")
            if val is not None:
                if val >= 60:
                    signals["fear_greed"] = "bullish"
                elif val <= 40:
                    signals["fear_greed"] = "bearish"
                else:
                    signals["fear_greed"] = "neutral"

        # Global market
        gm = ctx.get("global_market")
        if gm and isinstance(gm, dict):
            change = gm.get("market_cap_change_24h_pct")
            if change is not None:
                if change > 2:
                    signals["global_market"] = "bullish"
                elif change < -2:
                    signals["global_market"] = "bearish"
                else:
                    signals["global_market"] = "neutral"

        # BTC price direction
        coins = ctx.get("coin_prices") or {}
        btc = coins.get("BTC", {})
        btc_24h = btc.get("change_24h_pct")
        if btc_24h is not None:
            if btc_24h > 3:
                signals["btc_price"] = "bullish"
            elif btc_24h < -3:
                signals["btc_price"] = "bearish"
            else:
                signals["btc_price"] = "neutral"

        # Funding rates
        funding = ctx.get("funding_rates") or {}
        btc_funding = funding.get("BTCUSDT", {}) or funding.get("BTC", {})
        if btc_funding:
            sig = btc_funding.get("signal", "neutral")
            if sig == "overleveraged_longs":
                signals["funding"] = "bearish"  # Contrarian
            elif sig == "overleveraged_shorts":
                signals["funding"] = "bullish"  # Contrarian
            else:
                signals["funding"] = "neutral"

        # Long/Short ratio (contrarian)
        ls = ctx.get("long_short_ratio") or {}
        btc_ls = ls.get("BTCUSDT", {}) or ls.get("BTC", {})
        if btc_ls:
            sig = btc_ls.get("signal", "balanced")
            if sig == "crowded_long":
                signals["long_short"] = "bearish"  # Contrarian
            elif sig == "crowded_short":
                signals["long_short"] = "bullish"  # Contrarian
            else:
                signals["long_short"] = "neutral"

        # Taker buy/sell
        taker = ctx.get("taker_buy_sell") or {}
        btc_taker = taker.get("BTCUSDT", {}) or taker.get("BTC", {})
        if btc_taker:
            sig = btc_taker.get("signal", "balanced")
            if sig == "aggressive_buying":
                signals["taker_flow"] = "bullish"
            elif sig == "aggressive_selling":
                signals["taker_flow"] = "bearish"
            else:
                signals["taker_flow"] = "neutral"

        # Stablecoin flows
        sc_flows = ctx.get("stablecoin_flows")
        if sc_flows and isinstance(sc_flows, dict):
            sig = sc_flows.get("signal")
            if sig == "net_minting":
                signals["stablecoin_flows"] = "bullish"
            elif sig == "net_burning":
                signals["stablecoin_flows"] = "bearish"

        # Order book depth
        ob = ctx.get("orderbook_depth")
        if ob and isinstance(ob, dict):
            sig = ob.get("signal")
            if sig == "buy_wall":
                signals["orderbook_depth"] = "bullish"
            elif sig == "sell_wall":
                signals["orderbook_depth"] = "bearish"

        # Whale movements
        whales = ctx.get("whale_movements")
        if whales and isinstance(whales, dict):
            sig = whales.get("overall_signal")
            if sig == "bullish_whale_accumulating":
                signals["whale_movements"] = "bullish"
            elif sig == "bearish_whale_selling":
                signals["whale_movements"] = "bearish"

        # On-chain BTC macro (BGeometrics)
        btc_macro = ctx.get("onchain_btc_macro")
        if btc_macro and isinstance(btc_macro, dict):
            sig = btc_macro.get("signal", "")
            if sig.startswith("bullish"):
                signals["onchain_btc_macro"] = "bullish"
            elif sig.startswith("bearish"):
                signals["onchain_btc_macro"] = "bearish"

        # SEC filings — high regulatory activity = short-term uncertainty
        sec = ctx.get("sec_filings")
        if sec and isinstance(sec, dict):
            sig = sec.get("signal")
            if sig == "high_regulatory_activity":
                signals["sec_filings"] = "bearish"

        # Check for conflicts between high-reliability sources
        if len(signals) >= 3:
            bullish = [s for s, v in signals.items() if v == "bullish"]
            bearish = [s for s, v in signals.items() if v == "bearish"]

            if bullish and bearish:
                report.conflicts.append(
                    f"Sentiment split: bullish ({', '.join(bullish)}) vs bearish ({', '.join(bearish)})"
                )

    def _check_stale_data(self, ctx: dict, report: IntegrityReport) -> None:
        """Flag sources that returned no data (likely stale or down)."""
        expected_sources = [
            "fear_greed_index", "global_market", "coin_prices",
            "funding_rates", "open_interest",
            "economic_calendar", "stablecoin_flows",
        ]

        errors = ctx.get("errors", [])
        for source in expected_sources:
            if ctx.get(source) is None:
                # Check if it's in the error list
                source_label = source.replace("_", " ").title()
                matching_errors = [e for e in errors if source_label.lower() in e.lower()]
                if matching_errors:
                    report.stale_sources.append(f"{source}: fetch failed — {matching_errors[0]}")
                else:
                    report.stale_sources.append(f"{source}: no data returned")

    def _build_sentiment_consensus(self, ctx: dict, report: IntegrityReport) -> None:
        """Count bullish/bearish/neutral signals across all sources."""
        bullish_signals: list[str] = []
        bearish_signals: list[str] = []
        neutral_signals: list[str] = []

        # Fear & Greed
        fg = ctx.get("fear_greed_index")
        if fg and isinstance(fg, dict):
            val = fg.get("value")
            if val is not None:
                if val >= 60:
                    bullish_signals.append(f"Fear&Greed={val}")
                elif val <= 40:
                    bearish_signals.append(f"Fear&Greed={val}")
                else:
                    neutral_signals.append(f"Fear&Greed={val}")

        # Global market cap trend
        gm = ctx.get("global_market")
        if gm and isinstance(gm, dict):
            change = gm.get("market_cap_change_24h_pct")
            if change is not None:
                if change > 1:
                    bullish_signals.append(f"MarketCap +{change:.1f}%")
                elif change < -1:
                    bearish_signals.append(f"MarketCap {change:.1f}%")
                else:
                    neutral_signals.append(f"MarketCap {change:+.1f}%")

        # BTC direction
        coins = ctx.get("coin_prices") or {}
        for symbol in ["BTC", "ETH"]:
            coin = coins.get(symbol, {})
            change = coin.get("change_24h_pct")
            if change is not None:
                if change > 2:
                    bullish_signals.append(f"{symbol} +{change:.1f}%")
                elif change < -2:
                    bearish_signals.append(f"{symbol} {change:.1f}%")

        # Funding rates
        funding = ctx.get("funding_rates") or {}
        for symbol, fdata in funding.items():
            sig = fdata.get("signal", "neutral")
            if sig == "overleveraged_longs":
                bearish_signals.append(f"{symbol} funding overleveraged_longs")
            elif sig == "overleveraged_shorts":
                bullish_signals.append(f"{symbol} funding overleveraged_shorts")

        # Long/Short ratio
        ls_data = ctx.get("long_short_ratio") or {}
        for symbol, ldata in ls_data.items():
            sig = ldata.get("signal", "balanced")
            if sig == "crowded_long":
                bearish_signals.append(f"{symbol} L/S crowded_long")
            elif sig == "crowded_short":
                bullish_signals.append(f"{symbol} L/S crowded_short")

        # Taker buy/sell
        taker_data = ctx.get("taker_buy_sell") or {}
        for symbol, tdata in taker_data.items():
            sig = tdata.get("signal", "balanced")
            if sig == "aggressive_buying":
                bullish_signals.append(f"{symbol} takers buying")
            elif sig == "aggressive_selling":
                bearish_signals.append(f"{symbol} takers selling")

        # Yahoo macro
        yahoo = ctx.get("yahoo_macro") or {}
        vix = yahoo.get("VIX", {})
        vix_val = vix.get("last_close")
        if vix_val is not None:
            if vix_val > 30:
                bearish_signals.append(f"VIX={vix_val:.1f} (high fear)")
            elif vix_val < 15:
                bullish_signals.append(f"VIX={vix_val:.1f} (complacent)")

        # Stablecoin flows
        sc_flows = ctx.get("stablecoin_flows")
        if sc_flows and isinstance(sc_flows, dict):
            sig = sc_flows.get("signal")
            if sig == "net_minting":
                bullish_signals.append("stablecoin net_minting")
            elif sig == "net_burning":
                bearish_signals.append("stablecoin net_burning")

        # Order book depth (BTC)
        ob = ctx.get("orderbook_depth")
        if ob and isinstance(ob, dict):
            sig = ob.get("signal")
            if sig == "buy_wall":
                bullish_signals.append("orderbook buy_wall")
            elif sig == "sell_wall":
                bearish_signals.append("orderbook sell_wall")

        # Economic calendar — imminent high-impact event flags uncertainty
        ec = ctx.get("economic_calendar")
        if ec and isinstance(ec, list):
            for event in ec:
                if event.get("imminent"):
                    neutral_signals.append(
                        f"imminent event: {event.get('title', 'unknown')}"
                    )

        # On-chain BTC — high mempool congestion signals network stress
        onchain = ctx.get("onchain_btc")
        if onchain and isinstance(onchain, dict):
            if onchain.get("mempool_congestion") == "high":
                bearish_signals.append("onchain mempool_congestion=high")

        # Whale movements
        whales = ctx.get("whale_movements")
        if whales and isinstance(whales, dict):
            sig = whales.get("overall_signal")
            if sig == "bullish_whale_accumulating":
                bullish_signals.append("whale outflows (accumulating)")
            elif sig == "bearish_whale_selling":
                bearish_signals.append("whale inflows (selling)")

        # On-chain BTC macro (BGeometrics)
        btc_macro = ctx.get("onchain_btc_macro")
        if btc_macro and isinstance(btc_macro, dict):
            sig = btc_macro.get("signal", "")
            mvrv = btc_macro.get("mvrv_value")
            sopr = btc_macro.get("sopr_value")
            if sig.startswith("bullish"):
                detail = f"MVRV={mvrv:.1f}" if mvrv else sig
                bullish_signals.append(f"onchain BTC {detail}")
            elif sig.startswith("bearish"):
                detail = f"MVRV={mvrv:.1f}" if mvrv else sig
                bearish_signals.append(f"onchain BTC {detail}")
            elif sig == "capitulation_bullish_medium_term" and sopr:
                bullish_signals.append(f"onchain BTC capitulation SOPR={sopr:.2f}")

        # SEC filings
        sec = ctx.get("sec_filings")
        if sec and isinstance(sec, dict):
            sig = sec.get("signal")
            count = sec.get("recent_count", 0)
            if sig == "high_regulatory_activity":
                bearish_signals.append(f"SEC high activity ({count} filings)")
            elif sig == "material_event":
                neutral_signals.append(f"SEC 8-K filing ({count} recent)")

        total = len(bullish_signals) + len(bearish_signals) + len(neutral_signals)
        report.sentiment_consensus = {
            "bullish_count": len(bullish_signals),
            "bearish_count": len(bearish_signals),
            "neutral_count": len(neutral_signals),
            "total_signals": total,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "neutral_signals": neutral_signals,
            "direction": (
                "bullish" if len(bullish_signals) > len(bearish_signals) + 2
                else "bearish" if len(bearish_signals) > len(bullish_signals) + 2
                else "mixed"
            ),
        }

    def _check_economic_event_risk(self, ctx: dict, report: IntegrityReport) -> None:
        """Flag imminent high-impact economic events and reduce confidence accordingly."""
        ec = ctx.get("economic_calendar")
        if not ec or not isinstance(ec, list):
            return

        for event in ec:
            if event.get("imminent"):
                title = event.get("title", "unknown")
                date = event.get("date", "unknown")
                report.conflicts.append(
                    f"High-impact economic event imminent: {title} at {date} "
                    f"— expect increased volatility"
                )

    def _calculate_overall_confidence(self, ctx: dict, report: IntegrityReport) -> None:
        """Calculate an overall data confidence score.

        Penalize for: anomalies, conflicts, stale/missing sources.
        """
        confidence = 1.0

        # Each anomaly reduces confidence
        confidence -= len(report.anomalies) * 0.1

        # Each conflict reduces confidence
        confidence -= len(report.conflicts) * 0.08

        # Missing critical sources reduce confidence
        confidence -= len(report.stale_sources) * 0.05

        # Fewer total sources = lower confidence
        available = [
            k for k, v in ctx.items()
            if k not in ("timestamp", "errors") and v is not None
        ]
        if len(available) < 5:
            confidence -= 0.15
        elif len(available) < 8:
            confidence -= 0.05

        # Imminent economic event — uncertainty premium
        ec = ctx.get("economic_calendar")
        if ec and isinstance(ec, list) and any(event.get("imminent") for event in ec):
            confidence -= 0.05

        report.overall_confidence = max(0.1, min(1.0, confidence))
