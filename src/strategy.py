"""Strategy engine — generates trading signals from indicator data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd

from src.config import StrategyConfig
from src.exchange import parse_pair
from src.indicators import compute_all
from src.models import Signal
from src.regime import MarketRegime, RegimeDetector
from src.utils import log, now_iso


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig,
                enrich_fn=None) -> Signal | None:
        """Analyze multi-timeframe data for a pair and return a signal or None."""
        ...

    def _enrich(self, df: pd.DataFrame, config: StrategyConfig, enrich_fn=None) -> pd.DataFrame:
        """[OPT-3] Get enriched DataFrame using cache or computing fresh."""
        if enrich_fn is not None:
            return enrich_fn(df)
        return compute_all(df, config.__dict__)

    def _get_latest(self, df: pd.DataFrame) -> pd.Series | None:
        """Get the latest complete candle (second to last, since last may be incomplete)."""
        if len(df) < 2:
            return None
        return df.iloc[-2]


class MomentumStrategy(BaseStrategy):
    """RSI extremes + MACD crossover + volume confirmation."""

    name = "momentum"

    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig,
                enrich_fn=None) -> Signal | None:
        # Use 15m timeframe as primary
        tf = "15m" if "15m" in data else list(data.keys())[0]
        df = data[tf]
        if len(df) < config.lookback.min_candles:
            return None

        enriched = self._enrich(df, config, enrich_fn)
        latest = self._get_latest(enriched)
        prev = enriched.iloc[-3] if len(enriched) > 2 else None
        if latest is None or prev is None:
            return None

        rsi_val = latest.get("rsi")
        macd_hist = latest.get("macd_hist")
        prev_macd_hist = prev.get("macd_hist")
        vol_ratio = latest.get("volume_ratio")

        if pd.isna(rsi_val) or pd.isna(macd_hist) or pd.isna(vol_ratio):
            return None

        direction = "hold"
        confidence = 0.0
        reasoning_parts = []

        # Bullish: RSI oversold + MACD histogram turning positive + volume spike
        if rsi_val < config.rsi_oversold and macd_hist > prev_macd_hist and vol_ratio > 1.3:
            direction = "long"
            confidence = min(0.9, 0.5 + (config.rsi_oversold - rsi_val) / 100 + (vol_ratio - 1) * 0.2)
            reasoning_parts = [
                f"RSI oversold ({rsi_val:.1f})",
                f"MACD histogram turning up ({macd_hist:.4f})",
                f"Volume {vol_ratio:.1f}x average",
            ]

        # Bearish: RSI overbought + MACD histogram turning negative + volume spike
        elif rsi_val > config.rsi_overbought and macd_hist < prev_macd_hist and vol_ratio > 1.3:
            direction = "short"
            confidence = min(0.9, 0.5 + (rsi_val - config.rsi_overbought) / 100 + (vol_ratio - 1) * 0.2)
            reasoning_parts = [
                f"RSI overbought ({rsi_val:.1f})",
                f"MACD histogram turning down ({macd_hist:.4f})",
                f"Volume {vol_ratio:.1f}x average",
            ]

        if direction == "hold":
            return None

        return Signal(
            pair=pair, timeframe=tf, direction=direction, confidence=confidence,
            strategy_name=self.name,
            indicators={"rsi": rsi_val, "macd_hist": macd_hist, "volume_ratio": vol_ratio},
            reasoning="; ".join(reasoning_parts), timestamp=now_iso(),
        )


class TrendFollowingStrategy(BaseStrategy):
    """EMA crossover + ADX trend strength + ATR-based stops."""

    name = "trend_following"

    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig,
                enrich_fn=None) -> Signal | None:
        tf = "1h" if "1h" in data else list(data.keys())[0]
        df = data[tf]
        if len(df) < config.lookback.min_candles:
            return None

        enriched = self._enrich(df, config, enrich_fn)
        latest = self._get_latest(enriched)
        prev = enriched.iloc[-3] if len(enriched) > 2 else None
        if latest is None or prev is None:
            return None

        ema_f = latest.get("ema_fast")
        ema_s = latest.get("ema_slow")
        prev_ema_f = prev.get("ema_fast")
        prev_ema_s = prev.get("ema_slow")
        adx_val = latest.get("adx")
        bb_mid = latest.get("bb_middle")
        close = latest.get("close")

        if any(pd.isna(v) for v in [ema_f, ema_s, prev_ema_f, prev_ema_s, adx_val, close]):
            return None

        direction = "hold"
        confidence = 0.0
        reasoning_parts = []

        # Bullish crossover + strong trend + price above BB middle
        if ema_f > ema_s and prev_ema_f <= prev_ema_s and adx_val > 25:
            if not pd.isna(bb_mid) and close > bb_mid:
                direction = "long"
                confidence = min(0.85, 0.5 + (adx_val - 25) / 100 + 0.1)
                reasoning_parts = [
                    f"EMA bullish crossover ({ema_f:.2f} > {ema_s:.2f})",
                    f"Strong trend (ADX {adx_val:.1f})",
                    f"Price above BB middle",
                ]

        # Bearish crossover + strong trend + price below BB middle
        elif ema_f < ema_s and prev_ema_f >= prev_ema_s and adx_val > 25:
            if not pd.isna(bb_mid) and close < bb_mid:
                direction = "short"
                confidence = min(0.85, 0.5 + (adx_val - 25) / 100 + 0.1)
                reasoning_parts = [
                    f"EMA bearish crossover ({ema_f:.2f} < {ema_s:.2f})",
                    f"Strong trend (ADX {adx_val:.1f})",
                    f"Price below BB middle",
                ]

        if direction == "hold":
            return None

        return Signal(
            pair=pair, timeframe=tf, direction=direction, confidence=confidence,
            strategy_name=self.name,
            indicators={"ema_fast": ema_f, "ema_slow": ema_s, "adx": adx_val},
            reasoning="; ".join(reasoning_parts), timestamp=now_iso(),
        )


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band bounce + RSI divergence."""

    name = "mean_reversion"

    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig,
                enrich_fn=None) -> Signal | None:
        tf = "15m" if "15m" in data else list(data.keys())[0]
        df = data[tf]
        if len(df) < config.lookback.min_candles:
            return None

        enriched = self._enrich(df, config, enrich_fn)
        latest = self._get_latest(enriched)
        if latest is None:
            return None

        close = latest.get("close")
        bb_lower = latest.get("bb_lower")
        bb_upper = latest.get("bb_upper")
        bb_mid = latest.get("bb_middle")
        rsi_val = latest.get("rsi")
        stoch_k = latest.get("stoch_k")

        if any(pd.isna(v) for v in [close, bb_lower, bb_upper, rsi_val]):
            return None

        direction = "hold"
        confidence = 0.0
        reasoning_parts = []

        # Price at/below lower band + RSI oversold zone + stochastic oversold
        if close <= bb_lower and rsi_val < 35:
            confidence = min(0.8, 0.4 + (35 - rsi_val) / 100 + 0.15)
            if not pd.isna(stoch_k) and stoch_k < 20:
                confidence = min(0.85, confidence + 0.1)
                reasoning_parts.append(f"Stochastic oversold ({stoch_k:.1f})")
            direction = "long"
            reasoning_parts = [
                f"Price at lower BB ({close:.2f} <= {bb_lower:.2f})",
                f"RSI oversold zone ({rsi_val:.1f})",
            ] + reasoning_parts

        # Price at/above upper band + RSI overbought zone
        elif close >= bb_upper and rsi_val > 65:
            confidence = min(0.8, 0.4 + (rsi_val - 65) / 100 + 0.15)
            if not pd.isna(stoch_k) and stoch_k > 80:
                confidence = min(0.85, confidence + 0.1)
                reasoning_parts.append(f"Stochastic overbought ({stoch_k:.1f})")
            direction = "short"
            reasoning_parts = [
                f"Price at upper BB ({close:.2f} >= {bb_upper:.2f})",
                f"RSI overbought zone ({rsi_val:.1f})",
            ] + reasoning_parts

        if direction == "hold":
            return None

        return Signal(
            pair=pair, timeframe=tf, direction=direction, confidence=confidence,
            strategy_name=self.name,
            indicators={"close": close, "bb_lower": bb_lower, "bb_upper": bb_upper, "rsi": rsi_val},
            reasoning="; ".join(reasoning_parts), timestamp=now_iso(),
        )


class BreakoutStrategy(BaseStrategy):
    """Volume breakout above/below recent range."""

    name = "breakout"

    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig,
                enrich_fn=None) -> Signal | None:
        tf = "1h" if "1h" in data else list(data.keys())[0]
        df = data[tf]
        lookback_periods = config.lookback.get_lookback(pair)
        min_candles_needed = lookback_periods + 5  # Need lookback + buffer
        if len(df) < min_candles_needed:
            log.debug(f"Breakout: skipping {pair} — only {len(df)} candles (need {min_candles_needed})")
            return None

        enriched = self._enrich(df, config, enrich_fn)
        latest = self._get_latest(enriched)
        if latest is None:
            return None

        close = latest.get("close")
        vol_ratio = latest.get("volume_ratio")
        atr_val = latest.get("atr")

        if any(pd.isna(v) for v in [close, vol_ratio, atr_val]):
            return None

        # Configurable lookback for range detection
        lb_slice = enriched.iloc[-(lookback_periods + 2):-2]  # Exclude incomplete + signal candle
        if len(lb_slice) < int(lookback_periods * 0.75):
            return None

        period_high = lb_slice["high"].max()
        period_low = lb_slice["low"].min()

        direction = "hold"
        confidence = 0.0
        reasoning_parts = []

        # Breakout above range with volume
        if close > period_high and vol_ratio > 1.8:
            direction = "long"
            confidence = min(0.85, 0.5 + (vol_ratio - 1.5) * 0.15)
            reasoning_parts = [
                f"Price broke above {lookback_periods}-period high ({close:.2f} > {period_high:.2f})",
                f"Volume {vol_ratio:.1f}x average",
            ]

        # Breakdown below range with volume
        elif close < period_low and vol_ratio > 1.8:
            direction = "short"
            confidence = min(0.85, 0.5 + (vol_ratio - 1.5) * 0.15)
            reasoning_parts = [
                f"Price broke below {lookback_periods}-period low ({close:.2f} < {period_low:.2f})",
                f"Volume {vol_ratio:.1f}x average",
            ]

        if direction == "hold":
            return None

        return Signal(
            pair=pair, timeframe=tf, direction=direction, confidence=confidence,
            strategy_name=self.name,
            indicators={"close": close, "period_high": period_high, "period_low": period_low, "volume_ratio": vol_ratio},
            reasoning="; ".join(reasoning_parts), timestamp=now_iso(),
        )


class RangeStrategy(BaseStrategy):
    """Range/grid trading for SIDEWAYS markets.

    Exploits mean reversion within Bollinger Bands when price is consolidating.
    Only active in SIDEWAYS regime (gated in StrategyEngine.generate_signals).
    """

    name = "range"

    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig,
                enrich_fn=None) -> Signal | None:
        tf = "15m" if "15m" in data else list(data.keys())[0]
        df = data[tf]
        if len(df) < config.lookback.min_candles:
            return None

        enriched = self._enrich(df, config, enrich_fn)
        latest = self._get_latest(enriched)
        if latest is None:
            return None

        close = latest.get("close")
        bb_lower = latest.get("bb_lower")
        bb_mid = latest.get("bb_middle")
        bb_upper = latest.get("bb_upper")
        rsi = latest.get("rsi")

        if any(pd.isna(v) for v in [close, bb_lower, bb_mid, bb_upper, rsi]):
            return None

        band_width = bb_upper - bb_lower
        if band_width <= 0:
            return None

        position = (close - bb_lower) / band_width

        direction = "hold"
        confidence = 0.0
        reasoning_parts = []

        # BUY: price in lower third of band + RSI mildly oversold
        if position < 0.35 and rsi < 40:
            confidence = min(0.65, 0.45 + (0.35 - position) * 0.3 + (40 - rsi) / 200)
            direction = "long"
            reasoning_parts = [
                f"Range trade: price at {position*100:.0f}% of BB width",
                f"RSI mild oversold ({rsi:.1f})",
                f"Target: mid band ({bb_mid:.2f})",
            ]

        # SELL: price in upper third of band + RSI mildly overbought
        elif position > 0.65 and rsi > 60:
            confidence = min(0.65, 0.45 + (position - 0.65) * 0.3 + (rsi - 60) / 200)
            direction = "short"
            reasoning_parts = [
                f"Range trade: price at {position*100:.0f}% of BB width",
                f"RSI mild overbought ({rsi:.1f})",
                f"Target: mid band ({bb_mid:.2f})",
            ]

        if direction == "hold":
            return None

        return Signal(
            pair=pair, timeframe=tf, direction=direction, confidence=confidence,
            strategy_name=self.name,
            indicators={"close": close, "bb_lower": bb_lower, "bb_upper": bb_upper,
                        "bb_middle": bb_mid, "position_in_band": round(position, 3), "rsi": rsi},
            reasoning="; ".join(reasoning_parts), timestamp=now_iso(),
        )


class FlightToSafetyStrategy(BaseStrategy):
    """Crisis response — rotates to safe havens when macro conditions deteriorate.

    Triggers on: sharp drawdowns, high volatility spikes, bear regime with
    extreme momentum. When active, generates LONG signals for safe havens
    (gold) and SHORT signals for risk assets (crypto, tech stocks).
    """

    name = "flight_to_safety"

    # Safe havens: go long
    SAFE_HAVENS = {"XYZ-GOLD", "XYZ-SILVER"}
    # Risk assets: close / go short
    RISK_ASSETS = {"BTC", "ETH", "AAVE", "XYZ-NVDA", "XYZ-TSLA", "XYZ-AAPL", "XYZ-MSFT"}

    def analyze(self, data: dict[str, pd.DataFrame], pair: str, config: StrategyConfig,
                enrich_fn=None) -> Signal | None:
        tf = "1h" if "1h" in data else "15m" if "15m" in data else list(data.keys())[0]
        df = data[tf]
        if len(df) < max(30, config.lookback.min_candles):
            return None

        enriched = self._enrich(df, config, enrich_fn)
        latest = self._get_latest(enriched)
        if latest is None:
            return None

        close = latest.get("close")
        rsi = latest.get("rsi")
        atr = latest.get("atr")
        bb_upper = latest.get("bb_upper")
        bb_lower = latest.get("bb_lower")
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")

        if any(pd.isna(v) for v in [close, rsi, atr, ema_fast, ema_slow]):
            return None
        if close <= 0 or ema_slow <= 0:
            return None

        # Crisis detection: multiple signals must align
        crisis_score = 0.0
        crisis_reasons = []

        # 1. Sharp recent drop: close well below slow EMA
        if close < ema_slow * 0.97:
            crisis_score += 0.25
            pct_below = (1 - close / ema_slow) * 100
            crisis_reasons.append(f"Price {pct_below:.1f}% below EMA-slow (crash signal)")

        # 2. Extreme RSI: deeply oversold (panic selling) or momentum collapse
        if rsi < 25:
            crisis_score += 0.2
            crisis_reasons.append(f"RSI extreme oversold ({rsi:.0f}) — panic selling")
        elif rsi > 80:
            # Blow-off top — crash likely incoming
            crisis_score += 0.15
            crisis_reasons.append(f"RSI blow-off top ({rsi:.0f}) — reversal imminent")

        # 3. Volatility expansion: ATR spike (use BB width as proxy)
        if bb_upper and bb_lower and bb_lower > 0:
            bb_width = (bb_upper - bb_lower) / close
            if bb_width > 0.06:  # >6% band width = high vol
                crisis_score += 0.2
                crisis_reasons.append(f"Volatility spike: BB width {bb_width*100:.1f}%")

        # 4. Death cross: fast EMA below slow EMA with widening gap
        if ema_fast < ema_slow * 0.985:
            crisis_score += 0.2
            crisis_reasons.append("Death cross: fast EMA diverging below slow EMA")

        # 5. Consecutive red candles (look back 5 candles)
        if len(enriched) >= 7:
            recent = enriched.iloc[-7:-2]
            red_count = (recent["close"] < recent["open"]).sum()
            if red_count >= 4:
                crisis_score += 0.15
                crisis_reasons.append(f"{red_count}/5 consecutive red candles")

        # Need score >= 0.5 to trigger flight-to-safety
        if crisis_score < 0.5:
            return None

        base = parse_pair(pair)[0]
        confidence = min(0.85, 0.5 + crisis_score * 0.3)

        # Safe havens: go LONG (flight to quality)
        if base in self.SAFE_HAVENS:
            return Signal(
                pair=pair, timeframe=tf, direction="long", confidence=confidence,
                strategy_name=self.name,
                indicators={"crisis_score": round(crisis_score, 2), "rsi": rsi,
                            "close": close, "ema_slow": ema_slow},
                reasoning=f"FLIGHT TO SAFETY — {'; '.join(crisis_reasons)}. "
                          f"Rotating to safe haven {base}",
                timestamp=now_iso(),
            )

        # Risk assets: go SHORT (de-risk)
        if base in self.RISK_ASSETS:
            return Signal(
                pair=pair, timeframe=tf, direction="short", confidence=confidence,
                strategy_name=self.name,
                indicators={"crisis_score": round(crisis_score, 2), "rsi": rsi,
                            "close": close, "ema_slow": ema_slow},
                reasoning=f"FLIGHT TO SAFETY — {'; '.join(crisis_reasons)}. "
                          f"De-risking {base}, rotating to safe havens",
                timestamp=now_iso(),
            )

        return None


class StrategyEngine:
    """Runs all strategies and aggregates signals."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.strategies: list[BaseStrategy] = [
            MomentumStrategy(),
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            RangeStrategy(),
            FlightToSafetyStrategy(),
        ]
        # Regime detection — rule-based (fallback) + ML (when trained)
        self._regime_detector = RegimeDetector()
        self._ml_regime: "MLRegimeClassifier | None" = None
        self.current_regime: MarketRegime | None = None

        # ML price predictor (optional, when trained)
        self._price_predictor: "PricePredictor | None" = None

        # Lazy-init ML models (avoid import failure if sklearn not installed)
        try:
            from src.ml_signals import MLRegimeClassifier, PricePredictor
            self._ml_regime = MLRegimeClassifier()
            self._price_predictor = PricePredictor()
            if self._ml_regime.is_available():
                log.info("ML regime classifier loaded — will use ML over rule-based")
            if self._price_predictor.is_available():
                log.info("LSTM price predictor loaded")
        except Exception:
            pass

        # Funding/OI data injected each cycle from macro_analyst
        self._funding_rates: dict[str, float] = {}   # {"BTC": 0.0003, ...}
        self._open_interest: dict[str, dict] = {}    # {"BTC": {"oi": ..., "mark_price": ...}, ...}

        # [OPT-3] Per-cycle enriched DataFrame cache — avoids redundant compute_all calls
        self._enriched_cache: dict[int, pd.DataFrame] = {}

    def set_market_signals(self, funding_rates: dict[str, float], open_interest: dict[str, dict]) -> None:
        """Inject funding rates and open interest data for use in signal filters.

        Args:
            funding_rates: mapping of base asset to raw funding rate, e.g. {"BTC": 0.0003}
            open_interest: mapping of base asset to OI context, e.g. {"BTC": {"oi": 123456.0, "mark_price": 70000.0}}
        """
        self._funding_rates = funding_rates or {}
        self._open_interest = open_interest or {}
        log.debug(
            f"Market signals updated: funding={list(self._funding_rates.keys())}, "
            f"OI={list(self._open_interest.keys())}"
        )

    def load_evolved_strategies(self, evolved: list[BaseStrategy]):
        """Add evolved strategies to the engine (called by StrategyEvolver integration)."""
        # Remove any previously loaded evolved strategies
        self.strategies = [
            s for s in self.strategies
            if s.__class__.__module__ == "src.strategy"
        ]
        for strat in evolved:
            self.strategies.append(strat)
        if evolved:
            log.info(f"Loaded {len(evolved)} evolved strategies into engine")

    def get_enriched(self, df: pd.DataFrame) -> pd.DataFrame:
        """[OPT-3] Return enriched DataFrame from cache, computing only on first call per df."""
        df_id = id(df)
        if df_id not in self._enriched_cache:
            self._enriched_cache[df_id] = compute_all(df, self.config.__dict__)
        return self._enriched_cache[df_id]

    def generate_signals(self, market_data: dict[str, dict[str, pd.DataFrame]]) -> list[Signal]:
        """Run all strategies on all pairs, return signals."""
        # [OPT-3] Clear per-cycle cache at the start of each cycle
        self._enriched_cache.clear()

        # Detect regime from 1h BTC data (or the first available pair's 1h data)
        # Prefer ML classifier over rule-based when trained
        regime_df = self._select_regime_df(market_data)
        if regime_df is not None:
            try:
                ml_used = False
                if self._ml_regime and self._ml_regime.is_available():
                    ml_result = self._ml_regime.classify(regime_df)
                    if ml_result is not None:
                        self.current_regime = MarketRegime(
                            regime=ml_result.regime,
                            confidence=ml_result.confidence,
                            adx=0.0,  # ML doesn't use ADX directly
                            volatility_pct=0.0,
                            trend_direction="up" if ml_result.regime == "bull" else "down" if ml_result.regime == "bear" else "flat",
                        )
                        log.debug(f"ML regime: {ml_result.regime} (conf={ml_result.confidence:.0%}, probs={ml_result.probabilities})")
                        ml_used = True

                if not ml_used:
                    self.current_regime = self._regime_detector.detect(regime_df)
                    log.debug(f"Rule-based regime: {self.current_regime}")
            except Exception as e:
                log.warning(f"Regime detection failed (non-critical): {e}")
                self.current_regime = None
        else:
            self.current_regime = None

        # Pre-compute regime weight adjustments (neutral 1.0 if regime unknown)
        regime_adjustments: dict[str, float] = (
            self._regime_detector.get_strategy_weight_adjustments(self.current_regime)
            if self.current_regime is not None
            else {}
        )

        signals: list[Signal] = []

        for pair, timeframe_data in market_data.items():
            for strategy in self.strategies:
                # RangeStrategy only fires in SIDEWAYS regime
                if strategy.name == "range":
                    if not self.current_regime or self.current_regime.regime != "sideways":
                        continue
                try:
                    signal = strategy.analyze(timeframe_data, pair, self.config,
                                              enrich_fn=self.get_enriched)
                    if signal and signal.confidence >= 0.5:
                        # Apply per-pair strategy weight
                        signal = self._apply_pair_weight(signal)
                        # Apply regime weight on top of pair weight (multiplicative)
                        signal = self._apply_regime_weight(signal, regime_adjustments)
                        # Apply funding rate and OI filters
                        signal = self._apply_funding_filter(signal, pair)
                        signal = self._apply_oi_filter(signal, pair)
                        # Apply ML price prediction filter (if trained)
                        signal = self._apply_ml_prediction_filter(signal, timeframe_data)
                        if signal.confidence >= 0.5:  # Recheck after filters
                            signals.append(signal)
                except Exception as e:
                    log.warning(f"Strategy {strategy.name} failed on {pair}: {e}")

        # Apply correlation guard before aggregation
        signals = self._apply_correlation_guard(signals, market_data)

        return self._aggregate_signals(signals)

    def _select_regime_df(self, market_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame | None:
        """Pick the best DataFrame for regime detection.

        Preference order:
          1. BTC pair, 1h timeframe
          2. BTC pair, any timeframe
          3. First available pair, 1h timeframe
          4. First available pair, any timeframe
        """
        # Try BTC 1h first
        for pair_key, tf_data in market_data.items():
            if pair_key.startswith("BTC/") and "1h" in tf_data:
                return tf_data["1h"]

        # BTC with any timeframe
        for pair_key, tf_data in market_data.items():
            if pair_key.startswith("BTC/") and tf_data:
                return next(iter(tf_data.values()))

        # Fallback: first pair 1h
        for tf_data in market_data.values():
            if "1h" in tf_data:
                return tf_data["1h"]

        # Fallback: first pair first timeframe
        for tf_data in market_data.values():
            if tf_data:
                return next(iter(tf_data.values()))

        return None

    def _apply_regime_weight(self, signal: Signal, adjustments: dict[str, float]) -> Signal:
        """Multiply signal confidence by the regime-based strategy multiplier."""
        multiplier = adjustments.get(signal.strategy_name, 1.0)
        if multiplier == 1.0:
            return signal

        signal.confidence = min(0.95, max(0.1, signal.confidence * multiplier))

        regime_label = self.current_regime.regime if self.current_regime else "unknown"
        if multiplier > 1.0:
            signal.reasoning += f" | Regime-boosted ({multiplier:.1f}x, {regime_label})"
        else:
            signal.reasoning += f" | Regime-penalized ({multiplier:.1f}x, {regime_label})"

        return signal

    def _apply_pair_weight(self, signal: Signal) -> Signal:
        """Adjust signal confidence based on per-pair strategy weights."""
        weights = self.config.pair_weights.get(signal.pair, {})
        weight = weights.get(signal.strategy_name, 1.0)

        if weight != 1.0:
            signal.confidence = min(0.95, max(0.1, signal.confidence * weight))
            if weight > 1.0:
                signal.reasoning += f" | Pair-boosted ({weight:.1f}x for {signal.pair})"
            elif weight < 1.0:
                signal.reasoning += f" | Pair-penalized ({weight:.1f}x for {signal.pair})"

        return signal

    def _apply_funding_filter(self, signal: Signal, pair: str) -> Signal:
        """Adjust signal confidence using contrarian funding rate signals.

        Extreme positive funding (> 0.01%) means longs are crowded — bearish contrarian.
        Extreme negative funding (< -0.01%) means shorts are crowded — bullish contrarian.
        Very extreme levels (abs > 0.05%) double the adjustment.
        Confidence is clamped to [0.1, 0.95] after adjustment.
        """
        if not self._funding_rates:
            return signal

        try:
            base, _ = parse_pair(pair)
        except ValueError:
            return signal

        rate = self._funding_rates.get(base)
        if rate is None:
            return signal

        _EXTREME = 0.0001   # 0.01%
        _VERY_EXTREME = 0.0005  # 0.05%

        adjustment = 0.0
        note = ""

        if rate > _VERY_EXTREME:
            # Very extreme positive funding — strong contrarian bear signal
            if signal.direction == "short":
                adjustment = +0.10
                note = f"Funding very extreme +{rate*100:.4f}% (crowded longs) | Short boosted"
            elif signal.direction == "long":
                adjustment = -0.10
                note = f"Funding very extreme +{rate*100:.4f}% (crowded longs) | Long penalized"

        elif rate > _EXTREME:
            # Extreme positive funding — contrarian bear signal
            if signal.direction == "short":
                adjustment = +0.05
                note = f"Funding extreme +{rate*100:.4f}% (crowded longs) | Short boosted"
            elif signal.direction == "long":
                adjustment = -0.05
                note = f"Funding extreme +{rate*100:.4f}% (crowded longs) | Long penalized"

        elif rate < -_VERY_EXTREME:
            # Very extreme negative funding — strong contrarian bull signal
            if signal.direction == "long":
                adjustment = +0.10
                note = f"Funding very extreme {rate*100:.4f}% (crowded shorts) | Long boosted"
            elif signal.direction == "short":
                adjustment = -0.10
                note = f"Funding very extreme {rate*100:.4f}% (crowded shorts) | Short penalized"

        elif rate < -_EXTREME:
            # Extreme negative funding — contrarian bull signal
            if signal.direction == "long":
                adjustment = +0.05
                note = f"Funding extreme {rate*100:.4f}% (crowded shorts) | Long boosted"
            elif signal.direction == "short":
                adjustment = -0.05
                note = f"Funding extreme {rate*100:.4f}% (crowded shorts) | Short penalized"

        if adjustment != 0.0:
            signal.confidence = min(0.95, max(0.1, signal.confidence + adjustment))
            signal.reasoning += f" | {note}"

        return signal

    def _apply_oi_filter(self, signal: Signal, pair: str) -> Signal:
        """Boost breakout signal confidence when high open interest confirms momentum.

        High OI during a breakout means leveraged positions are building behind the
        move — a real momentum signal rather than a false breakout.
        """
        if not self._open_interest or signal.strategy_name != "breakout":
            return signal

        try:
            base, _ = parse_pair(pair)
        except ValueError:
            return signal

        oi_data = self._open_interest.get(base)
        if not oi_data:
            return signal

        oi = oi_data.get("oi", 0.0)
        mark_price = oi_data.get("mark_price", 0.0)
        oi_usd = oi * mark_price if mark_price else 0.0

        # "High OI" thresholds: BTC >5B USD, others >500M USD
        _HIGH_OI_USD = 5_000_000_000 if base == "BTC" else 500_000_000

        if oi_usd >= _HIGH_OI_USD:
            signal.confidence = min(0.95, signal.confidence + 0.08)
            signal.reasoning += (
                f" | High OI confirms breakout (${oi_usd/1e9:.2f}B open interest)"
            )

        return signal

    def _apply_ml_prediction_filter(self, signal: Signal, timeframe_data: dict[str, pd.DataFrame]) -> Signal:
        """Boost or penalize signal based on ML price direction prediction.

        If the LSTM predicts the same direction as the signal → boost confidence.
        If it predicts the opposite direction → penalize.
        Only applies when the predictor is trained and confident (>0.6).
        """
        if not self._price_predictor or not self._price_predictor.is_available():
            return signal

        # Use 1h data for prediction, fallback to 15m
        df = timeframe_data.get("1h") or timeframe_data.get("15m")
        if df is None or len(df) < 90:
            return signal

        try:
            prediction = self._price_predictor.predict(df)
        except Exception:
            return signal

        if prediction is None or prediction.confidence < 0.6:
            return signal

        # Map prediction to signal compatibility
        signal_bullish = signal.direction == "long"
        pred_bullish = prediction.direction == "up"
        pred_bearish = prediction.direction == "down"

        if signal_bullish and pred_bullish:
            signal.confidence = min(0.95, signal.confidence + 0.06)
            signal.reasoning += f" | LSTM confirms UP ({prediction.confidence:.0%})"
        elif signal_bullish and pred_bearish:
            signal.confidence = max(0.1, signal.confidence - 0.06)
            signal.reasoning += f" | LSTM contradicts: DOWN ({prediction.confidence:.0%})"
        elif not signal_bullish and pred_bearish:
            signal.confidence = min(0.95, signal.confidence + 0.06)
            signal.reasoning += f" | LSTM confirms DOWN ({prediction.confidence:.0%})"
        elif not signal_bullish and pred_bullish:
            signal.confidence = max(0.1, signal.confidence - 0.06)
            signal.reasoning += f" | LSTM contradicts: UP ({prediction.confidence:.0%})"

        return signal

    def _apply_correlation_guard(
        self,
        signals: list[Signal],
        market_data: dict[str, dict[str, pd.DataFrame]],
    ) -> list[Signal]:
        """Block ETH signals that conflict with BTC direction.

        If BTC dropped >threshold in 1h, don't go long ETH.
        If BTC pumped >threshold in 1h, don't go short ETH.
        """
        guard = self.config.correlation_guard
        if not guard.enabled:
            return signals

        # Get BTC 1h change
        # Find BTC pair in market data (handles both BTC/USDT and BTC/USDC:USDC formats)
        btc_data = {}
        for pair_key, pair_data in market_data.items():
            if pair_key.startswith("BTC/"):
                btc_data = pair_data
                break
        btc_1h = btc_data.get("1h") if btc_data else None
        if btc_1h is None or len(btc_1h) < 2:
            return signals

        btc_close = btc_1h.iloc[-2]["close"]  # Latest complete candle
        btc_prev = btc_1h.iloc[-3]["close"] if len(btc_1h) > 2 else btc_close
        btc_change_pct = ((btc_close - btc_prev) / btc_prev) * 100 if btc_prev else 0

        filtered = []
        for sig in signals:
            if sig.pair.startswith("ETH/"):
                # Block ETH long if BTC is dumping
                if sig.direction == "long" and btc_change_pct < guard.btc_drop_threshold_pct:
                    from src.utils import log
                    log.info(
                        f"Correlation guard: blocked ETH LONG "
                        f"(BTC 1h change {btc_change_pct:+.1f}% < {guard.btc_drop_threshold_pct}%)"
                    )
                    continue
                # Block ETH short if BTC is pumping
                if sig.direction == "short" and btc_change_pct > guard.btc_pump_threshold_pct:
                    from src.utils import log
                    log.info(
                        f"Correlation guard: blocked ETH SHORT "
                        f"(BTC 1h change {btc_change_pct:+.1f}% > +{guard.btc_pump_threshold_pct}%)"
                    )
                    continue
            filtered.append(sig)

        return filtered

    def _aggregate_signals(self, signals: list[Signal]) -> list[Signal]:
        """If multiple strategies agree on the same pair+direction, boost confidence."""
        grouped: dict[tuple[str, str], list[Signal]] = defaultdict(list)
        for sig in signals:
            grouped[(sig.pair, sig.direction)].append(sig)

        result: list[Signal] = []
        for (pair, direction), sigs in grouped.items():
            if len(sigs) == 1:
                result.append(sigs[0])
            else:
                # Multiple strategies agree — take the highest confidence and boost it
                best = max(sigs, key=lambda s: s.confidence)
                boost = min(0.15, 0.05 * (len(sigs) - 1))
                best.confidence = min(0.95, best.confidence + boost)
                strategy_names = [s.strategy_name for s in sigs]
                best.reasoning += f" | Confirmed by {len(sigs)} strategies: {', '.join(strategy_names)}"
                result.append(best)

        # Sort by confidence descending
        result.sort(key=lambda s: s.confidence, reverse=True)
        return result
