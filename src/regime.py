"""Market regime detection — classifies current market as bull, bear, or sideways.

Regime influences strategy weight multipliers applied on top of per-pair weights,
giving trend-following strategies more authority in trending markets and mean-reversion
strategies more authority when price is consolidating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.indicators import adx, atr, ema, sma


@dataclass
class MarketRegime:
    regime: str            # "bull" | "bear" | "sideways"
    confidence: float      # 0.0 – 1.0
    adx: float             # ADX value used in detection
    volatility_pct: float  # ATR as % of price (current)
    trend_direction: str   # "up" | "down" | "flat"

    def __str__(self) -> str:
        return (
            f"{self.regime.upper()} "
            f"(conf={self.confidence:.0%}, ADX={self.adx:.1f}, "
            f"vol={self.volatility_pct:.2%}, dir={self.trend_direction})"
        )


# Strategy weight multipliers per regime
_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "bull": {
        "trend_following": 1.3,
        "momentum": 1.2,
        "mean_reversion": 0.7,
        "breakout": 1.0,
    },
    "bear": {
        "trend_following": 1.0,
        "momentum": 1.1,
        "mean_reversion": 1.3,
        "breakout": 0.6,
    },
    "sideways": {
        "trend_following": 0.6,
        "momentum": 1.0,
        "mean_reversion": 1.8,   # Boosted from 1.4 — help signals survive 0.5 threshold
        "breakout": 0.5,
        "range": 1.5,            # Range strategy thrives in sideways
    },
}

# Thresholds
_ADX_TRENDING = 25.0    # ADX above this → directional market
_ADX_RANGING = 20.0     # ADX below this → ranging / sideways market
_EMA_SLOPE_PERIODS = 5  # Number of periods to measure EMA slope
_SMA_SHORT = 20
_SMA_LONG = 50
_EMA_FAST = 9


class RegimeDetector:
    """Classifies the current market regime from OHLCV + indicator data.

    Each regime is determined by a weighted vote across four signals:
      1. ADX strength (trending vs ranging)
      2. Price vs 20/50-period SMA (structural bias)
      3. ATR% volatility level
      4. EMA slope direction

    The vote totals produce a confidence score in [0, 1] and a regime label.
    """

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Detect regime from a DataFrame with OHLCV columns.

        The DataFrame must have at least 60 rows (to compute 50-period SMA and ADX).
        Returns a conservative sideways regime with zero confidence when data is
        insufficient.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].
                Indicators are computed internally — do NOT require pre-enriched data.

        Returns:
            MarketRegime dataclass with regime, confidence, and diagnostic values.
        """
        if len(df) < 60:
            return MarketRegime(
                regime="sideways",
                confidence=0.0,
                adx=0.0,
                volatility_pct=0.0,
                trend_direction="flat",
            )

        # --- Raw indicator values --------------------------------------------------
        adx_series = adx(df, 14)
        atr_series = atr(df, 14)
        ema_fast_series = ema(df, _EMA_FAST)
        sma20_series = sma(df, _SMA_SHORT)
        sma50_series = sma(df, _SMA_LONG)

        # Use second-to-last row (latest complete candle)
        last_idx = -2

        adx_val = float(adx_series.iloc[last_idx])
        atr_val = float(atr_series.iloc[last_idx])
        close_val = float(df["close"].iloc[last_idx])
        sma20_val = float(sma20_series.iloc[last_idx])
        sma50_val = float(sma50_series.iloc[last_idx])

        if close_val == 0 or np.isnan(adx_val) or np.isnan(atr_val):
            return MarketRegime(
                regime="sideways",
                confidence=0.0,
                adx=0.0,
                volatility_pct=0.0,
                trend_direction="flat",
            )

        volatility_pct = atr_val / close_val

        # EMA slope: compare current EMA to EMA _EMA_SLOPE_PERIODS candles back
        ema_current = float(ema_fast_series.iloc[last_idx])
        ema_prev = float(ema_fast_series.iloc[last_idx - _EMA_SLOPE_PERIODS])
        ema_slope_pct = (ema_current - ema_prev) / ema_prev if ema_prev != 0 else 0.0

        # --- Signal votes ---------------------------------------------------------
        # Each signal contributes ±1 to bull/bear scores and to "trending" vs "ranging"
        bull_score = 0.0
        bear_score = 0.0
        trending_score = 0.0  # Used to scale confidence away from sideways

        # Signal 1: ADX — is the market trending at all?
        if adx_val > _ADX_TRENDING:
            trending_score += 1.0
        elif adx_val < _ADX_RANGING:
            trending_score -= 1.0  # Strong evidence of ranging

        # Signal 2: Price structure — where is price relative to the two SMAs?
        if not (np.isnan(sma20_val) or np.isnan(sma50_val)):
            if close_val > sma20_val and sma20_val > sma50_val:
                # Classic bull structure: price > SMA20 > SMA50
                bull_score += 2.0
                trending_score += 0.5
            elif close_val < sma20_val and sma20_val < sma50_val:
                # Classic bear structure: price < SMA20 < SMA50
                bear_score += 2.0
                trending_score += 0.5
            elif close_val > sma20_val and close_val > sma50_val:
                # Price above both SMAs but no clean stack — mild bull bias
                bull_score += 1.0
            elif close_val < sma20_val and close_val < sma50_val:
                # Price below both SMAs but no clean stack — mild bear bias
                bear_score += 1.0
            # else: price between the SMAs — no clear structure

        # Signal 3: Volatility — high vol suggests directional expansion
        # Threshold: >0.5% ATR of price is elevated for 1h crypto candles
        if volatility_pct > 0.005:
            trending_score += 0.5

        # Signal 4: EMA slope — direction and strength
        # >0.1% slope per _EMA_SLOPE_PERIODS periods is meaningful
        if ema_slope_pct > 0.001:
            bull_score += 1.5
            trending_score += 0.5
        elif ema_slope_pct < -0.001:
            bear_score += 1.5
            trending_score += 0.5
        # else: flat slope — no additional trending_score

        # --- Regime classification -----------------------------------------------
        # Max possible scores: bull=5.5, bear=5.5, trending=3.0
        max_directional = 5.5
        total_directional = bull_score + bear_score

        if trending_score < 0 or (adx_val < _ADX_RANGING and total_directional < 2.0):
            # Clear ranging signal — classify as sideways regardless of direction
            regime = "sideways"
            # Confidence reflects how firmly we're sideways
            confidence = min(0.9, 0.4 + max(0.0, -trending_score) * 0.15 + max(0.0, (_ADX_RANGING - adx_val) / _ADX_RANGING) * 0.4)
        elif bull_score > bear_score:
            regime = "bull"
            net = bull_score - bear_score
            confidence = min(0.9, 0.35 + (net / max_directional) * 0.55 + max(0.0, (adx_val - _ADX_TRENDING) / 50) * 0.15)
        elif bear_score > bull_score:
            regime = "bear"
            net = bear_score - bull_score
            confidence = min(0.9, 0.35 + (net / max_directional) * 0.55 + max(0.0, (adx_val - _ADX_TRENDING) / 50) * 0.15)
        else:
            # Equal scores — default to sideways
            regime = "sideways"
            confidence = 0.35

        # Trend direction (independent of regime for diagnostic purposes)
        if ema_slope_pct > 0.001:
            trend_direction = "up"
        elif ema_slope_pct < -0.001:
            trend_direction = "down"
        else:
            trend_direction = "flat"

        return MarketRegime(
            regime=regime,
            confidence=round(confidence, 4),
            adx=round(adx_val, 2),
            volatility_pct=round(volatility_pct, 6),
            trend_direction=trend_direction,
        )

    def get_strategy_weight_adjustments(self, regime: MarketRegime) -> dict[str, float]:
        """Return per-strategy confidence multipliers for the given regime.

        These multipliers are applied multiplicatively on top of existing per-pair
        weights in the strategy engine. Values > 1 boost a strategy's signals;
        values < 1 penalise them.

        Args:
            regime: The detected MarketRegime.

        Returns:
            Dict mapping strategy name → float multiplier.
        """
        base = _REGIME_WEIGHTS.get(regime.regime, _REGIME_WEIGHTS["sideways"])

        if regime.confidence < 0.4:
            # Low-confidence detection — blend towards neutral (1.0) to avoid
            # making aggressive adjustments based on uncertain regime data
            blend = regime.confidence / 0.4  # 0.0 → neutral, 1.0 → full regime weights
            return {
                strategy: 1.0 + (multiplier - 1.0) * blend
                for strategy, multiplier in base.items()
            }

        return dict(base)
