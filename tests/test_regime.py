"""Tests for market regime detection."""

import numpy as np
import pandas as pd
import pytest

from src.regime import MarketRegime, RegimeDetector


# ---------------------------------------------------------------------------
# Helpers — synthetic OHLCV generators
# ---------------------------------------------------------------------------

def _make_ohlcv(close: np.ndarray) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a close price array."""
    n = len(close)
    np.random.seed(0)
    spread = np.abs(close) * 0.002  # tiny spread so high > close > low
    return pd.DataFrame({
        "open": close + np.random.randn(n) * spread * 0.5,
        "high": close + np.abs(np.random.randn(n)) * spread,
        "low": close - np.abs(np.random.randn(n)) * spread,
        "close": close,
        "volume": np.random.randint(500, 5000, n).astype(float),
    })


def _trending_up(n: int = 120, start: float = 100.0, slope: float = 0.5) -> pd.DataFrame:
    """Strong uptrend: linearly rising price with small noise."""
    np.random.seed(1)
    close = start + np.arange(n) * slope + np.random.randn(n) * 0.1
    return _make_ohlcv(close)


def _trending_down(n: int = 120, start: float = 200.0, slope: float = 0.5) -> pd.DataFrame:
    """Strong downtrend: linearly falling price with small noise."""
    np.random.seed(2)
    close = start - np.arange(n) * slope + np.random.randn(n) * 0.1
    return _make_ohlcv(close)


def _flat(n: int = 120, center: float = 100.0) -> pd.DataFrame:
    """Sideways market: price oscillating tightly around a center."""
    np.random.seed(3)
    close = center + np.random.randn(n) * 0.05
    return _make_ohlcv(close)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return RegimeDetector()


# ---------------------------------------------------------------------------
# Tests — detect()
# ---------------------------------------------------------------------------

class TestDetectInsufficientData:
    """detect() with fewer than 60 rows returns conservative sideways."""

    def test_too_few_rows(self, detector):
        df = _trending_up(n=30)
        result = detector.detect(df)

        assert result.regime == "sideways"
        assert result.confidence == 0.0
        assert result.adx == 0.0
        assert result.volatility_pct == 0.0
        assert result.trend_direction == "flat"

    def test_exactly_59_rows(self, detector):
        df = _trending_up(n=59)
        result = detector.detect(df)
        assert result.regime == "sideways"
        assert result.confidence == 0.0

    def test_empty_dataframe(self, detector):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = detector.detect(df)
        assert result.regime == "sideways"
        assert result.confidence == 0.0


class TestDetectBull:
    """Bull regime: price above both SMAs, positive EMA slope, high ADX."""

    def test_strong_uptrend_is_bull(self, detector):
        df = _trending_up(n=120, slope=0.5)
        result = detector.detect(df)
        assert result.regime == "bull"
        assert result.confidence > 0
        assert result.trend_direction == "up"

    def test_bull_confidence_in_range(self, detector):
        df = _trending_up(n=120, slope=0.8)
        result = detector.detect(df)
        assert 0.0 <= result.confidence <= 1.0


class TestDetectBear:
    """Bear regime: price below both SMAs, negative EMA slope."""

    def test_strong_downtrend_is_bear(self, detector):
        df = _trending_down(n=120, slope=0.5)
        result = detector.detect(df)
        assert result.regime == "bear"
        assert result.confidence > 0
        assert result.trend_direction == "down"

    def test_bear_confidence_in_range(self, detector):
        df = _trending_down(n=120, slope=0.8)
        result = detector.detect(df)
        assert 0.0 <= result.confidence <= 1.0


class TestDetectSideways:
    """Sideways regime: low ADX, no clear direction."""

    def test_flat_market_is_sideways(self, detector):
        df = _flat(n=120)
        result = detector.detect(df)
        assert result.regime == "sideways"

    def test_sideways_confidence_in_range(self, detector):
        df = _flat(n=120)
        result = detector.detect(df)
        assert 0.0 <= result.confidence <= 1.0


class TestConfidenceBounds:
    """Confidence must always be in [0, 1] regardless of input."""

    @pytest.mark.parametrize("builder", [_trending_up, _trending_down, _flat])
    def test_confidence_between_0_and_1(self, detector, builder):
        df = builder(n=120)
        result = detector.detect(df)
        assert 0.0 <= result.confidence <= 1.0

    def test_extreme_uptrend_confidence_capped(self, detector):
        df = _trending_up(n=200, slope=2.0)
        result = detector.detect(df)
        assert result.confidence <= 1.0

    def test_extreme_downtrend_confidence_capped(self, detector):
        df = _trending_down(n=200, start=500.0, slope=2.0)
        result = detector.detect(df)
        assert result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Tests — get_strategy_weight_adjustments()
# ---------------------------------------------------------------------------

class TestStrategyWeightAdjustments:
    """get_strategy_weight_adjustments() returns correct multipliers."""

    def test_bull_weights(self, detector):
        regime = MarketRegime(
            regime="bull", confidence=0.7, adx=30.0,
            volatility_pct=0.003, trend_direction="up",
        )
        weights = detector.get_strategy_weight_adjustments(regime)
        assert weights["trend_following"] == 1.3
        assert weights["momentum"] == 1.2
        assert weights["mean_reversion"] == 0.7
        assert weights["breakout"] == 1.0

    def test_bear_weights(self, detector):
        regime = MarketRegime(
            regime="bear", confidence=0.6, adx=28.0,
            volatility_pct=0.004, trend_direction="down",
        )
        weights = detector.get_strategy_weight_adjustments(regime)
        assert weights["trend_following"] == 1.0
        assert weights["momentum"] == 1.1
        assert weights["mean_reversion"] == 1.3
        assert weights["breakout"] == 0.6

    def test_sideways_weights(self, detector):
        regime = MarketRegime(
            regime="sideways", confidence=0.5, adx=15.0,
            volatility_pct=0.002, trend_direction="flat",
        )
        weights = detector.get_strategy_weight_adjustments(regime)
        assert weights["trend_following"] == 0.6
        assert weights["momentum"] == 1.0
        assert weights["mean_reversion"] == 1.8
        assert weights["breakout"] == 0.5
        assert weights["range"] == 1.5

    def test_low_confidence_blends_towards_neutral(self, detector):
        regime = MarketRegime(
            regime="bull", confidence=0.0, adx=30.0,
            volatility_pct=0.003, trend_direction="up",
        )
        weights = detector.get_strategy_weight_adjustments(regime)
        # At confidence 0.0, all weights should be exactly 1.0 (neutral)
        for w in weights.values():
            assert w == pytest.approx(1.0)

    def test_half_confidence_partially_blended(self, detector):
        regime = MarketRegime(
            regime="bull", confidence=0.2, adx=30.0,
            volatility_pct=0.003, trend_direction="up",
        )
        weights = detector.get_strategy_weight_adjustments(regime)
        # blend = 0.2 / 0.4 = 0.5 → halfway between neutral and full bull
        assert weights["trend_following"] == pytest.approx(1.0 + (1.3 - 1.0) * 0.5)
        assert weights["mean_reversion"] == pytest.approx(1.0 + (0.7 - 1.0) * 0.5)

    def test_at_threshold_returns_full_weights(self, detector):
        regime = MarketRegime(
            regime="bear", confidence=0.4, adx=25.0,
            volatility_pct=0.003, trend_direction="down",
        )
        weights = detector.get_strategy_weight_adjustments(regime)
        assert weights["mean_reversion"] == 1.3

    def test_unknown_regime_falls_back_to_sideways(self, detector):
        regime = MarketRegime(
            regime="unknown", confidence=0.8, adx=20.0,
            volatility_pct=0.002, trend_direction="flat",
        )
        weights = detector.get_strategy_weight_adjustments(regime)
        assert weights == {
            "trend_following": 0.6,
            "momentum": 1.0,
            "mean_reversion": 1.8,
            "breakout": 0.5,
            "range": 1.5,
        }


# ---------------------------------------------------------------------------
# Tests — MarketRegime dataclass
# ---------------------------------------------------------------------------

class TestMarketRegimeStr:
    def test_str_representation(self):
        r = MarketRegime(
            regime="bull", confidence=0.75, adx=32.5,
            volatility_pct=0.0045, trend_direction="up",
        )
        s = str(r)
        assert "BULL" in s
        assert "75%" in s
        assert "32.5" in s
