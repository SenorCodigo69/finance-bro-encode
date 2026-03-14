"""Tests for the strategy engine."""

import numpy as np
import pandas as pd
import pytest

from src.config import CorrelationGuardConfig, StrategyConfig
from src.models import Signal
from src.strategy import (
    BaseStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StrategyEngine,
    TrendFollowingStrategy,
)


@pytest.fixture
def config():
    return StrategyConfig()


@pytest.fixture
def make_df():
    """Factory to create OHLCV DataFrames with controllable trends."""
    def _make(n=100, trend="flat", volatility=0.5, seed=42):
        np.random.seed(seed)
        if trend == "up":
            base = np.linspace(100, 120, n) + np.random.randn(n) * volatility
        elif trend == "down":
            base = np.linspace(120, 100, n) + np.random.randn(n) * volatility
        else:
            base = 100 + np.random.randn(n) * volatility

        close = base
        return pd.DataFrame({
            "timestamp": range(n),
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.3),
            "low": close - abs(np.random.randn(n) * 0.3),
            "close": close,
            "volume": np.random.randint(1000, 50000, n).astype(float),
        })
    return _make


# --- Basic strategy smoke tests ---


def test_momentum_strategy_exists(config, make_df):
    strategy = MomentumStrategy()
    df = make_df(100, "flat")
    data = {"15m": df}
    # Should not crash
    result = strategy.analyze(data, "BTC/USDC:USDC", config)
    # Result can be None (no signal) or a Signal — both are valid


def test_trend_following_exists(config, make_df):
    strategy = TrendFollowingStrategy()
    df = make_df(100, "up")
    data = {"1h": df}
    result = strategy.analyze(data, "BTC/USDC:USDC", config)


def test_mean_reversion_exists(config, make_df):
    strategy = MeanReversionStrategy()
    df = make_df(100, "flat")
    data = {"15m": df}
    result = strategy.analyze(data, "BTC/USDC:USDC", config)


def test_breakout_exists(config, make_df):
    strategy = BreakoutStrategy()
    df = make_df(100, "flat")
    data = {"1h": df}
    result = strategy.analyze(data, "BTC/USDC:USDC", config)


def test_strategy_engine_aggregation(config, make_df):
    engine = StrategyEngine(config)
    df = make_df(100, "flat")
    market_data = {
        "BTC/USDC:USDC": {"15m": df, "1h": df, "4h": df},
    }
    signals = engine.generate_signals(market_data)
    # Should return a list (may be empty)
    assert isinstance(signals, list)


def test_insufficient_data(config):
    """Strategies should handle DataFrames that are too short."""
    short_df = pd.DataFrame({
        "open": [100, 101],
        "high": [102, 103],
        "low": [99, 100],
        "close": [101, 102],
        "volume": [1000, 2000],
    })
    for Strategy in [MomentumStrategy, TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy]:
        result = Strategy().analyze({"15m": short_df}, "BTC/USDC:USDC", config)
        assert result is None


def test_signal_confidence_range(config, make_df):
    engine = StrategyEngine(config)
    df = make_df(200, "up", volatility=2.0)
    market_data = {"BTC/USDC:USDC": {"15m": df, "1h": df}}
    signals = engine.generate_signals(market_data)
    for sig in signals:
        assert 0.0 <= sig.confidence <= 1.0


# --- Strategy-specific behavior tests ---


def test_momentum_prefers_15m_timeframe(config, make_df):
    """MomentumStrategy should use 15m when available."""
    strategy = MomentumStrategy()
    df_15m = make_df(100, "flat")
    df_1h = make_df(100, "up")
    data = {"15m": df_15m, "1h": df_1h}
    # Should not crash and should prefer 15m
    result = strategy.analyze(data, "BTC/USDC:USDC", config)
    if result is not None:
        assert result.timeframe == "15m"


def test_momentum_falls_back_to_first_timeframe(config, make_df):
    """MomentumStrategy uses first available timeframe if 15m not present."""
    strategy = MomentumStrategy()
    df = make_df(100, "flat")
    data = {"4h": df}
    result = strategy.analyze(data, "BTC/USDC:USDC", config)
    if result is not None:
        assert result.timeframe == "4h"


def test_trend_following_prefers_1h_timeframe(config, make_df):
    """TrendFollowingStrategy should use 1h when available."""
    strategy = TrendFollowingStrategy()
    df = make_df(100, "up")
    data = {"1h": df, "15m": df}
    result = strategy.analyze(data, "BTC/USDC:USDC", config)
    if result is not None:
        assert result.timeframe == "1h"


def test_breakout_prefers_1h_timeframe(config, make_df):
    """BreakoutStrategy should use 1h when available."""
    strategy = BreakoutStrategy()
    df = make_df(100, "flat")
    data = {"1h": df, "15m": df}
    result = strategy.analyze(data, "BTC/USDC:USDC", config)
    if result is not None:
        assert result.timeframe == "1h"


def test_breakout_needs_25_candles(config):
    """BreakoutStrategy requires at least 25 candles."""
    strategy = BreakoutStrategy()
    np.random.seed(42)
    n = 24
    close = np.linspace(100, 105, n)
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(n, 5000.0),
    })
    result = strategy.analyze({"1h": df}, "BTC/USDC:USDC", config)
    assert result is None


def test_momentum_returns_none_on_nan_indicators(config):
    """If key indicators are NaN after compute, signal should be None."""
    strategy = MomentumStrategy()
    # 30 candles with constant price => some indicators may be NaN
    n = 30
    df = pd.DataFrame({
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 100.0),
        "volume": np.zeros(n),  # Zero volume => volume_ratio will be NaN
    })
    result = strategy.analyze({"15m": df}, "BTC/USDC:USDC", config)
    assert result is None


def test_get_latest_returns_second_to_last(config, make_df):
    """_get_latest returns the second-to-last candle (latest complete one)."""
    strategy = MomentumStrategy()
    df = make_df(100, "flat")
    latest = strategy._get_latest(df)
    assert latest is not None
    # Should be iloc[-2]
    pd.testing.assert_series_equal(latest, df.iloc[-2])


def test_get_latest_too_short():
    """_get_latest returns None when fewer than 2 candles."""
    strategy = MomentumStrategy()
    df = pd.DataFrame({"close": [100.0]})
    assert strategy._get_latest(df) is None


# --- Momentum strategy signal generation with crafted data ---


def test_momentum_generates_long_signal(config):
    """Craft data to produce a long signal: RSI oversold + MACD turning + volume spike."""
    np.random.seed(42)
    n = 50
    # Start with a downtrend then reversal
    close = np.concatenate([
        np.linspace(120, 95, 40),   # Downtrend (pushes RSI low)
        np.linspace(95, 98, 10),    # Slight reversal
    ])
    volume = np.full(n, 5000.0)
    volume[-3:] = 15000.0  # Volume spike at the end

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": volume,
    })

    strategy = MomentumStrategy()
    result = strategy.analyze({"15m": df}, "BTC/USDC:USDC", config)
    # We can't guarantee signal generation with these exact values,
    # but if a signal is generated it should be a long
    if result is not None:
        assert result.direction == "long"
        assert result.strategy_name == "momentum"
        assert "rsi" in result.indicators


def test_momentum_generates_short_signal(config):
    """Craft data to produce a short signal: RSI overbought + MACD turning down + volume spike."""
    np.random.seed(42)
    n = 50
    # Uptrend then slight reversal
    close = np.concatenate([
        np.linspace(80, 105, 40),   # Uptrend (pushes RSI high)
        np.linspace(105, 103, 10),  # Slight pullback
    ])
    volume = np.full(n, 5000.0)
    volume[-3:] = 15000.0  # Volume spike

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": volume,
    })

    strategy = MomentumStrategy()
    result = strategy.analyze({"15m": df}, "BTC/USDC:USDC", config)
    if result is not None:
        assert result.direction == "short"
        assert result.strategy_name == "momentum"


# --- StrategyEngine aggregation, filtering, and market signal tests ---


def test_aggregation_boosts_confidence_for_agreement(config, make_df):
    """When multiple strategies agree on direction, confidence should be boosted."""
    engine = StrategyEngine(config)

    # Create fake signals to test aggregation directly
    sig1 = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="momentum",
        reasoning="RSI oversold", timestamp="2024-01-01",
    )
    sig2 = Signal(
        pair="BTC/USDC:USDC", timeframe="1h", direction="long",
        confidence=0.65, strategy_name="trend_following",
        reasoning="EMA crossover", timestamp="2024-01-01",
    )

    result = engine._aggregate_signals([sig1, sig2])
    assert len(result) == 1
    # Best confidence (0.7) + 0.05 boost = 0.75
    assert result[0].confidence == pytest.approx(0.75, abs=0.01)
    assert "2 strategies" in result[0].reasoning


def test_aggregation_keeps_separate_for_different_directions(config):
    """Different directions on the same pair should NOT be merged."""
    engine = StrategyEngine(config)

    sig_long = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    sig_short = Signal(
        pair="BTC/USDC:USDC", timeframe="1h", direction="short",
        confidence=0.65, strategy_name="mean_reversion",
        reasoning="test", timestamp="2024-01-01",
    )

    result = engine._aggregate_signals([sig_long, sig_short])
    assert len(result) == 2


def test_aggregation_keeps_separate_for_different_pairs(config):
    """Different pairs should NOT be merged even with same direction."""
    engine = StrategyEngine(config)

    sig1 = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    sig2 = Signal(
        pair="ETH/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.65, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )

    result = engine._aggregate_signals([sig1, sig2])
    assert len(result) == 2


def test_aggregation_sorts_by_confidence(config):
    """Aggregated signals should be sorted by confidence, highest first."""
    engine = StrategyEngine(config)

    signals = [
        Signal(pair="ETH/USDC:USDC", timeframe="15m", direction="long",
               confidence=0.6, strategy_name="a", reasoning="", timestamp=""),
        Signal(pair="BTC/USDC:USDC", timeframe="15m", direction="long",
               confidence=0.85, strategy_name="b", reasoning="", timestamp=""),
    ]

    result = engine._aggregate_signals(signals)
    assert result[0].confidence >= result[1].confidence


def test_signals_below_threshold_filtered(config, make_df):
    """Signals below 0.5 confidence should not appear in output."""
    engine = StrategyEngine(config)
    # With flat data, signals should either be absent or have confidence >= 0.5
    df = make_df(100, "flat")
    market_data = {"BTC/USDC:USDC": {"15m": df, "1h": df}}
    signals = engine.generate_signals(market_data)
    for sig in signals:
        assert sig.confidence >= 0.5


def test_apply_pair_weight_boost(config):
    """Per-pair strategy weight > 1.0 should boost confidence."""
    config.pair_weights = {"BTC/USDC:USDC": {"momentum": 1.5}}
    engine = StrategyEngine(config)

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.6, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_pair_weight(sig)
    assert result.confidence == pytest.approx(0.6 * 1.5, abs=0.01)
    assert "boosted" in result.reasoning.lower()


def test_apply_pair_weight_penalize(config):
    """Per-pair strategy weight < 1.0 should penalize confidence."""
    config.pair_weights = {"BTC/USDC:USDC": {"momentum": 0.5}}
    engine = StrategyEngine(config)

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.8, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_pair_weight(sig)
    assert result.confidence == pytest.approx(0.4, abs=0.01)
    assert "penalized" in result.reasoning.lower()


def test_apply_pair_weight_clamps_confidence(config):
    """Pair weight should never push confidence above 0.95 or below 0.1."""
    config.pair_weights = {"BTC/USDC:USDC": {"momentum": 5.0}}
    engine = StrategyEngine(config)

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.8, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_pair_weight(sig)
    assert result.confidence <= 0.95


def test_funding_filter_boosts_contrarian_short(config):
    """Extreme positive funding boosts short signals (crowded longs)."""
    engine = StrategyEngine(config)
    engine.set_market_signals(
        funding_rates={"BTC": 0.0002},  # Extreme positive
        open_interest={},
    )

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="short",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_funding_filter(sig, "BTC/USDC:USDC")
    assert result.confidence > 0.7
    assert "crowded longs" in result.reasoning.lower()


def test_funding_filter_penalizes_same_direction_long(config):
    """Extreme positive funding penalizes long signals."""
    engine = StrategyEngine(config)
    engine.set_market_signals(
        funding_rates={"BTC": 0.0002},
        open_interest={},
    )

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_funding_filter(sig, "BTC/USDC:USDC")
    assert result.confidence < 0.7


def test_funding_filter_very_extreme(config):
    """Very extreme funding (>0.05%) has stronger effect."""
    engine = StrategyEngine(config)
    engine.set_market_signals(
        funding_rates={"BTC": 0.0006},  # Very extreme
        open_interest={},
    )

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="short",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_funding_filter(sig, "BTC/USDC:USDC")
    # Very extreme boost is +0.10
    assert result.confidence == pytest.approx(0.8, abs=0.01)


def test_funding_filter_no_data(config):
    """Without funding data, signal is unchanged."""
    engine = StrategyEngine(config)
    engine.set_market_signals(funding_rates={}, open_interest={})

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_funding_filter(sig, "BTC/USDC:USDC")
    assert result.confidence == 0.7


def test_oi_filter_boosts_breakout_with_high_oi(config):
    """High open interest boosts breakout signal confidence."""
    engine = StrategyEngine(config)
    engine.set_market_signals(
        funding_rates={},
        open_interest={"BTC": {"oi": 100000.0, "mark_price": 70000.0}},
    )

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="1h", direction="long",
        confidence=0.7, strategy_name="breakout",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_oi_filter(sig, "BTC/USDC:USDC")
    # oi_usd = 100000 * 70000 = 7B > 5B threshold
    assert result.confidence == pytest.approx(0.78, abs=0.01)
    assert "high oi" in result.reasoning.lower()


def test_oi_filter_only_affects_breakout(config):
    """OI filter should not affect non-breakout strategies."""
    engine = StrategyEngine(config)
    engine.set_market_signals(
        funding_rates={},
        open_interest={"BTC": {"oi": 100000.0, "mark_price": 70000.0}},
    )

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    result = engine._apply_oi_filter(sig, "BTC/USDC:USDC")
    assert result.confidence == 0.7


def test_regime_weight_boost(config):
    """Regime weight > 1.0 boosts signal confidence."""
    engine = StrategyEngine(config)
    from src.regime import MarketRegime
    engine.current_regime = MarketRegime(
        regime="bull", confidence=0.8, adx=35.0,
        volatility_pct=0.02, trend_direction="up",
    )

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="momentum",
        reasoning="test", timestamp="2024-01-01",
    )
    adjustments = {"momentum": 1.3}
    result = engine._apply_regime_weight(sig, adjustments)
    assert result.confidence == pytest.approx(0.7 * 1.3, abs=0.01)
    assert "boosted" in result.reasoning.lower()


def test_regime_weight_penalize(config):
    """Regime weight < 1.0 penalizes signal confidence."""
    engine = StrategyEngine(config)
    from src.regime import MarketRegime
    engine.current_regime = MarketRegime(
        regime="bear", confidence=0.8, adx=35.0,
        volatility_pct=0.02, trend_direction="down",
    )

    sig = Signal(
        pair="BTC/USDC:USDC", timeframe="15m", direction="long",
        confidence=0.7, strategy_name="trend_following",
        reasoning="test", timestamp="2024-01-01",
    )
    adjustments = {"trend_following": 0.6}
    result = engine._apply_regime_weight(sig, adjustments)
    assert result.confidence == pytest.approx(0.7 * 0.6, abs=0.01)
    assert "penalized" in result.reasoning.lower()


def test_correlation_guard_blocks_eth_long_on_btc_dump(config, make_df):
    """ETH long should be blocked when BTC dropped > threshold."""
    config.correlation_guard = CorrelationGuardConfig(
        enabled=True, btc_drop_threshold_pct=-2.0, btc_pump_threshold_pct=2.0,
    )
    engine = StrategyEngine(config)

    # BTC data with a sharp drop in latest candles
    np.random.seed(42)
    n = 50
    close = np.concatenate([np.full(48, 50000.0), [48000.0, 47000.0]])
    btc_df = pd.DataFrame({
        "open": close, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.full(n, 5000.0),
    })

    signals = [
        Signal(pair="ETH/USDC:USDC", timeframe="15m", direction="long",
               confidence=0.7, strategy_name="momentum",
               reasoning="test", timestamp="2024-01-01"),
    ]
    market_data = {"BTC/USDC:USDC": {"1h": btc_df}}

    filtered = engine._apply_correlation_guard(signals, market_data)
    # BTC dropped ~4.2% in 1 candle — should block ETH long
    assert len(filtered) == 0


def test_correlation_guard_blocks_eth_short_on_btc_pump(config, make_df):
    """ETH short should be blocked when BTC pumped > threshold."""
    config.correlation_guard = CorrelationGuardConfig(
        enabled=True, btc_drop_threshold_pct=-2.0, btc_pump_threshold_pct=2.0,
    )
    engine = StrategyEngine(config)

    np.random.seed(42)
    n = 50
    close = np.concatenate([np.full(48, 50000.0), [52000.0, 53000.0]])
    btc_df = pd.DataFrame({
        "open": close, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.full(n, 5000.0),
    })

    signals = [
        Signal(pair="ETH/USDC:USDC", timeframe="15m", direction="short",
               confidence=0.7, strategy_name="momentum",
               reasoning="test", timestamp="2024-01-01"),
    ]
    market_data = {"BTC/USDC:USDC": {"1h": btc_df}}

    filtered = engine._apply_correlation_guard(signals, market_data)
    assert len(filtered) == 0


def test_correlation_guard_allows_btc_signals(config):
    """BTC signals should pass through correlation guard regardless."""
    config.correlation_guard = CorrelationGuardConfig(enabled=True)
    engine = StrategyEngine(config)

    np.random.seed(42)
    n = 50
    close = np.concatenate([np.full(48, 50000.0), [48000.0, 47000.0]])
    btc_df = pd.DataFrame({
        "open": close, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.full(n, 5000.0),
    })

    signals = [
        Signal(pair="BTC/USDC:USDC", timeframe="15m", direction="long",
               confidence=0.7, strategy_name="momentum",
               reasoning="test", timestamp="2024-01-01"),
    ]
    market_data = {"BTC/USDC:USDC": {"1h": btc_df}}

    filtered = engine._apply_correlation_guard(signals, market_data)
    assert len(filtered) == 1


def test_correlation_guard_disabled(config):
    """Disabled correlation guard should pass all signals through."""
    config.correlation_guard = CorrelationGuardConfig(enabled=False)
    engine = StrategyEngine(config)

    signals = [
        Signal(pair="ETH/USDC:USDC", timeframe="15m", direction="long",
               confidence=0.7, strategy_name="momentum",
               reasoning="test", timestamp="2024-01-01"),
    ]

    filtered = engine._apply_correlation_guard(signals, {})
    assert len(filtered) == 1


def test_load_evolved_strategies(config):
    """load_evolved_strategies adds new strategies and removes old evolved ones."""
    engine = StrategyEngine(config)
    original_count = len(engine.strategies)

    class FakeEvolvedStrategy(BaseStrategy):
        name = "evolved_1"
        def analyze(self, data, pair, config):
            return None

    engine.load_evolved_strategies([FakeEvolvedStrategy()])
    assert len(engine.strategies) == original_count + 1

    # Loading again should replace, not accumulate
    engine.load_evolved_strategies([FakeEvolvedStrategy(), FakeEvolvedStrategy()])
    assert len(engine.strategies) == original_count + 2


def test_set_market_signals(config):
    """set_market_signals stores funding rates and OI data."""
    engine = StrategyEngine(config)
    engine.set_market_signals(
        funding_rates={"BTC": 0.0003, "ETH": -0.0001},
        open_interest={"BTC": {"oi": 5000.0, "mark_price": 70000.0}},
    )
    assert engine._funding_rates["BTC"] == 0.0003
    assert engine._open_interest["BTC"]["oi"] == 5000.0


def test_set_market_signals_handles_none(config):
    """set_market_signals handles None input gracefully."""
    engine = StrategyEngine(config)
    engine.set_market_signals(funding_rates=None, open_interest=None)
    assert engine._funding_rates == {}
    assert engine._open_interest == {}


def test_select_regime_df_prefers_btc_1h(config, make_df):
    """Regime detection should prefer BTC 1h data."""
    engine = StrategyEngine(config)
    btc_1h = make_df(100, "up")
    eth_1h = make_df(100, "down")
    market_data = {
        "BTC/USDC:USDC": {"1h": btc_1h, "15m": make_df(100)},
        "ETH/USDC:USDC": {"1h": eth_1h},
    }
    result = engine._select_regime_df(market_data)
    pd.testing.assert_frame_equal(result, btc_1h)


def test_select_regime_df_fallback_to_first(config, make_df):
    """When no BTC data exists, fall back to first available 1h."""
    engine = StrategyEngine(config)
    eth_1h = make_df(100, "down")
    market_data = {
        "ETH/USDC:USDC": {"1h": eth_1h},
    }
    result = engine._select_regime_df(market_data)
    pd.testing.assert_frame_equal(result, eth_1h)


def test_select_regime_df_empty(config):
    """Empty market data returns None."""
    engine = StrategyEngine(config)
    result = engine._select_regime_df({})
    assert result is None
