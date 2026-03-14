"""Tests for technical analysis indicators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    adx,
    atr,
    bollinger_bands,
    compute_all,
    ema,
    macd,
    obv,
    rsi,
    sma,
    stochastic,
    volume_sma,
)


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


@pytest.fixture
def constant_df():
    """DataFrame with constant price — tests boundary conditions."""
    n = 50
    return pd.DataFrame({
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
    })


@pytest.fixture
def tiny_df():
    """Minimal DataFrame — just a few rows."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 99.0],
        "high": [102.0, 103.0, 101.0],
        "low": [98.0, 99.0, 97.0],
        "close": [101.0, 100.0, 99.5],
        "volume": [1000.0, 1500.0, 800.0],
    })


# --- SMA ---


def test_sma(sample_df):
    result = sma(sample_df, period=20)
    assert len(result) == len(sample_df)
    assert result.iloc[:19].isna().all()  # First 19 should be NaN
    assert not result.iloc[19:].isna().any()  # Rest should have values


def test_sma_with_period_1(sample_df):
    """SMA with period 1 should equal the close price."""
    result = sma(sample_df, period=1)
    pd.testing.assert_series_equal(result, sample_df["close"], check_names=False)


def test_sma_constant_price(constant_df):
    """SMA of constant price should equal that price."""
    result = sma(constant_df, period=20)
    valid = result.dropna()
    assert (valid == 100.0).all()


def test_sma_small_df(tiny_df):
    """SMA with period longer than data should be all NaN."""
    result = sma(tiny_df, period=10)
    assert result.isna().all()


# --- EMA ---


def test_ema(sample_df):
    result = ema(sample_df, period=20)
    assert len(result) == len(sample_df)
    assert not result.isna().all()


def test_ema_constant_price(constant_df):
    """EMA of constant price should converge to that price."""
    result = ema(constant_df, period=20)
    assert result.iloc[-1] == pytest.approx(100.0, abs=0.01)


def test_ema_faster_reacts_more(sample_df):
    """Faster EMA should follow price more closely than slower one."""
    fast = ema(sample_df, period=5)
    slow = ema(sample_df, period=50)
    # Fast EMA should have higher variance (reacts more)
    assert fast.std() > slow.std()


# --- RSI ---


def test_rsi(sample_df):
    result = rsi(sample_df, period=14)
    assert len(result) == len(sample_df)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_all_gains():
    """RSI of purely rising prices should approach 100."""
    n = 50
    df = pd.DataFrame({"close": np.linspace(100, 150, n)})
    result = rsi(df, period=14)
    valid = result.dropna()
    assert valid.iloc[-1] > 90


def test_rsi_all_losses():
    """RSI of purely falling prices should approach 0."""
    n = 50
    df = pd.DataFrame({"close": np.linspace(150, 100, n)})
    result = rsi(df, period=14)
    valid = result.dropna()
    assert valid.iloc[-1] < 10


def test_rsi_constant_price(constant_df):
    """RSI of constant price produces NaN (0/0 division in RS calc)."""
    result = rsi(constant_df, period=14)
    # With no gains or losses, RSI is undefined (NaN)
    # or could be 50.0 depending on implementation. Either is acceptable.
    valid = result.dropna()
    if len(valid) > 0:
        assert (valid >= 0).all() and (valid <= 100).all()


# --- MACD ---


def test_macd(sample_df):
    result = macd(sample_df)
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_hist" in result.columns
    assert len(result) == len(sample_df)


def test_macd_histogram_is_difference(sample_df):
    """MACD histogram should equal MACD line minus signal line."""
    result = macd(sample_df)
    expected_hist = result["macd"] - result["macd_signal"]
    pd.testing.assert_series_equal(
        result["macd_hist"], expected_hist, check_names=False
    )


def test_macd_constant_price(constant_df):
    """MACD of constant price should converge to zero."""
    result = macd(constant_df)
    assert result["macd"].iloc[-1] == pytest.approx(0.0, abs=0.01)
    assert result["macd_hist"].iloc[-1] == pytest.approx(0.0, abs=0.01)


def test_macd_custom_params(sample_df):
    """MACD with custom parameters should not crash."""
    result = macd(sample_df, fast=8, slow=17, signal=5)
    assert len(result) == len(sample_df)


# --- Bollinger Bands ---


def test_bollinger_bands(sample_df):
    result = bollinger_bands(sample_df)
    assert "bb_upper" in result.columns
    assert "bb_middle" in result.columns
    assert "bb_lower" in result.columns
    valid_idx = result.dropna().index
    assert (result.loc[valid_idx, "bb_upper"] >= result.loc[valid_idx, "bb_middle"]).all()
    assert (result.loc[valid_idx, "bb_middle"] >= result.loc[valid_idx, "bb_lower"]).all()


def test_bollinger_bands_constant_price(constant_df):
    """Constant price should make upper == middle == lower (std=0)."""
    result = bollinger_bands(constant_df, period=20)
    valid_idx = result.dropna().index
    # When std is 0, upper and lower should equal middle
    assert (result.loc[valid_idx, "bb_upper"] == result.loc[valid_idx, "bb_middle"]).all()
    assert (result.loc[valid_idx, "bb_lower"] == result.loc[valid_idx, "bb_middle"]).all()


def test_bollinger_bands_wider_std(sample_df):
    """Wider std multiplier should produce wider bands."""
    narrow = bollinger_bands(sample_df, std=1.0)
    wide = bollinger_bands(sample_df, std=3.0)
    valid_idx = narrow.dropna().index
    narrow_width = narrow.loc[valid_idx, "bb_upper"] - narrow.loc[valid_idx, "bb_lower"]
    wide_width = wide.loc[valid_idx, "bb_upper"] - wide.loc[valid_idx, "bb_lower"]
    assert (wide_width >= narrow_width).all()


def test_bollinger_bands_middle_is_sma(sample_df):
    """BB middle should equal SMA of the close price."""
    result = bollinger_bands(sample_df, period=20)
    expected_sma = sma(sample_df, period=20)
    pd.testing.assert_series_equal(
        result["bb_middle"], expected_sma, check_names=False
    )


# --- ATR ---


def test_atr(sample_df):
    result = atr(sample_df, period=14)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()


def test_atr_constant_price(constant_df):
    """ATR of constant price should be zero (no true range)."""
    result = atr(constant_df, period=14)
    valid = result.dropna()
    assert valid.abs().max() < 0.01


def test_atr_high_volatility():
    """Higher volatility data should produce higher ATR."""
    np.random.seed(42)
    n = 50
    close_calm = 100 + np.cumsum(np.random.randn(n) * 0.1)
    close_wild = 100 + np.cumsum(np.random.randn(n) * 5.0)

    df_calm = pd.DataFrame({
        "open": close_calm, "high": close_calm + 0.2,
        "low": close_calm - 0.2, "close": close_calm,
    })
    df_wild = pd.DataFrame({
        "open": close_wild, "high": close_wild + 5.0,
        "low": close_wild - 5.0, "close": close_wild,
    })

    atr_calm = atr(df_calm, period=14).dropna().mean()
    atr_wild = atr(df_wild, period=14).dropna().mean()
    assert atr_wild > atr_calm


# --- ADX ---


def test_adx(sample_df):
    """ADX should produce values between 0 and 100 (approximately)."""
    result = adx(sample_df, period=14)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()


def test_adx_trending_vs_flat():
    """Strongly trending data should have higher ADX than flat data."""
    np.random.seed(42)
    n = 100

    # Strong uptrend
    close_trend = np.linspace(100, 150, n)
    df_trend = pd.DataFrame({
        "open": close_trend - 0.1, "high": close_trend + 0.5,
        "low": close_trend - 0.5, "close": close_trend,
    })

    # Flat / noisy
    close_flat = 100 + np.random.randn(n) * 0.1
    df_flat = pd.DataFrame({
        "open": close_flat - 0.1, "high": close_flat + 0.5,
        "low": close_flat - 0.5, "close": close_flat,
    })

    adx_trend = adx(df_trend, period=14).dropna().iloc[-1]
    adx_flat = adx(df_flat, period=14).dropna().iloc[-1]
    assert adx_trend > adx_flat


# --- Stochastic ---


def test_stochastic(sample_df):
    result = stochastic(sample_df)
    assert "stoch_k" in result.columns
    assert "stoch_d" in result.columns


def test_stochastic_range(sample_df):
    """Stochastic %K should be between 0 and 100."""
    result = stochastic(sample_df)
    valid_k = result["stoch_k"].dropna()
    assert (valid_k >= 0).all()
    assert (valid_k <= 100).all()


def test_stochastic_d_is_smoothed_k(sample_df):
    """Stochastic %D is the moving average of %K."""
    result = stochastic(sample_df, k_period=14, d_period=3)
    # %D should be smoother than %K
    valid_k = result["stoch_k"].dropna()
    valid_d = result["stoch_d"].dropna()
    assert valid_d.std() <= valid_k.std()


# --- OBV ---


def test_obv(sample_df):
    result = obv(sample_df)
    assert len(result) == len(sample_df)


def test_obv_direction():
    """OBV should increase on up days and decrease on down days."""
    df = pd.DataFrame({
        "close": [100, 105, 110, 108, 106],
        "volume": [1000, 2000, 3000, 2000, 1000],
    })
    result = obv(df)
    # Day 1 to 2: price up => OBV increases
    # Day 3 to 4: price down => OBV decreases
    assert result.iloc[2] > result.iloc[1]  # After 2 up days
    assert result.iloc[4] < result.iloc[3]  # After down day


# --- Volume SMA ---


def test_volume_sma(sample_df):
    result = volume_sma(sample_df, period=20)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()


def test_volume_sma_matches_sma(sample_df):
    """volume_sma should equal SMA applied to the volume column."""
    result = volume_sma(sample_df, period=20)
    expected = sample_df["volume"].rolling(window=20).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)


# --- compute_all ---


def test_compute_all(sample_df):
    result = compute_all(sample_df, {
        "ema_fast": 9,
        "ema_slow": 21,
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "volume_sma_period": 20,
    })
    expected_cols = [
        "ema_fast", "ema_slow", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "atr", "adx", "stoch_k", "stoch_d",
        "obv", "volume_sma", "volume_ratio",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_compute_all_preserves_original(sample_df):
    """compute_all should not modify the input DataFrame."""
    original = sample_df.copy()
    compute_all(sample_df, {})
    pd.testing.assert_frame_equal(sample_df, original)


def test_compute_all_uses_config_defaults(sample_df):
    """compute_all with empty config should use defaults without crashing."""
    result = compute_all(sample_df, {})
    assert "ema_fast" in result.columns
    assert "rsi" in result.columns
    assert "atr" in result.columns


def test_compute_all_volume_ratio(sample_df):
    """volume_ratio should be volume / volume_sma."""
    result = compute_all(sample_df, {"volume_sma_period": 20})
    valid_idx = result["volume_sma"].dropna().index
    expected = sample_df.loc[valid_idx, "volume"] / result.loc[valid_idx, "volume_sma"]
    pd.testing.assert_series_equal(
        result.loc[valid_idx, "volume_ratio"], expected, check_names=False
    )


def test_compute_all_with_tiny_data(tiny_df):
    """compute_all should not crash on tiny DataFrames (mostly NaN output)."""
    result = compute_all(tiny_df, {})
    assert len(result) == len(tiny_df)
    # Most indicators will be NaN, but it should not crash
    assert "rsi" in result.columns
