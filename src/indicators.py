"""Technical analysis indicators — pure functions operating on DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    return df[col].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "close"
) -> pd.DataFrame:
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0, col: str = "close") -> pd.DataFrame:
    middle = df[col].rolling(window=period).mean()
    rolling_std = df[col].rolling(window=period).std()
    upper = middle + (rolling_std * std)
    lower = middle - (rolling_std * std)
    return pd.DataFrame({"bb_upper": upper, "bb_middle": middle, "bb_lower": lower})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_vals = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_vals)
    minus_di = 100 * (minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_vals)

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    return dx.ewm(com=period - 1, min_periods=period).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff())
    return (direction * df["volume"]).cumsum()


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(window=period).mean()


def compute_all(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach all configured indicators to the DataFrame."""
    result = df.copy()

    # Trend
    result["ema_fast"] = ema(df, config.get("ema_fast", 9))
    result["ema_slow"] = ema(df, config.get("ema_slow", 21))
    result["sma_20"] = sma(df, 20)

    # Momentum
    result["rsi"] = rsi(df, config.get("rsi_period", 14))
    macd_df = macd(df, config.get("macd_fast", 12), config.get("macd_slow", 26), config.get("macd_signal", 9))
    result = pd.concat([result, macd_df], axis=1)

    # Volatility
    bb_df = bollinger_bands(df, config.get("bb_period", 20), config.get("bb_std", 2.0))
    result = pd.concat([result, bb_df], axis=1)
    result["atr"] = atr(df, config.get("atr_period", 14))

    # Trend strength
    result["adx"] = adx(df, 14)

    # Stochastic
    stoch_df = stochastic(df)
    result = pd.concat([result, stoch_df], axis=1)

    # Volume
    result["obv"] = obv(df)
    result["volume_sma"] = volume_sma(df, config.get("volume_sma_period", 20))
    result["volume_ratio"] = df["volume"] / result["volume_sma"].replace(0, float("nan"))

    return result
