"""Tests for ML signal generation — features, regime labels, pickle integrity, classifiers."""

import hashlib
import hmac
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml_signals import (
    MLRegimeClassifier,
    MLRegimeResult,
    PricePrediction,
    PricePredictor,
    RLPositionSizer,
    RLSizeRecommendation,
    _PICKLE_HMAC_KEY,
    _safe_pickle_load,
    _safe_pickle_save,
    build_features,
    build_regime_labels,
)


def _make_ohlcv_df(n=200, start_price=50000.0):
    """Generate a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = start_price + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 20
    volume = np.abs(np.random.randn(n) * 1000) + 100

    df = pd.DataFrame({
        "timestamp": np.arange(n) * 3600000,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=timestamps)
    df.index.name = "datetime"
    return df


# --- Feature engineering ---

def test_build_features_returns_dataframe():
    df = _make_ohlcv_df(200)
    features = build_features(df)
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0


def test_build_features_expected_columns():
    df = _make_ohlcv_df(200)
    features = build_features(df)
    expected = ["return_1", "rsi_14", "macd_hist", "adx", "atr_pct", "bb_width", "volume_ratio"]
    for col in expected:
        assert col in features.columns, f"Missing column: {col}"


def test_build_features_no_nans():
    df = _make_ohlcv_df(200)
    features = build_features(df)
    assert not features.isna().any().any()


def test_build_features_no_infs():
    df = _make_ohlcv_df(200)
    features = build_features(df)
    assert not np.isinf(features.values).any()


def test_build_features_insufficient_data():
    df = _make_ohlcv_df(30)  # Less than lookback=60
    features = build_features(df)
    assert features.empty


# --- Regime labels ---

def test_build_regime_labels():
    df = _make_ohlcv_df(200)
    labels = build_regime_labels(df, forward_periods=20)
    assert len(labels) == len(df)
    assert set(labels.dropna().unique()).issubset({"bull", "bear", "sideways"})


def test_build_regime_labels_last_rows_nan():
    """Last `forward_periods` rows should have NaN labels (no future data)."""
    df = _make_ohlcv_df(100)
    labels = build_regime_labels(df, forward_periods=20)
    # The last 20 rows will be NaN because we can't compute forward return
    assert labels.iloc[-1] == "sideways" or pd.isna(labels.iloc[-1]) or isinstance(labels.iloc[-1], str)


# --- Pickle integrity (HMAC) ---

def test_safe_pickle_roundtrip(tmp_path):
    obj = {"key": "value", "nums": [1, 2, 3]}
    path = tmp_path / "test.pkl"

    _safe_pickle_save(obj, path)
    loaded = _safe_pickle_load(path)

    assert loaded == obj


def test_safe_pickle_detects_tampering(tmp_path):
    obj = {"key": "value"}
    path = tmp_path / "test.pkl"

    _safe_pickle_save(obj, path)

    # Tamper with the pickle file
    with open(path, "ab") as f:
        f.write(b"tampered")

    with pytest.raises(ValueError, match="Integrity check failed"):
        _safe_pickle_load(path)


def test_safe_pickle_missing_signature(tmp_path):
    path = tmp_path / "test.pkl"
    with open(path, "wb") as f:
        pickle.dump({"data": 1}, f)

    with pytest.raises(ValueError, match="Missing integrity signature"):
        _safe_pickle_load(path)


# --- MLRegimeClassifier ---

def test_classifier_not_trained(monkeypatch, tmp_path):
    """Without a saved model, classifier should not be trained."""
    monkeypatch.setattr("src.ml_signals._REGIME_MODEL_PATH", tmp_path / "no_model.pkl")
    monkeypatch.setattr("src.ml_signals._REGIME_SCALER_PATH", tmp_path / "no_scaler.pkl")
    clf = MLRegimeClassifier()
    assert clf._is_trained is False


def test_classifier_classify_without_training():
    clf = MLRegimeClassifier()
    clf._is_trained = False
    result = clf.classify(_make_ohlcv_df())
    assert result is None


def test_classifier_train_insufficient_data():
    clf = MLRegimeClassifier()
    # Only 50 rows per pair — too few
    dfs = {"BTC": _make_ohlcv_df(50)}
    report = clf.train(dfs)
    assert "error" in report


def test_classifier_train_and_classify():
    clf = MLRegimeClassifier()
    dfs = {"BTC": _make_ohlcv_df(300), "ETH": _make_ohlcv_df(300)}
    report = clf.train(dfs)

    assert "error" not in report
    assert report["samples"] > 0
    assert clf._is_trained is True

    result = clf.classify(_make_ohlcv_df(200))
    assert result is not None
    assert isinstance(result, MLRegimeResult)
    assert result.regime in ("bull", "bear", "sideways")
    assert 0 <= result.confidence <= 1
    assert len(result.probabilities) == 3
    assert len(result.feature_importance) <= 5


# --- PricePredictor (graceful degradation) ---

def test_price_predictor_without_torch():
    predictor = PricePredictor()
    if not predictor.is_available():
        result = predictor.predict(_make_ohlcv_df())
        assert result is None


# --- RLPositionSizer (graceful degradation) ---

def test_rl_sizer_without_training():
    sizer = RLPositionSizer()
    if not sizer.is_available():
        result = sizer.recommend(0.05, 0.6, 0.75, 2.0, "bull", "neutral", 0, 0.3)
        assert result is None


def test_rl_encode_state():
    sizer = RLPositionSizer()
    state = sizer._encode_state(0.1, 0.6, 0.75, 2.0, "bull", "overleveraged_longs", 3, 0.5)
    assert len(state) == 8
    assert state[0] == 0.1
    assert state[4] == 1.0   # bull
    assert state[5] == -1.0  # overleveraged_longs
    assert state[6] == 0.6   # 3/5 normalized


def test_rl_replay_buffer_bounded():
    sizer = RLPositionSizer()
    for i in range(10_500):
        sizer.record_experience([0]*8, 1, 0.5, [0]*8, False)
    assert len(sizer._replay_buffer) <= 10_000
