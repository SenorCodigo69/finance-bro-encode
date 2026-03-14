"""ML signal generation — regime classification, price prediction, RL position sizing.

Phase 4 implementation:
  1. MLRegimeClassifier — Random Forest replacing rule-based regime detection
  2. PricePredictor — LSTM-based price direction prediction (requires torch)
  3. RLPositionSizer — DQN-based position sizing (requires torch)

Models 2 and 3 gracefully degrade to no-op when torch is not installed.
Model 1 uses scikit-learn (lightweight, always available).
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.indicators import adx, atr, ema, sma, rsi, macd, bollinger_bands, stochastic
from src.regime import MarketRegime
from src.utils import log

# Optional heavy imports — graceful degradation
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Feature engineering (shared across all ML models)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Build ML feature matrix from raw OHLCV data.

    Features:
    - Price-based: returns (1,5,10,20 period), log returns, close/SMA ratios
    - Momentum: RSI, MACD histogram, Stochastic %K/%D
    - Trend: ADX, EMA slope (9-period), SMA 20/50 crossover distance
    - Volatility: ATR%, Bollinger Band width, BB position
    - Volume: volume ratio, OBV slope
    - Regime hints: rolling mean of returns, rolling std
    """
    if len(df) < lookback:
        return pd.DataFrame()

    features = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Price returns ---
    for period in [1, 5, 10, 20]:
        features[f"return_{period}"] = close.pct_change(period)

    features["log_return_1"] = np.log(close / close.shift(1))

    # --- SMA ratios ---
    sma20 = sma(df, 20)
    sma50 = sma(df, 50)
    features["close_sma20_ratio"] = close / sma20
    features["close_sma50_ratio"] = close / sma50
    features["sma20_sma50_ratio"] = sma20 / sma50

    # --- Momentum ---
    features["rsi_14"] = rsi(df, 14)
    macd_df = macd(df, 12, 26, 9)
    features["macd_hist"] = macd_df["macd_hist"]
    features["macd_hist_slope"] = macd_df["macd_hist"] - macd_df["macd_hist"].shift(1)
    stoch_df = stochastic(df, 14)
    features["stoch_k"] = stoch_df["stoch_k"]
    features["stoch_d"] = stoch_df["stoch_d"]

    # --- Trend ---
    adx_series = adx(df, 14)
    features["adx"] = adx_series
    ema9 = ema(df, 9)
    features["ema9_slope"] = (ema9 - ema9.shift(5)) / ema9.shift(5)

    # --- Volatility ---
    atr_series = atr(df, 14)
    features["atr_pct"] = atr_series / close
    bb_df = bollinger_bands(df, 20, 2.0)
    bb_width = (bb_df["bb_upper"] - bb_df["bb_lower"]) / bb_df["bb_middle"]
    features["bb_width"] = bb_width
    features["bb_position"] = (close - bb_df["bb_lower"]) / (bb_df["bb_upper"] - bb_df["bb_lower"])

    # --- Volume ---
    vol_sma = volume.rolling(20).mean()
    features["volume_ratio"] = volume / vol_sma

    # --- Rolling stats (regime indicators) ---
    features["rolling_mean_20"] = close.pct_change(1).rolling(20).mean()
    features["rolling_std_20"] = close.pct_change(1).rolling(20).std()
    features["rolling_skew_20"] = close.pct_change(1).rolling(20).skew()

    # --- Candle patterns ---
    body = abs(close - df["open"])
    wick = high - low
    features["body_ratio"] = body / wick.replace(0, np.nan)

    # [S5-H4] Replace inf values and drop NaN rows from indicator warmup
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    return features


def build_regime_labels(df: pd.DataFrame, forward_periods: int = 20) -> pd.Series:
    """Generate regime labels from future price movement.

    Uses forward-looking returns to classify:
    - "bull": forward return > +1%
    - "bear": forward return < -1%
    - "sideways": forward return between -1% and +1%
    """
    forward_return = df["close"].pct_change(forward_periods).shift(-forward_periods)

    labels = pd.Series("sideways", index=df.index)
    labels[forward_return > 0.01] = "bull"
    labels[forward_return < -0.01] = "bear"

    return labels


# ---------------------------------------------------------------------------
# 1. ML Regime Classifier (scikit-learn)
# ---------------------------------------------------------------------------

_REGIME_MODEL_PATH = Path("data/models/regime_classifier.pkl")
_REGIME_SCALER_PATH = Path("data/models/regime_scaler.pkl")

# [S5-C1] HMAC key for pickle integrity verification.
# This prevents loading tampered model files. The key is derived from the
# project path — not a secret, but ensures only this codebase's own saves
# are loaded.
_PICKLE_HMAC_KEY = hashlib.sha256(b"finance_agent_model_integrity_v1").digest()


def _safe_pickle_save(obj: Any, path: Path) -> None:
    """Save a pickle file with an HMAC integrity signature."""
    data = pickle.dumps(obj)
    sig = _hmac.new(_PICKLE_HMAC_KEY, data, hashlib.sha256).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    with open(str(path) + ".sig", "w") as f:
        f.write(sig)


def _safe_pickle_load(path: Path) -> Any:
    """Load a pickle file only if its HMAC signature is valid.

    Raises ValueError if the signature is missing or doesn't match.
    """
    sig_path = Path(str(path) + ".sig")
    if not sig_path.exists():
        raise ValueError(f"Missing integrity signature for {path}")

    with open(path, "rb") as f:
        data = f.read()
    with open(sig_path, "r") as f:
        expected_sig = f.read().strip()

    actual_sig = _hmac.new(_PICKLE_HMAC_KEY, data, hashlib.sha256).hexdigest()
    if not _hmac.compare_digest(actual_sig, expected_sig):
        raise ValueError(f"Integrity check failed for {path} — file may be tampered")

    return pickle.loads(data)  # noqa: S301 — verified by HMAC above


@dataclass
class MLRegimeResult:
    """Result from ML regime classification."""
    regime: str                     # "bull" | "bear" | "sideways"
    confidence: float               # 0.0 – 1.0
    probabilities: dict[str, float] # {"bull": 0.4, "bear": 0.1, "sideways": 0.5}
    feature_importance: dict[str, float]  # Top 5 features
    model_type: str = "ml"          # "ml" vs "rule_based" for tracking


class MLRegimeClassifier:
    """Random Forest regime classifier trained on historical OHLCV data.

    Replaces rule-based RegimeDetector when trained. Falls back to rule-based
    when no trained model exists or when scikit-learn is not installed.
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []
        self._is_trained = False

        if HAS_SKLEARN:
            self._load_model()

    def is_available(self) -> bool:
        return HAS_SKLEARN and self._is_trained

    def classify(self, df: pd.DataFrame) -> MLRegimeResult | None:
        """Classify current regime from OHLCV data.

        Returns None if model is not trained or data is insufficient.
        """
        if not self.is_available():
            return None

        features = build_features(df)
        if features.empty or len(features) < 1:
            return None

        # Use the latest row
        X = features.iloc[[-1]][self._feature_names]
        X_scaled = self._scaler.transform(X)

        # Predict
        regime = self._model.predict(X_scaled)[0]
        probas = self._model.predict_proba(X_scaled)[0]
        classes = self._model.classes_

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}
        confidence = max(probas)

        # Feature importance (top 5)
        importances = dict(zip(
            self._feature_names,
            self._model.feature_importances_
        ))
        top5 = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5])

        return MLRegimeResult(
            regime=regime,
            confidence=round(float(confidence), 4),
            probabilities=prob_dict,
            feature_importance={k: round(v, 4) for k, v in top5.items()},
        )

    def train(
        self,
        dfs: dict[str, pd.DataFrame],
        forward_periods: int = 20,
        n_estimators: int = 200,
    ) -> dict[str, Any]:
        """Train the regime classifier on historical data from multiple pairs.

        Args:
            dfs: Mapping of pair name → OHLCV DataFrame (at least 200 rows each).
            forward_periods: How many candles ahead to look for labeling.
            n_estimators: Number of trees in the Random Forest.

        Returns:
            Training report dict with accuracy, class distribution, etc.
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed"}

        all_features = []
        all_labels = []

        for pair, df in dfs.items():
            if len(df) < 200:
                log.debug(f"Skipping {pair} for training (only {len(df)} rows)")
                continue

            features = build_features(df)
            labels = build_regime_labels(df, forward_periods)

            # Align features and labels
            common_idx = features.index.intersection(labels.dropna().index)
            if len(common_idx) < 50:
                continue

            all_features.append(features.loc[common_idx])
            all_labels.append(labels.loc[common_idx])

        if not all_features:
            return {"error": "Insufficient data for training"}

        X = pd.concat(all_features)
        y = pd.concat(all_labels)

        self._feature_names = list(X.columns)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            score = accuracy_score(y_val, rf.predict(X_val))
            cv_scores.append(score)

        # Final model on all data
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_scaled, y)
        self._is_trained = True

        # Save model
        self._save_model()

        # Feature importance
        importances = dict(zip(self._feature_names, self._model.feature_importances_))
        top10 = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

        report = {
            "samples": len(X),
            "features": len(self._feature_names),
            "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
            "class_distribution": y.value_counts().to_dict(),
            "top_features": {k: round(v, 4) for k, v in top10.items()},
            "pairs_used": list(dfs.keys()),
        }

        log.info(
            f"MLRegimeClassifier trained: {report['samples']} samples, "
            f"CV accuracy {report['cv_accuracy_mean']:.1%} ± {report['cv_accuracy_std']:.1%}"
        )

        return report

    def _save_model(self) -> None:
        # [S5-C1] Use HMAC-verified pickle to prevent tampered model loading
        _safe_pickle_save(self._model, _REGIME_MODEL_PATH)
        _safe_pickle_save((self._scaler, self._feature_names), _REGIME_SCALER_PATH)
        log.debug(f"Saved regime model to {_REGIME_MODEL_PATH}")

    def _load_model(self) -> None:
        if not _REGIME_MODEL_PATH.exists() or not _REGIME_SCALER_PATH.exists():
            return
        try:
            # [S5-C1] Verify HMAC signature before loading
            self._model = _safe_pickle_load(_REGIME_MODEL_PATH)
            self._scaler, self._feature_names = _safe_pickle_load(_REGIME_SCALER_PATH)
            self._is_trained = True
            log.info("Loaded trained ML regime classifier")
        except Exception as e:
            log.warning(f"Failed to load regime model: {e}")
            self._is_trained = False


# ---------------------------------------------------------------------------
# 2. LSTM Price Predictor (PyTorch — optional)
# ---------------------------------------------------------------------------

_LSTM_MODEL_PATH = Path("data/models/price_lstm.pt")


@dataclass
class PricePrediction:
    """Result from LSTM price prediction."""
    direction: str          # "up" | "down" | "flat"
    confidence: float       # 0.0 – 1.0
    predicted_return: float  # Expected % return next N candles
    model_type: str = "lstm"


class PricePredictor:
    """LSTM-based price direction predictor.

    Architecture: Feature extraction → LSTM → FC → Softmax
    Predicts probability of up/down/flat over the next N candles.

    Requires PyTorch. Returns None when torch is not installed.
    """

    def __init__(self, input_size: int = 24, hidden_size: int = 64, num_layers: int = 2):
        self._is_trained = False
        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []

        if not HAS_TORCH:
            log.debug("PyTorch not installed — PricePredictor disabled")
            return

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._load_model()

    def is_available(self) -> bool:
        return HAS_TORCH and self._is_trained

    def predict(self, df: pd.DataFrame, sequence_length: int = 30) -> PricePrediction | None:
        """Predict price direction from recent OHLCV data.

        Args:
            df: OHLCV DataFrame with at least sequence_length + 60 rows.
            sequence_length: Number of candles in the input sequence.
        """
        if not self.is_available():
            return None

        features = build_features(df)
        if len(features) < sequence_length:
            return None

        # Take last sequence_length rows
        X = features.iloc[-sequence_length:][self._feature_names].values
        X_scaled = self._scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)  # (1, seq_len, features)

        self._model.eval()
        with torch.no_grad():
            output = self._model(X_tensor)
            probas = torch.softmax(output, dim=1).squeeze().numpy()

        classes = ["down", "flat", "up"]
        pred_idx = int(np.argmax(probas))
        direction = classes[pred_idx]
        confidence = float(probas[pred_idx])

        # Estimate return magnitude from class probabilities
        predicted_return = float(probas[2] - probas[0]) * 0.02  # Rough scaling

        return PricePrediction(
            direction=direction,
            confidence=round(confidence, 4),
            predicted_return=round(predicted_return, 6),
        )

    def train(
        self,
        dfs: dict[str, pd.DataFrame],
        sequence_length: int = 30,
        forward_periods: int = 10,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> dict[str, Any]:
        """Train the LSTM on historical data."""
        if not HAS_TORCH:
            return {"error": "PyTorch not installed"}

        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed (needed for scaling)"}

        all_X = []
        all_y = []

        for pair, df in dfs.items():
            if len(df) < 200:
                continue

            features = build_features(df)
            if features.empty:
                continue

            # Labels: forward return classification
            close = df["close"].reindex(features.index)
            forward_ret = close.pct_change(forward_periods).shift(-forward_periods)

            # Classify: 0=down, 1=flat, 2=up
            labels = pd.Series(1, index=features.index)  # flat
            labels[forward_ret > 0.005] = 2   # up
            labels[forward_ret < -0.005] = 0  # down

            # Remove NaN
            valid = features.index.intersection(labels.dropna().index)
            features = features.loc[valid]
            labels = labels.loc[valid]

            if len(features) < sequence_length + 10:
                continue

            # Create sequences
            self._feature_names = list(features.columns)
            vals = features.values
            labs = labels.values

            for i in range(sequence_length, len(vals)):
                all_X.append(vals[i - sequence_length:i])
                all_y.append(labs[i])

        if not all_X:
            return {"error": "Insufficient data"}

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)

        # Scale
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        self._scaler = StandardScaler()
        X_flat_scaled = self._scaler.fit_transform(X_flat)
        X = X_flat_scaled.reshape(n_samples, seq_len, n_features)

        # Train/val split (time-aware: last 20% for validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)

        # Build model
        self._input_size = n_features
        self._model = _LSTMModel(n_features, self._hidden_size, self._num_layers, 3)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for epoch in range(epochs):
            self._model.train()
            optimizer.zero_grad()
            output = self._model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_out = self._model(X_val_t)
                val_preds = val_out.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 10 == 0:
                log.debug(
                    f"LSTM epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, "
                    f"val_acc={val_acc:.1%}"
                )

        self._is_trained = True
        self._save_model()

        return {
            "samples": len(X),
            "sequence_length": sequence_length,
            "features": n_features,
            "epochs": epochs,
            "best_val_accuracy": round(best_val_acc, 4),
            "class_distribution": {
                "down": int((y == 0).sum()),
                "flat": int((y == 1).sum()),
                "up": int((y == 2).sum()),
            },
        }

    def _save_model(self) -> None:
        if not HAS_TORCH or self._model is None:
            return
        _LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # [S5-C2] Save only model weights (safe), metadata in separate JSON
        torch.save(self._model.state_dict(), _LSTM_MODEL_PATH)
        meta_path = _LSTM_MODEL_PATH.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "input_size": self._input_size,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
                "feature_names": self._feature_names,
            }, f)
        # Save scaler with HMAC verification
        if self._scaler is not None:
            scaler_path = _LSTM_MODEL_PATH.with_suffix(".scaler.pkl")
            _safe_pickle_save(self._scaler, scaler_path)

    def _load_model(self) -> None:
        if not HAS_TORCH or not _LSTM_MODEL_PATH.exists():
            return
        try:
            # [S5-C2] Load metadata from JSON (safe), weights with weights_only=True
            meta_path = _LSTM_MODEL_PATH.with_suffix(".meta.json")
            if not meta_path.exists():
                log.debug("LSTM metadata file missing — skipping load")
                return

            with open(meta_path, "r") as f:
                meta = json.load(f)

            self._input_size = meta["input_size"]
            self._hidden_size = meta["hidden_size"]
            self._num_layers = meta["num_layers"]
            self._feature_names = meta["feature_names"]

            self._model = _LSTMModel(
                self._input_size, self._hidden_size, self._num_layers, 3
            )
            state_dict = torch.load(_LSTM_MODEL_PATH, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()

            scaler_path = _LSTM_MODEL_PATH.with_suffix(".scaler.pkl")
            if scaler_path.exists():
                self._scaler = _safe_pickle_load(scaler_path)

            self._is_trained = True
            log.info("Loaded trained LSTM price predictor")
        except Exception as e:
            log.warning(f"Failed to load LSTM model: {e}")


# ---------------------------------------------------------------------------
# 3. RL Position Sizer (PyTorch — optional)
# ---------------------------------------------------------------------------

@dataclass
class RLSizeRecommendation:
    """Result from RL-based position sizing."""
    size_modifier: float    # 0.1 – 2.0 multiplier on base position size
    action: str             # "aggressive" | "normal" | "conservative" | "skip"
    q_values: dict[str, float]
    model_type: str = "rl"


class RLPositionSizer:
    """DQN-based position sizing agent.

    State: portfolio drawdown, recent win rate, signal confidence, volatility,
           regime, funding rate signal.
    Actions: skip (0x), conservative (0.5x), normal (1.0x), aggressive (1.5x)
    Reward: trade P&L (delayed reward after trade closes)

    Requires PyTorch. Falls back to None when not installed.
    """

    _ACTIONS = {
        0: ("skip", 0.0),
        1: ("conservative", 0.5),
        2: ("normal", 1.0),
        3: ("aggressive", 1.5),
    }

    _MODEL_PATH = Path("data/models/rl_position_sizer.pt")

    def __init__(self, state_size: int = 8):
        self._state_size = state_size
        self._model = None
        self._is_trained = False
        self._replay_buffer: list[tuple] = []
        self._epsilon = 0.1  # Exploration rate

        if not HAS_TORCH:
            log.debug("PyTorch not installed — RLPositionSizer disabled")
            return

        self._load_model()

    def is_available(self) -> bool:
        return HAS_TORCH and self._is_trained

    def recommend(
        self,
        drawdown_pct: float,
        recent_win_rate: float,
        signal_confidence: float,
        volatility_pct: float,
        regime: str,
        funding_signal: str,
        consecutive_losses: int,
        portfolio_utilization: float,
    ) -> RLSizeRecommendation | None:
        """Get position size recommendation from the RL agent."""
        if not self.is_available():
            return None

        state = self._encode_state(
            drawdown_pct, recent_win_rate, signal_confidence,
            volatility_pct, regime, funding_signal,
            consecutive_losses, portfolio_utilization,
        )

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            q_values = self._model(state_tensor).squeeze().numpy()

        action_idx = int(np.argmax(q_values))
        action_name, size_mod = self._ACTIONS[action_idx]

        q_dict = {name: round(float(q_values[i]), 4) for i, (name, _) in self._ACTIONS.items()}

        return RLSizeRecommendation(
            size_modifier=size_mod,
            action=action_name,
            q_values=q_dict,
        )

    def record_experience(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer for training."""
        self._replay_buffer.append((state, action, reward, next_state, done))
        # Keep buffer bounded
        if len(self._replay_buffer) > 10_000:
            self._replay_buffer = self._replay_buffer[-10_000:]

    def train_step(self, batch_size: int = 32, gamma: float = 0.99) -> float | None:
        """Run one training step from the replay buffer. Returns loss."""
        if not HAS_TORCH or len(self._replay_buffer) < batch_size:
            return None

        # Sample batch
        import random
        batch = random.sample(self._replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)

        # Q-learning update
        current_q = self._model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self._model(next_states_t).max(1)[0]
            target_q = rewards_t + gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(current_q, target_q)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss.item())

    def _encode_state(
        self,
        drawdown_pct: float,
        recent_win_rate: float,
        signal_confidence: float,
        volatility_pct: float,
        regime: str,
        funding_signal: str,
        consecutive_losses: int,
        portfolio_utilization: float,
    ) -> list[float]:
        """Encode the current state as a fixed-size float vector."""
        regime_map = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}
        funding_map = {
            "overleveraged_longs": -1.0,
            "overleveraged_shorts": 1.0,
            "neutral": 0.0,
        }

        return [
            drawdown_pct,
            recent_win_rate,
            signal_confidence,
            volatility_pct,
            regime_map.get(regime, 0.0),
            funding_map.get(funding_signal, 0.0),
            min(consecutive_losses / 5.0, 1.0),  # Normalize
            portfolio_utilization,
        ]

    def _save_model(self) -> None:
        if not HAS_TORCH or self._model is None:
            return
        self._MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # [S5-C2] Save only model weights, metadata in JSON
        torch.save(self._model.state_dict(), self._MODEL_PATH)
        meta_path = self._MODEL_PATH.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "state_size": self._state_size,
                "replay_buffer_size": len(self._replay_buffer),
            }, f)

    def _load_model(self) -> None:
        if not self._MODEL_PATH.exists():
            # Initialize fresh model
            self._model = _DQNModel(self._state_size, len(self._ACTIONS))
            return
        try:
            # [S5-C2] Load metadata from JSON, weights with weights_only=True
            meta_path = self._MODEL_PATH.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self._state_size = meta["state_size"]

            self._model = _DQNModel(self._state_size, len(self._ACTIONS))
            state_dict = torch.load(self._MODEL_PATH, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)
            self._is_trained = True
            log.info("Loaded trained RL position sizer")
        except Exception as e:
            log.warning(f"Failed to load RL model: {e}")
            self._model = _DQNModel(self._state_size, len(self._ACTIONS))


# ---------------------------------------------------------------------------
# PyTorch model architectures (only defined if torch available)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _LSTMModel(nn.Module):
        """LSTM for price direction classification."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.2 if num_layers > 1 else 0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size // 2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]  # Take last timestep
            return self.fc(last_hidden)

    class _DQNModel(nn.Module):
        """Simple DQN for position sizing."""

        def __init__(self, state_size: int, action_size: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:
    # Stub classes when torch is not available
    class _LSTMModel:  # type: ignore[no-redef]
        pass

    class _DQNModel:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Training script entry point
# ---------------------------------------------------------------------------

def train_all_models(data_dir: str = "data/training") -> dict[str, Any]:
    """Train all ML models from historical data files.

    Expected data format: CSV files in data_dir/ named {PAIR}.csv
    with columns: timestamp, open, high, low, close, volume

    Returns training reports for each model.
    """
    reports = {}

    # Load data
    data_path = Path(data_dir)
    dfs: dict[str, pd.DataFrame] = {}

    if data_path.exists():
        for csv_file in data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 200:
                    dfs[csv_file.stem] = df
            except Exception as e:
                log.warning(f"Failed to load {csv_file}: {e}")

    if not dfs:
        log.warning("No training data found — ML models not trained")
        return {"error": "No training data available"}

    # 1. Regime classifier
    regime_clf = MLRegimeClassifier()
    reports["regime_classifier"] = regime_clf.train(dfs)

    # 2. Price predictor
    if HAS_TORCH:
        predictor = PricePredictor()
        reports["price_predictor"] = predictor.train(dfs)
    else:
        reports["price_predictor"] = {"status": "skipped", "reason": "torch not installed"}

    log.info(f"ML training complete: {list(reports.keys())}")
    return reports
