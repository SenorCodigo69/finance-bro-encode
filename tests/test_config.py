"""Tests for configuration loading."""

import pytest
import yaml

from src.config import (
    Config,
    CorrelationGuardConfig,
    EscalationConfig,
    ExchangeConfig,
    FallbackExchangeConfig,
    ProfitSplitConfig,
    RiskConfig,
    StrategyConfig,
    AgentConfig,
    load_config,
)


# ── Defaults ──────────────────────────────────────────────────────────


def test_load_config_returns_valid_defaults(tmp_path):
    """load_config() with no YAML file still returns a Config with defaults."""
    cfg = load_config(config_path=str(tmp_path / "nonexistent.yaml"))

    assert isinstance(cfg, Config)
    assert cfg.exchange.name == "hyperliquid"
    assert cfg.exchange.sandbox is True
    assert cfg.exchange.quote_currency == "USDC"
    assert cfg.risk.starting_capital == 1000.0
    assert cfg.risk.max_drawdown_pct == 0.30
    assert cfg.risk.max_position_pct == 0.25
    assert cfg.agent.mode == "paper"
    assert cfg.agent.pairs == ["BTC/USDC:USDC", "ETH/USDC:USDC"]
    assert cfg.strategy.rsi_period == 14


# ── Nested dataclass defaults ─────────────────────────────────────────


def test_profit_split_defaults():
    ps = ProfitSplitConfig()
    assert ps.bot_pct == 0.70
    assert ps.user_pct == 0.30
    assert ps.user_take_profit_pct == 0.04
    assert ps.bot_take_profit_pct == 0.08


def test_correlation_guard_defaults():
    cg = CorrelationGuardConfig()
    assert cg.enabled is True
    assert cg.btc_drop_threshold_pct == -2.0
    assert cg.btc_pump_threshold_pct == 2.0


def test_escalation_defaults():
    esc = EscalationConfig()
    assert esc.enabled is True
    assert esc.min_signal_confidence == 0.7
    assert esc.tiebreaker_provider == "claude"


def test_risk_config_contains_profit_split():
    rc = RiskConfig()
    assert isinstance(rc.profit_split, ProfitSplitConfig)


def test_strategy_config_contains_correlation_guard():
    sc = StrategyConfig()
    assert isinstance(sc.correlation_guard, CorrelationGuardConfig)


def test_agent_config_contains_escalation():
    ac = AgentConfig()
    assert isinstance(ac.escalation, EscalationConfig)


# ── __repr__ masks secrets ────────────────────────────────────────────


def test_repr_masks_api_keys():
    from src.config import SecretStr
    cfg = Config(anthropic_api_key=SecretStr("sk-secret-key"), exchange_api_key=SecretStr("ex-secret"))
    r = repr(cfg)
    assert "sk-secret-key" not in r
    assert "ex-secret" not in r
    assert "***" in r


def test_repr_empty_keys_not_masked():
    from src.config import SecretStr
    cfg = Config(anthropic_api_key=SecretStr(""), exchange_api_key=SecretStr(""))
    r = repr(cfg)
    # Empty SecretStr repr is '' not '***'
    assert "sk-" not in r


# ── Loading from YAML ─────────────────────────────────────────────────


def test_load_from_yaml(tmp_path):
    settings = {
        "exchange": {"name": "bybit", "sandbox": False, "rate_limit_ms": 500},
        "risk": {
            "starting_capital": 5000.0,
            "max_drawdown_pct": 0.20,
            "profit_split": {"bot_pct": 0.60, "user_pct": 0.40},
        },
        "strategy": {
            "rsi_period": 21,
            "correlation_guard": {"enabled": False},
        },
        "agent": {
            "mode": "live",
            "cycle_interval_sec": 60,
            "escalation": {"min_signal_confidence": 0.9},
        },
    }
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text(yaml.dump(settings))

    cfg = load_config(config_path=str(yaml_file))

    assert cfg.exchange.name == "bybit"
    assert cfg.exchange.sandbox is False
    assert cfg.exchange.rate_limit_ms == 500
    assert cfg.risk.starting_capital == 5000.0
    assert cfg.risk.max_drawdown_pct == 0.20
    assert cfg.risk.profit_split.bot_pct == 0.60
    assert cfg.risk.profit_split.user_pct == 0.40
    assert cfg.strategy.rsi_period == 21
    assert cfg.strategy.correlation_guard.enabled is False
    assert cfg.agent.mode == "live"
    assert cfg.agent.cycle_interval_sec == 60
    assert cfg.agent.escalation.min_signal_confidence == 0.9


def test_load_yaml_with_fallbacks(tmp_path):
    settings = {
        "exchange": {
            "name": "hyperliquid",
            "fallbacks": [
                {"name": "okx", "rate_limit_ms": 800},
                {"name": "bybit", "rate_limit_ms": 1000},
            ],
        },
    }
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text(yaml.dump(settings))

    cfg = load_config(config_path=str(yaml_file))

    assert len(cfg.exchange.fallbacks) == 2
    assert cfg.exchange.fallbacks[0].name == "okx"
    assert cfg.exchange.fallbacks[0].rate_limit_ms == 800
    assert cfg.exchange.fallbacks[1].name == "bybit"


# ── Environment variables ─────────────────────────────────────────────


def test_env_vars_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    monkeypatch.setenv("EXCHANGE_API_KEY", "ex-key-456")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "ex-secret-789")
    monkeypatch.setenv("EXCHANGE_PASSWORD", "ex-pass-000")

    cfg = load_config(config_path=str(tmp_path / "nonexistent.yaml"))

    assert cfg.anthropic_api_key.get_secret_value() == "sk-ant-test-123"
    assert cfg.exchange_api_key.get_secret_value() == "ex-key-456"
    assert cfg.exchange_api_secret.get_secret_value() == "ex-secret-789"
    assert cfg.exchange_password.get_secret_value() == "ex-pass-000"


def test_missing_env_vars_default_empty(tmp_path, monkeypatch):
    # Patch load_dotenv to no-op so it doesn't read the real .env file
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
    monkeypatch.delenv("EXCHANGE_PASSWORD", raising=False)

    cfg = load_config(config_path=str(tmp_path / "nonexistent.yaml"))

    assert cfg.anthropic_api_key.get_secret_value() == ""
    assert cfg.exchange_api_key.get_secret_value() == ""
    assert cfg.exchange_api_secret.get_secret_value() == ""
    assert cfg.exchange_password.get_secret_value() == ""


# ── Unknown YAML keys silently ignored ────────────────────────────────


def test_unknown_yaml_keys_ignored(tmp_path):
    settings = {
        "exchange": {"name": "hyperliquid", "bogus_key": 999},
        "risk": {"starting_capital": 2000.0, "unknown_field": True},
        "strategy": {"rsi_period": 10, "nonexistent": "value"},
        "agent": {"mode": "paper", "extra_stuff": [1, 2, 3]},
        "totally_unknown_section": {"a": 1},
    }
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text(yaml.dump(settings))

    cfg = load_config(config_path=str(yaml_file))

    assert cfg.exchange.name == "hyperliquid"
    assert cfg.risk.starting_capital == 2000.0
    assert cfg.strategy.rsi_period == 10
    assert cfg.agent.mode == "paper"
