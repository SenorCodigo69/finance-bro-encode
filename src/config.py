"""Configuration loader — merges .env secrets with YAML settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SecretStr:
    """String wrapper that prevents accidental logging of secrets."""

    __slots__ = ("_value",)

    def __init__(self, value: str = ""):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "'***'" if self._value else "''"

    def __str__(self) -> str:
        return "***" if self._value else ""

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return self._value == other._value
        return NotImplemented


@dataclass
class FallbackExchangeConfig:
    name: str = "bybit"
    rate_limit_ms: int = 1200


@dataclass
class DataSourcesConfig:
    max_price_divergence_pct: float = 2.0
    anomaly_zscore_threshold: float = 3.0


@dataclass
class EventBlockingConfig:
    enabled: bool = True
    stocks_hours_before: int = 24
    stocks_hours_after: int = 2
    macro_events: list[str] = field(default_factory=lambda: ["FOMC", "CPI", "NFP"])


@dataclass
class DexVenueConfig:
    name: str = "dydx"
    enabled: bool = True
    taker_fee_bps: float = 5.0


@dataclass
class DexScannerConfig:
    enabled: bool = True
    venues: list[DexVenueConfig] = field(default_factory=lambda: [
        DexVenueConfig(name="dydx", enabled=True),
        DexVenueConfig(name="gmx", enabled=False),
    ])
    scan_interval_sec: int = 60
    divergence_alert_pct: float = 1.0


@dataclass
class TriggerOrderConfig:
    enabled: bool = True
    trailing_update_min_interval_sec: int = 300
    trailing_update_min_move_pct: float = 0.5


@dataclass
class ExecutionRouterConfig:
    primary_venue: str = "hyperliquid"
    enable_multi_venue: bool = False
    slippage_tracking: bool = True


@dataclass
class ExchangeConfig:
    name: str = "hyperliquid"
    sandbox: bool = True
    rate_limit_ms: int = 1200
    quote_currency: str = "USDC"
    fallbacks: list[FallbackExchangeConfig] = field(default_factory=lambda: [
        FallbackExchangeConfig(name="bybit", rate_limit_ms=1200),
    ])


@dataclass
class ProfitSplitConfig:
    bot_pct: float = 0.70
    user_pct: float = 0.30
    user_take_profit_pct: float = 0.04
    bot_take_profit_pct: float = 0.08
    user_trailing_stop_pct: float = 0.02
    bot_trailing_stop_pct: float = 0.035


@dataclass
class RiskConfig:
    starting_capital: float = 1000.0
    max_drawdown_pct: float = 0.30
    max_position_pct: float = 0.25
    default_stop_loss_pct: float = 0.03
    max_open_positions: int = 3
    max_daily_trades: int = 10
    cooldown_after_loss_min: int = 30
    max_hold_hours: float = 0  # 0 = disabled, >0 = auto-close after N hours
    profit_split: ProfitSplitConfig = field(default_factory=ProfitSplitConfig)


@dataclass
class CorrelationGuardConfig:
    enabled: bool = True
    btc_drop_threshold_pct: float = -2.0
    btc_pump_threshold_pct: float = 2.0


@dataclass
class LookbackConfig:
    default: int = 20
    min_candles: int = 30
    per_asset: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.default, int) or self.default < 1 or self.default > 10000:
            raise ValueError(f"lookback default must be int in [1, 10000], got {self.default!r}")
        if not isinstance(self.min_candles, int) or self.min_candles < 1 or self.min_candles > 100000:
            raise ValueError(f"min_candles must be int in [1, 100000], got {self.min_candles!r}")
        for pair, cfg in self.per_asset.items():
            if not isinstance(cfg, dict):
                raise TypeError(f"per_asset[{pair}] must be dict, got {type(cfg).__name__}")
            lb = cfg.get("lookback")
            if lb is not None and (not isinstance(lb, int) or lb < 1 or lb > 10000):
                raise ValueError(f"per_asset[{pair}].lookback must be int in [1, 10000], got {lb!r}")

    def get_lookback(self, pair: str) -> int:
        """Return lookback periods for a given pair, falling back to default."""
        asset_cfg = self.per_asset.get(pair, {})
        return asset_cfg.get("lookback", self.default)


@dataclass
class StrategyConfig:
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volume_sma_period: int = 20
    pair_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    correlation_guard: CorrelationGuardConfig = field(default_factory=CorrelationGuardConfig)
    lookback: LookbackConfig = field(default_factory=LookbackConfig)


@dataclass
class EscalationConfig:
    enabled: bool = True
    min_signal_confidence: float = 0.7    # Only escalate high-confidence signals
    tiebreaker_provider: str = "claude"   # Which provider to use as tiebreaker


@dataclass
class AgentConfig:
    mode: str = "paper"
    cycle_interval_sec: int = 300
    pairs: list[str] = field(default_factory=lambda: ["BTC/USDC:USDC", "ETH/USDC:USDC"])
    timeframes: list[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h"])
    claude_model: str = "claude-haiku-4-5-20251001"
    review_every_n_trades: int = 10
    dry_run: bool = False
    consensus_threshold: float = 0.66  # 2/3 majority needed to approve
    providers: list[dict] = field(default_factory=lambda: [
        {"name": "gemini", "model": "gemini-2.0-flash", "enabled": True},
        {"name": "claude", "model": "claude-haiku-4-5-20251001", "enabled": False},
        {"name": "ollama", "model": "qwen3:8b", "enabled": True},
    ])
    escalation: EscalationConfig = field(default_factory=EscalationConfig)


@dataclass
class Config:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    event_blocking: EventBlockingConfig = field(default_factory=EventBlockingConfig)
    dex_scanner: DexScannerConfig = field(default_factory=DexScannerConfig)
    trigger_orders: TriggerOrderConfig = field(default_factory=TriggerOrderConfig)
    execution_router: ExecutionRouterConfig = field(default_factory=ExecutionRouterConfig)

    # Secrets from .env — wrapped to prevent accidental logging
    anthropic_api_key: SecretStr = field(default_factory=SecretStr)
    exchange_api_key: SecretStr = field(default_factory=SecretStr)
    exchange_api_secret: SecretStr = field(default_factory=SecretStr)
    exchange_password: SecretStr = field(default_factory=SecretStr)
    alpha_vantage_api_key: SecretStr = field(default_factory=SecretStr)
    fred_api_key: SecretStr = field(default_factory=SecretStr)


def load_config(config_path: str | None = None) -> Config:
    """Load config from .env + settings.yaml."""
    load_dotenv(PROJECT_ROOT / ".env")

    yaml_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "settings.yaml"
    raw: dict = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}

    exchange_raw = raw.get("exchange", {})
    risk_raw = raw.get("risk", {})
    strategy_raw = raw.get("strategy", {})
    agent_raw = raw.get("agent", {})

    # Parse nested profit_split from risk config
    profit_split_raw = risk_raw.pop("profit_split", {}) if "profit_split" in risk_raw else {}
    profit_split = ProfitSplitConfig(
        **{k: v for k, v in profit_split_raw.items() if k in ProfitSplitConfig.__dataclass_fields__}
    )

    risk_kwargs = {k: v for k, v in risk_raw.items() if k in RiskConfig.__dataclass_fields__}
    risk_kwargs["profit_split"] = profit_split

    # Parse nested correlation_guard from strategy config
    corr_guard_raw = strategy_raw.pop("correlation_guard", {}) if "correlation_guard" in strategy_raw else {}
    corr_guard = CorrelationGuardConfig(
        **{k: v for k, v in corr_guard_raw.items() if k in CorrelationGuardConfig.__dataclass_fields__}
    )

    # Parse nested lookback from strategy config
    lookback_raw = strategy_raw.pop("lookback", {}) if "lookback" in strategy_raw else {}
    lookback = LookbackConfig(
        **{k: v for k, v in lookback_raw.items() if k in LookbackConfig.__dataclass_fields__}
    )

    strategy_kwargs = {k: v for k, v in strategy_raw.items() if k in StrategyConfig.__dataclass_fields__}
    strategy_kwargs["correlation_guard"] = corr_guard
    strategy_kwargs["lookback"] = lookback

    # Parse nested fallbacks from exchange config
    fallbacks_raw = exchange_raw.pop("fallbacks", []) if "fallbacks" in exchange_raw else []
    fallbacks = [
        FallbackExchangeConfig(**{k: v for k, v in fb.items() if k in FallbackExchangeConfig.__dataclass_fields__})
        for fb in fallbacks_raw
    ]

    exchange_kwargs = {k: v for k, v in exchange_raw.items() if k in ExchangeConfig.__dataclass_fields__}
    if fallbacks:
        exchange_kwargs["fallbacks"] = fallbacks

    # Parse nested escalation from agent config
    escalation_raw = agent_raw.pop("escalation", {}) if "escalation" in agent_raw else {}
    escalation = EscalationConfig(
        **{k: v for k, v in escalation_raw.items() if k in EscalationConfig.__dataclass_fields__}
    )

    agent_kwargs = {k: v for k, v in agent_raw.items() if k in AgentConfig.__dataclass_fields__}
    agent_kwargs["escalation"] = escalation

    # Parse data_sources section
    data_sources_raw = raw.get("data_sources", {})
    data_sources = DataSourcesConfig(
        **{k: v for k, v in data_sources_raw.items() if k in DataSourcesConfig.__dataclass_fields__}
    )

    # Parse event_blocking section
    event_blocking_raw = raw.get("event_blocking", {})
    # Handle macro_events list properly
    event_blocking = EventBlockingConfig(
        **{k: v for k, v in event_blocking_raw.items() if k in EventBlockingConfig.__dataclass_fields__}
    )

    # Parse dex_scanner section
    dex_scanner_raw = raw.get("dex_scanner", {})
    dex_venues_raw = dex_scanner_raw.pop("venues", []) if "venues" in dex_scanner_raw else []
    dex_venues = [
        DexVenueConfig(**{k: v for k, v in dv.items() if k in DexVenueConfig.__dataclass_fields__})
        for dv in dex_venues_raw
    ]
    dex_scanner_kwargs = {k: v for k, v in dex_scanner_raw.items() if k in DexScannerConfig.__dataclass_fields__}
    if dex_venues:
        dex_scanner_kwargs["venues"] = dex_venues
    dex_scanner = DexScannerConfig(**dex_scanner_kwargs)

    # Parse trigger_orders section
    trigger_orders_raw = raw.get("trigger_orders", {})
    trigger_orders = TriggerOrderConfig(
        **{k: v for k, v in trigger_orders_raw.items() if k in TriggerOrderConfig.__dataclass_fields__}
    )

    # Parse execution_router section
    exec_router_raw = raw.get("execution_router", {})
    exec_router = ExecutionRouterConfig(
        **{k: v for k, v in exec_router_raw.items() if k in ExecutionRouterConfig.__dataclass_fields__}
    )

    return Config(
        exchange=ExchangeConfig(**exchange_kwargs),
        risk=RiskConfig(**risk_kwargs),
        strategy=StrategyConfig(**strategy_kwargs),
        agent=AgentConfig(**agent_kwargs),
        data_sources=data_sources,
        event_blocking=event_blocking,
        dex_scanner=dex_scanner,
        trigger_orders=trigger_orders,
        execution_router=exec_router,
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
        exchange_api_key=SecretStr(os.getenv("EXCHANGE_API_KEY", "")),
        exchange_api_secret=SecretStr(os.getenv("EXCHANGE_API_SECRET", "")),
        exchange_password=SecretStr(os.getenv("EXCHANGE_PASSWORD", "")),
        alpha_vantage_api_key=SecretStr(os.getenv("ALPHA_VANTAGE_API_KEY", "")),
        fred_api_key=SecretStr(os.getenv("FRED_API_KEY", "")),
    )
