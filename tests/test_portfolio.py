"""Tests for the portfolio module — dual profit pools and balance persistence."""

import pytest

from src.config import ExchangeConfig, ProfitSplitConfig, RiskConfig
from src.database import Database
from src.exchange import Exchange
from src.portfolio import Portfolio


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def risk_config():
    return RiskConfig(
        starting_capital=1000.0,
        profit_split=ProfitSplitConfig(bot_pct=0.70, user_pct=0.30),
    )


@pytest.fixture
def exchange():
    return Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=1000.0)


@pytest.fixture
def portfolio(exchange, db, risk_config):
    return Portfolio(exchange, db, risk_config)


# --- apply_profit_split: profitable trades ---


def test_profit_split_positive_default_ratio(portfolio):
    """Profitable trade splits 70/30 between bot and user pools."""
    bot_share, user_share = portfolio.apply_profit_split(100.0)

    assert bot_share == pytest.approx(70.0)
    assert user_share == pytest.approx(30.0)
    assert portfolio.bot_balance == pytest.approx(1070.0)
    assert portfolio.user_balance == pytest.approx(30.0)


def test_profit_split_custom_ratio(db):
    """Profit split respects non-default ratios."""
    config = RiskConfig(
        starting_capital=500.0,
        profit_split=ProfitSplitConfig(bot_pct=0.50, user_pct=0.50),
    )
    exchange = Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=500.0)
    port = Portfolio(exchange, db, config)

    bot_share, user_share = port.apply_profit_split(200.0)

    assert bot_share == pytest.approx(100.0)
    assert user_share == pytest.approx(100.0)
    assert port.bot_balance == pytest.approx(600.0)
    assert port.user_balance == pytest.approx(100.0)


def test_profit_split_accumulates(portfolio):
    """Multiple profitable trades accumulate in both pools."""
    portfolio.apply_profit_split(100.0)
    portfolio.apply_profit_split(50.0)

    # bot: 1000 + 70 + 35 = 1105, user: 0 + 30 + 15 = 45
    assert portfolio.bot_balance == pytest.approx(1105.0)
    assert portfolio.user_balance == pytest.approx(45.0)


# --- apply_profit_split: losing trades ---


def test_loss_absorbed_by_bot_only(portfolio):
    """Losses come entirely from bot pool; user pool stays at zero."""
    bot_share, user_share = portfolio.apply_profit_split(-50.0)

    assert bot_share == pytest.approx(-50.0)
    assert user_share == pytest.approx(0.0)
    assert portfolio.bot_balance == pytest.approx(950.0)
    assert portfolio.user_balance == pytest.approx(0.0)


def test_user_balance_never_reduced_by_loss(portfolio):
    """User pool is protected: profit first, then a loss must not touch user balance."""
    portfolio.apply_profit_split(100.0)  # user gets 30
    user_after_profit = portfolio.user_balance

    portfolio.apply_profit_split(-80.0)  # loss absorbed by bot only

    assert portfolio.user_balance == pytest.approx(user_after_profit)
    assert portfolio.bot_balance == pytest.approx(1070.0 - 80.0)  # 990


def test_user_balance_stays_zero_on_pure_losses(portfolio):
    """Repeated losses never make user balance go negative."""
    portfolio.apply_profit_split(-100.0)
    portfolio.apply_profit_split(-200.0)

    assert portfolio.user_balance == pytest.approx(0.0)
    assert portfolio.bot_balance == pytest.approx(700.0)


# --- Balance persistence via _persist_balances ---


def test_persist_balances_written_to_db(portfolio, db):
    """apply_profit_split persists both pool balances to agent_state."""
    portfolio.apply_profit_split(100.0)

    stored_bot = db.get_state("bot_balance")
    stored_user = db.get_state("user_balance")

    assert stored_bot is not None
    assert stored_user is not None
    assert float(stored_bot) == pytest.approx(1070.0)
    assert float(stored_user) == pytest.approx(30.0)


def test_persist_balances_after_loss(portfolio, db):
    """Loss is persisted correctly in agent_state."""
    portfolio.apply_profit_split(-25.0)

    assert float(db.get_state("bot_balance")) == pytest.approx(975.0)
    assert float(db.get_state("user_balance")) == pytest.approx(0.0)


def test_balances_survive_reconstruction(db, risk_config):
    """A new Portfolio instance loads persisted balances from DB."""
    exchange1 = Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=1000.0)
    port1 = Portfolio(exchange1, db, risk_config)
    port1.apply_profit_split(200.0)  # bot: 1140, user: 60

    # Simulate restart: new Portfolio with same DB
    exchange2 = Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=1000.0)
    port2 = Portfolio(exchange2, db, risk_config)

    assert port2.bot_balance == pytest.approx(1140.0)
    assert port2.user_balance == pytest.approx(60.0)


def test_balances_survive_profit_then_loss_then_reload(db, risk_config):
    """Full round-trip: profit, loss, persist, reload."""
    exchange1 = Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=1000.0)
    port1 = Portfolio(exchange1, db, risk_config)
    port1.apply_profit_split(100.0)   # bot: 1070, user: 30
    port1.apply_profit_split(-40.0)   # bot: 1030, user: 30

    exchange2 = Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=1000.0)
    port2 = Portfolio(exchange2, db, risk_config)

    assert port2.bot_balance == pytest.approx(1030.0)
    assert port2.user_balance == pytest.approx(30.0)


# --- Edge cases ---


def test_zero_pnl_no_change(portfolio):
    """Zero P&L doesn't change either pool (treated as a profit of 0)."""
    bot_share, user_share = portfolio.apply_profit_split(0.0)

    # pnl=0 is not > 0, so it falls into the loss branch with bot_share=0
    assert bot_share == pytest.approx(0.0)
    assert user_share == pytest.approx(0.0)
    assert portfolio.bot_balance == pytest.approx(1000.0)
    assert portfolio.user_balance == pytest.approx(0.0)


def test_tiny_profit_split(portfolio):
    """Very small profit still splits correctly."""
    bot_share, user_share = portfolio.apply_profit_split(0.01)

    assert bot_share == pytest.approx(0.007)
    assert user_share == pytest.approx(0.003)
