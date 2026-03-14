"""Tests for the exchange module — parse_pair + paper trading logic."""

import pytest

from src.config import ExchangeConfig
from src.exchange import Exchange, parse_pair


# ── parse_pair ──────────────────────────────────────────────────────


class TestParsePair:

    def test_simple_pair(self):
        assert parse_pair("BTC/USDC") == ("BTC", "USDC")

    def test_contract_pair(self):
        assert parse_pair("BTC/USDC:USDC") == ("BTC", "USDC")

    def test_eth_pair(self):
        assert parse_pair("ETH/USDC") == ("ETH", "USDC")

    def test_contract_pair_different_settle(self):
        assert parse_pair("ETH/USDT:USDT") == ("ETH", "USDT")

    def test_whitespace_in_pair(self):
        assert parse_pair(" BTC / USDC ") == ("BTC", "USDC")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid pair format"):
            parse_pair("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Invalid pair format"):
            parse_pair(None)

    def test_no_slash_raises(self):
        with pytest.raises(ValueError, match="Invalid pair format"):
            parse_pair("BTCUSDC")

    def test_empty_base_raises(self):
        with pytest.raises(ValueError, match="Empty base or quote"):
            parse_pair("/USDC")

    def test_empty_quote_raises(self):
        with pytest.raises(ValueError, match="Empty base or quote"):
            parse_pair("BTC/")

    def test_slash_only_raises(self):
        with pytest.raises(ValueError, match="Empty base or quote"):
            parse_pair("/")


# ── Exchange paper mode ─────────────────────────────────────────────


@pytest.fixture
def exchange():
    return Exchange(ExchangeConfig(name="hyperliquid"), mode="paper", starting_capital=200.0)


def test_initial_balance(exchange):
    assert exchange.paper_balance["USDC"] == 200.0
    assert len(exchange.paper_balance) == 1


def test_mode_is_paper(exchange):
    assert exchange.mode == "paper"


@pytest.mark.asyncio
async def test_fetch_balance_paper(exchange):
    result = await exchange.fetch_balance()
    assert result["free"]["USDC"] == 200.0
    assert result["total"]["USDC"] == 200.0
    # Returned dicts should be copies, not references
    result["free"]["USDC"] = 0.0
    assert exchange.paper_balance["USDC"] == 200.0


# ── Market order buy ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_buy_order_updates_balances(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0, "bid": 49999.0, "ask": 50001.0}

    exchange.fetch_ticker = mock_ticker

    order = await exchange.create_market_order("BTC/USDC:USDC", "buy", 0.001)

    assert order["status"] == "closed"
    assert order["side"] == "buy"
    assert order["type"] == "market"
    assert order["amount"] == 0.001
    assert order["id"] == "paper-1"
    assert exchange.paper_balance["BTC"] == 0.001
    assert exchange.paper_balance["USDC"] < 200.0


@pytest.mark.asyncio
async def test_buy_order_deducts_cost_plus_fee(exchange):
    async def mock_ticker(pair):
        return {"last": 100.0}

    exchange.fetch_ticker = mock_ticker

    order = await exchange.create_market_order("ETH/USDC:USDC", "buy", 1.0)

    cost = order["cost"]
    fee = order["fee"]
    expected_remaining = 200.0 - cost - fee
    assert exchange.paper_balance["USDC"] == pytest.approx(expected_remaining, abs=0.01)


@pytest.mark.asyncio
async def test_buy_order_fee_calculation(exchange):
    async def mock_ticker(pair):
        return {"last": 10000.0}

    exchange.fetch_ticker = mock_ticker

    order = await exchange.create_market_order("BTC/USDC:USDC", "buy", 0.01)

    cost = order["cost"]
    expected_fee = cost * 0.00045
    assert order["fee"] == pytest.approx(expected_fee, rel=1e-6)


# ── Market order sell ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sell_order_updates_balances(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker

    # Seed BTC balance first
    exchange.paper_balance["BTC"] = 0.01

    order = await exchange.create_market_order("BTC/USDC:USDC", "sell", 0.005)

    assert order["status"] == "closed"
    assert order["side"] == "sell"
    assert exchange.paper_balance["BTC"] == pytest.approx(0.005, abs=1e-9)
    assert exchange.paper_balance["USDC"] > 200.0  # Got USDC back


@pytest.mark.asyncio
async def test_sell_order_credits_cost_minus_fee(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker
    exchange.paper_balance["BTC"] = 0.01

    order = await exchange.create_market_order("BTC/USDC:USDC", "sell", 0.01)

    cost = order["cost"]
    fee = order["fee"]
    expected_usdc = 200.0 + cost - fee
    assert exchange.paper_balance["USDC"] == pytest.approx(expected_usdc, abs=0.01)


# ── Insufficient balance ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_insufficient_quote_for_buy(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker

    with pytest.raises(Exception, match="Insufficient USDC"):
        await exchange.create_market_order("BTC/USDC:USDC", "buy", 1.0)  # 50k > 200


@pytest.mark.asyncio
async def test_insufficient_base_for_sell(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker

    with pytest.raises(Exception, match="Insufficient BTC"):
        await exchange.create_market_order("BTC/USDC:USDC", "sell", 0.01)  # No BTC held


@pytest.mark.asyncio
async def test_insufficient_base_partial_sell(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker
    exchange.paper_balance["BTC"] = 0.001

    with pytest.raises(Exception, match="Insufficient BTC"):
        await exchange.create_market_order("BTC/USDC:USDC", "sell", 0.01)


# ── Fee calculation ─────────────────────────────────────────────────


def test_fee_rate_default(exchange):
    assert exchange._fee_rate == 0.00045


@pytest.mark.asyncio
async def test_fee_nonzero_on_order(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker

    order = await exchange.create_market_order("BTC/USDC:USDC", "buy", 0.001)
    assert order["fee"] > 0


@pytest.mark.asyncio
async def test_fee_proportional_to_cost(exchange):
    async def mock_ticker(pair):
        return {"last": 20000.0}

    exchange.fetch_ticker = mock_ticker

    order = await exchange.create_market_order("BTC/USDC:USDC", "buy", 0.005)
    assert order["fee"] == pytest.approx(order["cost"] * 0.00045, rel=1e-6)


# ── Slippage ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_buy_fill_price_above_last(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker

    order = await exchange.create_market_order("BTC/USDC:USDC", "buy", 0.001)
    assert order["price"] > 50000.0  # Slippage moves price up on buy


@pytest.mark.asyncio
async def test_sell_fill_price_below_last(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker
    exchange.paper_balance["BTC"] = 0.01

    order = await exchange.create_market_order("BTC/USDC:USDC", "sell", 0.001)
    assert order["price"] < 50000.0  # Slippage moves price down on sell


# ── Portfolio value ─────────────────────────────────────────────────


def test_portfolio_value_cash_only(exchange):
    value = exchange.get_paper_portfolio_value({})
    assert value == 200.0


def test_portfolio_value_with_holdings(exchange):
    exchange.paper_balance = {"USDC": 100.0, "BTC": 0.002}
    value = exchange.get_paper_portfolio_value({"BTC/USDC:USDC": 50000.0})
    assert value == pytest.approx(200.0, abs=0.01)


def test_portfolio_value_ignores_unknown_assets(exchange):
    exchange.paper_balance = {"USDC": 100.0, "XYZ": 10.0}
    value = exchange.get_paper_portfolio_value({"BTC/USDC:USDC": 50000.0})
    # XYZ has no price — should be skipped, only USDC counted
    assert value == 100.0


def test_portfolio_value_ignores_zero_holdings(exchange):
    exchange.paper_balance = {"USDC": 100.0, "BTC": 0.0}
    value = exchange.get_paper_portfolio_value({"BTC/USDC:USDC": 50000.0})
    assert value == 100.0


def test_portfolio_value_multiple_assets(exchange):
    exchange.paper_balance = {"USDC": 50.0, "BTC": 0.001, "ETH": 0.1}
    prices = {"BTC/USDC:USDC": 50000.0, "ETH/USDC:USDC": 3000.0}
    value = exchange.get_paper_portfolio_value(prices)
    # 50 + 0.001*50000 + 0.1*3000 = 50 + 50 + 300 = 400
    assert value == pytest.approx(400.0, abs=0.01)


# ── Order counter + order history ───────────────────────────────────


@pytest.mark.asyncio
async def test_order_counter_increments(exchange):
    async def mock_ticker(pair):
        return {"last": 1000.0}

    exchange.fetch_ticker = mock_ticker

    o1 = await exchange.create_market_order("ETH/USDC:USDC", "buy", 0.01)
    o2 = await exchange.create_market_order("ETH/USDC:USDC", "buy", 0.01)

    assert o1["id"] == "paper-1"
    assert o2["id"] == "paper-2"
    assert len(exchange.paper_orders) == 2


# ── Limit orders (paper) ───────────────────────────────────────────


def test_limit_order_recorded_as_open(exchange):
    order = exchange._paper_limit_order("BTC/USDC:USDC", "buy", 0.001, 48000.0)
    assert order["status"] == "open"
    assert order["type"] == "limit"
    assert order["price"] == 48000.0
    assert order["cost"] == 48000.0 * 0.001


# ── Cancel orders (paper) ──────────────────────────────────────────


def test_cancel_open_order(exchange):
    order = exchange._paper_limit_order("BTC/USDC:USDC", "buy", 0.001, 48000.0)
    cancelled = exchange._paper_cancel_order(order["id"])
    assert cancelled["status"] == "cancelled"


def test_cancel_nonexistent_order_raises(exchange):
    with pytest.raises(Exception, match="not found or not open"):
        exchange._paper_cancel_order("paper-999")


def test_cancel_already_cancelled_raises(exchange):
    order = exchange._paper_limit_order("BTC/USDC:USDC", "buy", 0.001, 48000.0)
    exchange._paper_cancel_order(order["id"])
    with pytest.raises(Exception, match="not found or not open"):
        exchange._paper_cancel_order(order["id"])


# ── Round-trip buy then sell ────────────────────────────────────────


@pytest.mark.asyncio
async def test_buy_then_sell_round_trip(exchange):
    async def mock_ticker(pair):
        return {"last": 50000.0}

    exchange.fetch_ticker = mock_ticker

    await exchange.create_market_order("BTC/USDC:USDC", "buy", 0.001)
    assert exchange.paper_balance["BTC"] == 0.001

    await exchange.create_market_order("BTC/USDC:USDC", "sell", 0.001)
    assert exchange.paper_balance["BTC"] == pytest.approx(0.0, abs=1e-12)
    # After fees + slippage, should have slightly less than started
    assert exchange.paper_balance["USDC"] < 200.0


# ── Input validation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_side_raises(exchange):
    with pytest.raises(ValueError, match="Invalid side"):
        await exchange.create_market_order("BTC/USDC:USDC", "BUY", 0.001)


@pytest.mark.asyncio
async def test_negative_amount_raises(exchange):
    with pytest.raises(ValueError, match="Amount must be positive"):
        await exchange.create_market_order("BTC/USDC:USDC", "buy", -0.001)


@pytest.mark.asyncio
async def test_zero_amount_raises(exchange):
    with pytest.raises(ValueError, match="Amount must be positive"):
        await exchange.create_market_order("BTC/USDC:USDC", "buy", 0)


@pytest.mark.asyncio
async def test_limit_order_negative_price_raises(exchange):
    with pytest.raises(ValueError, match="Price must be positive"):
        await exchange.create_limit_order("BTC/USDC:USDC", "buy", 0.001, -100.0)


# ── _pair_to_hl_coin ──────────────────────────────────────────────

from src.exchange import _pair_to_hl_coin


def test_pair_to_hl_coin_crypto():
    assert _pair_to_hl_coin("BTC/USDC:USDC") == "BTC"
    assert _pair_to_hl_coin("ETH/USDC:USDC") == "ETH"


def test_pair_to_hl_coin_synthetic():
    assert _pair_to_hl_coin("XYZ-NVDA/USDC:USDC") == "xyz:NVDA"
    assert _pair_to_hl_coin("XYZ-GOLD/USDC:USDC") == "xyz:GOLD"


def test_pair_to_hl_coin_invalid():
    with pytest.raises(ValueError):
        _pair_to_hl_coin("INVALID")
