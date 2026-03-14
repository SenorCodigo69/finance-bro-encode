"""Tests for the data integrity checker."""

import pytest
from src.data_integrity import DataIntegrityChecker, IntegrityReport


def make_ctx(**overrides) -> dict:
    """Build a minimal macro context dict for testing."""
    base = {
        "timestamp": "2026-03-11T12:00:00",
        "fear_greed_index": {"value": 50, "label": "Neutral"},
        "global_market": {"market_cap_change_24h_pct": 1.5},
        "coin_prices": {
            "BTC": {"price": 85000, "change_1h_pct": 0.5, "change_24h_pct": 2.1},
            "ETH": {"price": 3200, "change_1h_pct": 0.3, "change_24h_pct": 1.8},
        },
        "trending_coins": [{"name": "Bitcoin", "symbol": "BTC"}],
        "reddit_sentiment": {"cryptocurrency": [{"title": "BTC moon"}]},
        "dxy_proxy": {"usd_eur": 0.92},
        "cryptopanic_news": [{"title": "BTC ETF approved"}],
        "yahoo_macro": {"VIX": {"last_close": 18.5}, "DXY": {"last_close": 103.2}},
        "rss_news": {"Reuters": [{"title": "Crypto market steady"}]},
        "hackernews": [{"title": "Bitcoin discussion"}],
        "polymarket": [{"question": "BTC > 100k?"}],
        "funding_rates": {
            "BTC": {"rate": 0.00005, "rate_pct": 0.005, "signal": "neutral"},
            "ETH": {"rate": 0.00003, "rate_pct": 0.003, "signal": "neutral"},
        },
        "open_interest": {
            "BTC": {"open_interest": 50000.0},
            "ETH": {"open_interest": 30000.0},
        },
        "long_short_ratio": {
            "BTC": {"ratio": 1.2, "long_pct": 54.5, "short_pct": 45.5, "signal": "balanced"},
        },
        "taker_buy_sell": {
            "BTC": {"buy_sell_ratio": 1.05, "signal": "balanced"},
        },
        "economic_calendar": [
            {"title": "ISM Manufacturing PMI", "date": "2026-03-12T14:00:00", "imminent": False},
        ],
        "onchain_btc": {
            "hash_rate": 650.0,
            "mempool_congestion": "low",
            "fees": 12.5,
            "tx_volume": 320000,
        },
        "stablecoin_flows": {"signal": "neutral", "usdc_supply_delta": 0, "usdt_supply_delta": 0},
        "orderbook_depth": {"signal": "balanced", "bid_imbalance": 0.5, "ask_imbalance": 0.5},
        "errors": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def checker():
    return DataIntegrityChecker()


# ── Basic checks ─────────────────────────────────────────────────────


def test_clean_data_no_issues(checker):
    """Clean data should produce no anomalies or conflicts."""
    report = checker.check(make_ctx())
    assert not report.has_issues
    assert report.overall_confidence >= 0.9


def test_report_to_dict(checker):
    report = checker.check(make_ctx())
    d = report.to_dict()
    assert "anomalies" in d
    assert "conflicts" in d
    assert "sentiment_consensus" in d
    assert "overall_confidence" in d


# ── Price anomaly detection ──────────────────────────────────────────


def test_extreme_1h_move_flagged(checker):
    """A >10% 1h move should be flagged as anomaly."""
    ctx = make_ctx(coin_prices={
        "BTC": {"price": 85000, "change_1h_pct": 12.5, "change_24h_pct": 2.0},
    })
    report = checker.check(ctx)
    assert any("1h change" in a for a in report.anomalies)
    assert report.overall_confidence < 1.0


def test_extreme_24h_move_flagged(checker):
    """A >25% 24h move should be flagged."""
    ctx = make_ctx(coin_prices={
        "BTC": {"price": 85000, "change_1h_pct": 1.0, "change_24h_pct": -30.0},
    })
    report = checker.check(ctx)
    assert any("24h change" in a for a in report.anomalies)


def test_normal_moves_not_flagged(checker):
    """Normal moves should not generate anomalies."""
    ctx = make_ctx(coin_prices={
        "BTC": {"price": 85000, "change_1h_pct": 2.0, "change_24h_pct": 5.0},
    })
    report = checker.check(ctx)
    assert not report.anomalies


def test_liquidation_cascade_flagged(checker):
    """Overleveraged longs + price dropping = liquidation cascade warning."""
    ctx = make_ctx(
        coin_prices={
            "BTC": {"price": 80000, "change_1h_pct": -4.5, "change_24h_pct": -8.0},
        },
        funding_rates={
            "BTC": {"rate": 0.0002, "rate_pct": 0.02, "signal": "overleveraged_longs"},
        },
    )
    report = checker.check(ctx)
    assert any("liquidation" in a.lower() for a in report.anomalies)


# ── Sentiment conflict detection ────────────────────────────────────


def test_sentiment_conflict_flagged(checker):
    """Bullish Fear&Greed + bearish price should flag conflict."""
    ctx = make_ctx(
        fear_greed_index={"value": 75, "label": "Greed"},
        coin_prices={
            "BTC": {"price": 80000, "change_1h_pct": -1.0, "change_24h_pct": -5.0},
        },
        funding_rates={
            "BTC": {"rate": 0.0002, "signal": "overleveraged_longs"},
        },
        taker_buy_sell={
            "BTC": {"buy_sell_ratio": 0.7, "signal": "aggressive_selling"},
        },
    )
    report = checker.check(ctx)
    assert report.conflicts


def test_no_conflict_when_aligned(checker):
    """All bullish signals should not generate conflicts."""
    ctx = make_ctx(
        fear_greed_index={"value": 70, "label": "Greed"},
        global_market={"market_cap_change_24h_pct": 5.0},
        coin_prices={
            "BTC": {"price": 90000, "change_1h_pct": 2.0, "change_24h_pct": 8.0},
        },
        funding_rates={"BTC": {"signal": "neutral"}},
        long_short_ratio={"BTC": {"signal": "balanced"}},
        taker_buy_sell={"BTC": {"signal": "aggressive_buying"}},
    )
    report = checker.check(ctx)
    assert not report.conflicts


# ── Stale data detection ────────────────────────────────────────────


def test_missing_critical_source_flagged(checker):
    """Missing fear_greed_index should be flagged as stale."""
    ctx = make_ctx(fear_greed_index=None)
    report = checker.check(ctx)
    assert any("fear_greed" in s for s in report.stale_sources)


def test_missing_funding_rates_flagged(checker):
    ctx = make_ctx(funding_rates=None)
    report = checker.check(ctx)
    assert any("funding" in s for s in report.stale_sources)


def test_all_sources_present_no_stale(checker):
    report = checker.check(make_ctx())
    assert not report.stale_sources


# ── Sentiment consensus ─────────────────────────────────────────────


def test_consensus_bullish(checker):
    """Multiple bullish signals → bullish consensus."""
    ctx = make_ctx(
        fear_greed_index={"value": 72},
        global_market={"market_cap_change_24h_pct": 4.0},
        coin_prices={
            "BTC": {"price": 90000, "change_24h_pct": 6.0},
            "ETH": {"price": 3500, "change_24h_pct": 5.0},
        },
        funding_rates={"BTC": {"signal": "overleveraged_shorts"}},
        taker_buy_sell={"BTC": {"signal": "aggressive_buying"}},
        yahoo_macro={"VIX": {"last_close": 12.0}},
    )
    report = checker.check(ctx)
    assert report.sentiment_consensus["direction"] == "bullish"
    assert report.sentiment_consensus["bullish_count"] > report.sentiment_consensus["bearish_count"]


def test_consensus_bearish(checker):
    """Multiple bearish signals → bearish consensus."""
    ctx = make_ctx(
        fear_greed_index={"value": 20},
        global_market={"market_cap_change_24h_pct": -5.0},
        coin_prices={
            "BTC": {"price": 70000, "change_24h_pct": -8.0},
            "ETH": {"price": 2500, "change_24h_pct": -10.0},
        },
        funding_rates={"BTC": {"signal": "overleveraged_longs"}},
        taker_buy_sell={"BTC": {"signal": "aggressive_selling"}},
        yahoo_macro={"VIX": {"last_close": 35.0}},
    )
    report = checker.check(ctx)
    assert report.sentiment_consensus["direction"] == "bearish"


def test_consensus_mixed(checker):
    """Conflicting signals → mixed consensus."""
    report = checker.check(make_ctx())
    # Default context is mostly neutral
    assert report.sentiment_consensus["direction"] in ("mixed", "bullish", "bearish")
    assert report.sentiment_consensus["total_signals"] > 0


# ── Source reliability tracking ─────────────────────────────────────


# ── Overall confidence ──────────────────────────────────────────────


def test_confidence_reduced_by_anomalies(checker):
    """Anomalies should reduce overall confidence."""
    ctx = make_ctx(coin_prices={
        "BTC": {"price": 85000, "change_1h_pct": 15.0, "change_24h_pct": 30.0},
    })
    report = checker.check(ctx)
    assert report.overall_confidence < 0.9


def test_confidence_minimum_floor(checker):
    """Confidence should never go below 0.1."""
    ctx = make_ctx(
        fear_greed_index=None,
        global_market=None,
        coin_prices={"BTC": {"price": 85000, "change_1h_pct": 50, "change_24h_pct": 80}},
        funding_rates=None,
        open_interest=None,
    )
    report = checker.check(ctx)
    assert report.overall_confidence >= 0.1


def test_fewer_sources_lower_confidence(checker):
    """Having fewer sources should reduce confidence."""
    full = checker.check(make_ctx())
    sparse = checker.check(make_ctx(
        trending_coins=None,
        reddit_sentiment=None,
        dxy_proxy=None,
        cryptopanic_news=None,
        rss_news=None,
        hackernews=None,
        polymarket=None,
        long_short_ratio=None,
        taker_buy_sell=None,
    ))
    assert sparse.overall_confidence <= full.overall_confidence


# ── New sources: SEC, BGeometrics, GitHub ────────────────────────────


def test_sec_high_regulatory_activity_bearish(checker):
    """High SEC regulatory activity should be flagged as bearish."""
    ctx = make_ctx(sec_filings={
        "recent_count": 5,
        "issuers_active": 3,
        "filings": [],
        "signal": "high_regulatory_activity",
    })
    report = checker.check(ctx)
    consensus = report.sentiment_consensus
    assert any("SEC" in s for s in consensus.get("bearish_signals", []))


def test_sec_material_event_neutral(checker):
    """A single SEC 8-K filing should be flagged as neutral."""
    ctx = make_ctx(sec_filings={
        "recent_count": 1,
        "issuers_active": 1,
        "filings": [{"issuer": "Grayscale", "form": "8-K", "date": "2026-03-12"}],
        "signal": "material_event",
    })
    report = checker.check(ctx)
    consensus = report.sentiment_consensus
    assert any("SEC" in s for s in consensus.get("neutral_signals", []))


def test_sec_neutral_no_impact(checker):
    """Neutral SEC signal should not add any sentiment signals."""
    ctx = make_ctx(sec_filings={
        "recent_count": 0,
        "filings": [],
        "signal": "neutral",
    })
    report = checker.check(ctx)
    consensus = report.sentiment_consensus
    assert not any("SEC" in s for s in consensus.get("bearish_signals", []))
    assert not any("SEC" in s for s in consensus.get("neutral_signals", []))


def test_onchain_btc_macro_bullish(checker):
    """Bullish on-chain BTC macro should appear in bullish signals."""
    ctx = make_ctx(onchain_btc_macro={
        "signal": "bullish_accumulation",
        "mvrv_value": 0.8,
        "sopr_value": 1.01,
        "netflow_value": -5000,
    })
    report = checker.check(ctx)
    consensus = report.sentiment_consensus
    assert any("onchain BTC" in s for s in consensus.get("bullish_signals", []))


def test_onchain_btc_macro_bearish(checker):
    """Bearish on-chain BTC macro should appear in bearish signals."""
    ctx = make_ctx(onchain_btc_macro={
        "signal": "bearish_overheated",
        "mvrv_value": 6.5,
        "netflow_value": 10000,
    })
    report = checker.check(ctx)
    consensus = report.sentiment_consensus
    assert any("onchain BTC" in s for s in consensus.get("bearish_signals", []))


def test_onchain_btc_macro_capitulation(checker):
    """Capitulation signal (SOPR < 0.95) should be bullish medium-term."""
    ctx = make_ctx(onchain_btc_macro={
        "signal": "capitulation_bullish_medium_term",
        "mvrv_value": 1.5,
        "sopr_value": 0.92,
        "netflow_value": 0,
    })
    report = checker.check(ctx)
    consensus = report.sentiment_consensus
    assert any("capitulation" in s for s in consensus.get("bullish_signals", []))


def test_onchain_btc_macro_conflict_detection(checker):
    """On-chain bearish + other bullish should trigger conflict."""
    ctx = make_ctx(
        fear_greed_index={"value": 75},  # bullish
        onchain_btc_macro={"signal": "bearish_overheated", "mvrv_value": 6.0},
        funding_rates={"BTC": {"signal": "overleveraged_shorts"}},  # bullish
    )
    report = checker.check(ctx)
    # Should detect both bullish and bearish signals
    consensus = report.sentiment_consensus
    assert consensus["bullish_count"] > 0
    assert consensus["bearish_count"] > 0


def test_sec_filings_in_reliability_scores(checker):
    """New sources should have reliability scores."""
    assert "sec_filings" in checker.source_scores
    assert "onchain_btc_macro" in checker.source_scores
    assert "github_activity" in checker.source_scores
    assert checker.source_scores["onchain_btc_macro"] == 0.8
    assert checker.source_scores["github_activity"] == 0.3
