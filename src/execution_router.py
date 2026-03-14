"""Execution router — venue selection and slippage tracking.

Currently always routes to Hyperliquid (primary venue). The abstraction exists
so that when capital scales, adding multi-venue execution is a config change,
not an architecture change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import ExecutionRouterConfig
from src.dex_scanner import DexScanner, DexScanResult
from src.exchange import Exchange, parse_pair
from src.utils import log, now_iso


@dataclass
class VenueScore:
    venue: str
    effective_price: float
    taker_fee_bps: float
    spread_bps: float
    depth_at_size: float
    latency_ms: float
    score: float  # Lower = better for buys, higher = better for sells


@dataclass
class ExecutionPlan:
    venue: str
    pair: str
    side: str
    amount: float
    expected_price: float
    expected_fee_bps: float
    expected_slippage_bps: float
    reasoning: str
    alternatives: list[VenueScore] = field(default_factory=list)
    mid_market_price: float | None = None  # For slippage tracking


class ExecutionRouter:
    """Route trades to the best venue. Currently Hyperliquid-only."""

    def __init__(
        self,
        exchange: Exchange,
        config: ExecutionRouterConfig,
        dex_scanner: DexScanner | None = None,
    ):
        self._exchange = exchange
        self._config = config
        self._dex_scanner = dex_scanner

    async def plan_execution(self, pair: str, side: str, amount: float) -> ExecutionPlan:
        """Determine best venue and expected costs."""
        # Get current mid-market price for slippage baseline
        mid_price = None
        try:
            ticker = await self._exchange.fetch_ticker(pair)
            mid_price = ticker.get("last")
        except Exception:
            pass

        # Check DEX scanner for cross-venue data
        alternatives: list[VenueScore] = []
        best_venue = self._config.primary_venue
        reasoning = f"Primary venue ({self._config.primary_venue})"

        if self._dex_scanner and self._config.enable_multi_venue:
            try:
                scan = await self._dex_scanner.scan_pair(pair)
                scanner_best = self._dex_scanner.get_best_venue(pair, side)

                # Build venue scores from scan data
                for venue_name, snap in scan.venue_snapshots.items():
                    price = snap.best_ask if side == "buy" else snap.best_bid
                    if price and price > 0:
                        alternatives.append(VenueScore(
                            venue=venue_name,
                            effective_price=price,
                            taker_fee_bps=snap.taker_fee_bps,
                            spread_bps=snap.spread_bps or 0,
                            depth_at_size=0,  # Future: estimate from order book
                            latency_ms=snap.latency_ms,
                            score=price if side == "buy" else -price,
                        ))

                if scanner_best and scanner_best != self._config.primary_venue:
                    # Only switch venue if savings justify it
                    primary_snap = scan.venue_snapshots.get(self._config.primary_venue)
                    alt_snap = scan.venue_snapshots.get(scanner_best)
                    if primary_snap and alt_snap:
                        primary_price = primary_snap.best_ask if side == "buy" else primary_snap.best_bid
                        alt_price = alt_snap.best_ask if side == "buy" else alt_snap.best_bid
                        if primary_price and alt_price and primary_price > 0:
                            savings_bps = abs(primary_price - alt_price) / primary_price * 10000
                            # Need at least 5bps savings to justify switching
                            if savings_bps > 5:
                                best_venue = scanner_best
                                reasoning = (
                                    f"Better price on {scanner_best}: "
                                    f"saves {savings_bps:.1f}bps vs {self._config.primary_venue}"
                                )
            except Exception as e:
                log.debug(f"DEX scan for routing failed (using primary): {e}")

        return ExecutionPlan(
            venue=best_venue,
            pair=pair,
            side=side,
            amount=amount,
            expected_price=mid_price or 0,
            expected_fee_bps=self._get_venue_fee(best_venue),
            expected_slippage_bps=5.0,  # Conservative estimate
            reasoning=reasoning,
            alternatives=alternatives,
            mid_market_price=mid_price,
        )

    async def execute(self, plan: ExecutionPlan) -> dict:
        """Execute the plan on the selected venue.

        Currently only supports Hyperliquid. Returns the order result dict.
        """
        if plan.venue != "hyperliquid":
            if not self._config.enable_multi_venue:
                raise ValueError(
                    f"Venue {plan.venue} not supported — multi-venue is disabled. "
                    f"Set execution_router.enable_multi_venue: true to allow."
                )
            log.warning(
                f"Venue {plan.venue} not yet implemented for execution, "
                f"falling back to hyperliquid"
            )

        order = await self._exchange.create_market_order(plan.pair, plan.side, plan.amount)

        # Track actual slippage
        if plan.mid_market_price and plan.mid_market_price > 0:
            fill_price = order.get("price", 0)
            if fill_price > 0:
                if plan.side == "buy":
                    slippage_bps = (fill_price - plan.mid_market_price) / plan.mid_market_price * 10000
                else:
                    slippage_bps = (plan.mid_market_price - fill_price) / plan.mid_market_price * 10000
                order["slippage_actual_bps"] = round(slippage_bps, 2)

        order["execution_venue"] = plan.venue
        order["routing_reasoning"] = plan.reasoning
        return order

    def _get_venue_fee(self, venue: str) -> float:
        """Look up taker fee for a venue from scanner config or default."""
        if self._dex_scanner:
            return self._dex_scanner._venue_fees.get(venue, 4.5)
        return 4.5  # Default Hyperliquid fee

    def calculate_slippage(self, fill_price: float, mid_price: float, side: str) -> float | None:
        """Calculate actual slippage in basis points."""
        if not mid_price or mid_price <= 0 or not fill_price or fill_price <= 0:
            return None
        if side == "buy":
            return (fill_price - mid_price) / mid_price * 10000
        return (mid_price - fill_price) / mid_price * 10000
