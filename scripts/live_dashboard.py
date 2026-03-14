"""Live Dashboard — FastAPI + WebSocket real-time trading dashboard.

Shows portfolio, trades, equity curve, model accuracy, negotiation debates,
strategy evolution leaderboard, and live alert feed.

Usage:
    python scripts/live_dashboard.py
    python scripts/live_dashboard.py --port 8421
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.config import load_config
from src.database import Database
from src.journal import Journal
from src.models import PortfolioSnapshot, Signal
from src.negotiation import NegotiationEngine
from src.providers import build_providers

app = FastAPI(title="Finance Bro Live Dashboard")
db: Database | None = None
config = None
negotiation_engine: NegotiationEngine | None = None

# Serve static fonts
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── REST Endpoints (ported from web_dashboard.py + new) ──────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.get("/api/portfolio")
async def get_portfolio():
    curve = db.get_equity_curve(days=1)
    if not curve:
        return {
            "total_value": config.risk.starting_capital,
            "cash": config.risk.starting_capital,
            "positions_value": 0, "drawdown_pct": 0,
            "daily_pnl": 0, "total_pnl": 0, "total_pnl_pct": 0,
            "open_positions": 0, "high_water_mark": config.risk.starting_capital,
            "bot_balance": 0, "user_balance": 0,
        }
    snap = curve[-1]
    return {
        "total_value": snap.total_value, "cash": snap.cash,
        "positions_value": snap.positions_value,
        "open_positions": snap.open_positions,
        "drawdown_pct": snap.drawdown_pct,
        "high_water_mark": snap.high_water_mark,
        "daily_pnl": snap.daily_pnl, "total_pnl": snap.total_pnl,
        "total_pnl_pct": snap.total_pnl_pct,
        "timestamp": snap.timestamp,
        "bot_balance": snap.bot_balance,
        "user_balance": snap.user_balance,
    }


@app.get("/api/trades")
async def get_trades(n: int = 20):
    n = max(1, min(100, n))  # [SEC] Bound query param
    trades = db.get_recent_trades(n)
    return [
        {
            "id": t.id, "pair": t.pair, "direction": t.direction,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "quantity": t.quantity, "stop_loss": t.stop_loss,
            "take_profit": t.take_profit, "status": t.status,
            "pnl": t.pnl, "pnl_pct": t.pnl_pct, "fees": t.fees,
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "ai_reasoning": t.ai_reasoning,
            "strategy": t.signal_data.get("strategy", ""),
        }
        for t in trades
    ]


@app.get("/api/equity")
async def get_equity(days: int = 7):
    days = max(1, min(90, days))  # [SEC] Bound query param
    curve = db.get_equity_curve(days)
    return [
        {"timestamp": s.timestamp, "value": s.total_value, "drawdown": s.drawdown_pct}
        for s in curve
    ]


@app.get("/api/signals")
async def get_signals():
    rows = db.conn.execute(
        "SELECT * FROM signals ORDER BY id DESC LIMIT 30"
    ).fetchall()
    return [
        {
            "pair": r["pair"], "direction": r["direction"],
            "confidence": r["confidence"], "strategy": r["strategy_name"],
            "acted_on": bool(r["acted_on"]),
            "reason": r["block_reason"] or "", "timestamp": r["timestamp"],
        }
        for r in rows
    ]


@app.get("/api/performance")
async def get_performance():
    j = Journal(db)
    return j.get_performance_summary(days=30)


@app.get("/api/status")
async def get_status():
    status = db.get_state("agent_status") or "RUNNING"
    trades_since_review = db.get_state("trades_since_review") or "0"
    cycle_count = db.get_state("cycle_count") or "0"
    return {
        "status": status,
        "mode": config.agent.mode,
        "trades_since_review": int(trades_since_review),
        "cycle_count": int(cycle_count),
        "pairs": config.agent.pairs,
        "cycle_sec": config.agent.cycle_interval_sec,
    }


@app.get("/api/negotiations")
async def get_negotiations():
    """Return recent negotiation records from agent_state."""
    rows = db.conn.execute(
        "SELECT key, value FROM agent_state WHERE key LIKE 'negotiation_%' "
        "ORDER BY key DESC LIMIT 20"
    ).fetchall()
    records = []
    for r in rows:
        try:
            records.append(json.loads(r["value"]))
        except (json.JSONDecodeError, TypeError):
            pass
    return records


@app.get("/api/model-accuracy")
async def get_model_accuracy():
    """Return per-model voting accuracy stats."""
    from src.multi_brain import ModelAccuracyTracker
    tracker = ModelAccuracyTracker(db)
    stats = tracker.get_model_stats()
    weights = tracker.get_model_weights()
    best = tracker.get_best_model()
    return {
        "models": stats,
        "weights": weights,
        "best_model": best,
    }


@app.get("/api/evolved-strategies")
async def get_evolved_strategies():
    """Return strategy evolution leaderboard."""
    try:
        db.init_evolved_strategies_table()
        strategies = db.get_evolved_strategies()
        return [
            {
                "name": m.name, "generation": m.generation,
                "status": m.status, "total_trades": m.total_trades,
                "wins": m.wins, "losses": m.losses,
                "win_rate": m.win_rate, "total_pnl": m.total_pnl,
                "created_at": m.created_at, "rationale": m.rationale,
            }
            for m in strategies
        ]
    except Exception:
        return []


# ── Demo Trigger ──────────────────────────────────────────────────────

# Allowed pairs for demo triggers — prevents arbitrary pair injection
_DEMO_PAIRS = {
    "BTC", "ETH", "AAVE", "TSLA", "NVDA", "MSFT", "AAPL",
    "GOLD", "SILVER", "OIL", "TSM", "EUR",
}

_PAIR_MAP = {
    "BTC": "BTC/USDC:USDC", "ETH": "ETH/USDC:USDC", "AAVE": "AAVE/USDC:USDC",
    "TSLA": "XYZ-TSLA/USDC:USDC", "NVDA": "XYZ-NVDA/USDC:USDC",
    "MSFT": "XYZ-MSFT/USDC:USDC", "AAPL": "XYZ-AAPL/USDC:USDC",
    "GOLD": "XYZ-GOLD/USDC:USDC", "SILVER": "XYZ-SILVER/USDC:USDC",
    "OIL": "XYZ-BRENTOIL/USDC:USDC", "TSM": "XYZ-TSM/USDC:USDC",
    "EUR": "XYZ-EUR/USDC:USDC",
}


@app.post("/api/trigger")
async def trigger_negotiation(
    pair: str = "BTC",
    direction: str = "long",
    confidence: float = 0.75,
    strategy: str = "momentum",
):
    """Inject a test signal and run it through the negotiation engine.

    Usage:
        curl -X POST "http://localhost:8421/api/trigger?pair=BTC&direction=long&confidence=0.75"
    """
    if negotiation_engine is None:
        return {"error": "Negotiation engine not initialized — no AI providers configured"}

    # Validate inputs
    pair_upper = pair.upper()
    if pair_upper not in _DEMO_PAIRS:
        return {"error": f"Invalid pair. Use one of: {', '.join(sorted(_DEMO_PAIRS))}"}
    direction = direction.lower()
    if direction not in ("long", "short"):
        return {"error": "Direction must be 'long' or 'short'"}
    confidence = max(0.3, min(0.99, confidence))

    full_pair = _PAIR_MAP[pair_upper]

    signal = Signal(
        pair=full_pair,
        timeframe="15m",
        direction=direction,
        confidence=confidence,
        strategy_name=strategy,
        indicators={"rsi": 32 if direction == "long" else 72, "macd_hist": 0.5, "demo": True},
        reasoning=f"Demo trigger: {pair_upper} {direction.upper()} at {confidence:.0%} confidence",
    )

    # Build a portfolio snapshot from latest DB data
    curve = db.get_equity_curve(days=1)
    if curve:
        snap = curve[-1]
    else:
        snap = PortfolioSnapshot(
            timestamp="", total_value=config.risk.starting_capital,
            cash=config.risk.starting_capital, positions_value=0,
            open_positions=0, drawdown_pct=0, high_water_mark=config.risk.starting_capital,
            daily_pnl=0, total_pnl=0, total_pnl_pct=0,
        )

    # Run negotiation (this will take 10-30s as agents debate)
    try:
        sig_out, record, approved = await asyncio.wait_for(
            negotiation_engine.negotiate_signal(
                signal, snap, {"demo_mode": True, "note": "Manually triggered for demo"}
            ),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        return {"error": "Negotiation timed out after 90s — LLM providers may be slow"}
    except Exception as e:
        return {"error": f"Negotiation failed: {str(e)[:200]}"}

    return {
        "approved": approved,
        "consensus_type": record.consensus_type,
        "total_rounds": record.total_rounds,
        "vetoed_by": record.vetoed_by,
        "final_size_modifier": record.final_size_modifier,
        "final_reasoning": record.final_reasoning,
        "opinions": [
            {
                "agent": op.agent_name,
                "approved": op.approved,
                "risk_score": op.risk_score,
                "size_modifier": op.size_modifier,
                "reasoning": op.reasoning,
                "vetoed": op.vetoed,
            }
            for op in record.initial_opinions
        ],
        "rounds": [
            {
                "round": r.round_num,
                "speaker": r.speaker,
                "action": r.action,
                "content": r.content,
            }
            for r in record.rounds
            if r.round_num > 0
        ],
    }


# ── WebSocket ─────────────────────────────────────────────────────────

connected_clients: set[WebSocket] = set()
MAX_WS_CLIENTS = 20  # [SEC] Prevent WebSocket connection exhaustion


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    if len(connected_clients) >= MAX_WS_CLIENTS:
        await websocket.close(code=1013, reason="Too many connections")
        return
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            try:
                data = {
                    "portfolio": await get_portfolio(),
                    "status": await get_status(),
                    "trades": await get_trades(10),
                    "signals": await get_signals(),
                    "negotiations": await get_negotiations(),
                    "model_accuracy": await get_model_accuracy(),
                    "evolved_strategies": await get_evolved_strategies(),
                }
            except Exception:
                data = {"error": "Failed to gather data"}
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except (WebSocketDisconnect, RuntimeError, Exception):
        connected_clients.discard(websocket)


# ── Dashboard HTML ────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finance Bro</title>
<style>
  @font-face { font-family: 'Redaction35'; src: url('/static/fonts/Redaction35-BWXjw.otf') format('opentype'); font-weight: normal; font-style: normal; }
  @font-face { font-family: 'Redaction35'; src: url('/static/fonts/Redaction35Bold-MV3LP.otf') format('opentype'); font-weight: bold; font-style: normal; }
  @font-face { font-family: 'Redaction35'; src: url('/static/fonts/Redaction35Italic-rgDrB.otf') format('opentype'); font-weight: normal; font-style: italic; }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  /* ── Theme Variables ───────────────────────────── */
  :root {
    --bg: #fff;
    --fg: #000;
    --bg-alt: #f5f5f5;
    --border: #000;
    --dot: 1px dotted #000;

    /* Agent colors */
    --alpha: #2255cc;
    --beta: #118844;
    --gamma: #cc6600;
    --delta: #cc2244;

    /* Trade colors */
    --long: #118844;
    --short: #cc2244;
    --profit: #118844;
    --loss: #cc2244;
    --warn: #cc8800;
  }

  [data-theme="dark"] {
    --bg: #1a1a1e;
    --fg: #c8c8c8;
    --bg-alt: #222226;
    --border: #555;
    --dot: 1px dotted #555;

    --alpha: #6699ff;
    --beta: #44cc88;
    --gamma: #ffaa44;
    --delta: #ff5577;

    --long: #44cc88;
    --short: #ff5577;
    --profit: #44cc88;
    --loss: #ff5577;
    --warn: #ffcc44;
  }

  body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Redaction35', 'Chicago', 'Geneva', serif;
    font-size: 13px;
    padding: 0;
    transition: background 0.15s, color 0.15s;
  }

  /* ── Menu Bar ──────────────────────────────────── */
  .menubar {
    background: var(--bg);
    border-bottom: 2px solid var(--border);
    padding: 4px 12px;
    display: flex;
    align-items: center;
    gap: 0;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .menubar-logo {
    font-size: 16px;
    font-weight: bold;
    margin-right: 16px;
    letter-spacing: -0.5px;
  }
  .menubar-item {
    padding: 2px 12px;
    font-size: 13px;
    font-weight: bold;
    cursor: default;
    border: 1px solid transparent;
  }
  .menubar-item:hover { background: var(--fg); color: var(--bg); }
  .menubar-item.toggle-theme {
    cursor: pointer;
    border: 1px solid var(--border);
    margin-left: 8px;
    font-size: 12px;
    padding: 2px 10px;
  }
  .menubar-right {
    margin-left: auto;
    font-size: 12px;
    font-weight: normal;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .desktop {
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 960px;
    margin: 0 auto;
  }

  /* ── System 6 Window ───────────────────────────── */
  .win {
    border: 2px solid var(--border);
    background: var(--bg);
  }
  .win-title {
    background: var(--bg);
    border-bottom: 2px solid var(--border);
    padding: 3px 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: default;
    user-select: none;
  }
  .win-close {
    width: 14px; height: 14px;
    border: 2px solid var(--border);
    display: inline-block;
    flex-shrink: 0;
  }
  .win-title-text {
    flex: 1;
    text-align: center;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .win-stripes {
    flex: 0 0 40px;
    height: 14px;
    background: repeating-linear-gradient(
      0deg,
      var(--border) 0px, var(--border) 2px,
      var(--bg) 2px, var(--bg) 4px
    );
  }
  .win-body {
    padding: 10px;
  }

  /* Title styling for Finance Bro header */
  .main-title {
    font-style: italic;
    font-size: 22px;
    letter-spacing: 1px;
  }

  /* ── Stats ─────────────────────────────────────── */
  .stat {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: var(--dot);
  }
  .stat:last-child { border-bottom: none; }
  .stat-label { font-size: 12px; }
  .stat-value { font-weight: bold; font-size: 13px; }
  .positive { color: var(--profit); }
  .negative { color: var(--loss); }
  .neutral { color: var(--warn); }

  /* ── Tables ────────────────────────────────────── */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    text-align: left;
    padding: 4px 6px;
    border-bottom: 2px solid var(--border);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  td {
    padding: 4px 6px;
    border-bottom: 1px solid var(--border);
  }

  /* ── Badges ────────────────────────────────────── */
  .badge {
    padding: 1px 6px;
    font-size: 10px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 1px solid;
  }
  .badge-long { background: var(--long); color: #111; border-color: var(--long); font-weight: bold; }
  .badge-short { background: var(--short); color: #111; border-color: var(--short); font-weight: bold; }
  .badge-open { background: var(--fg); color: var(--bg); border-color: var(--fg); }
  .badge-closed { border-color: var(--profit); color: var(--profit); }
  .badge-stopped { background: var(--loss); color: var(--bg); border-color: var(--loss); }

  /* ── Status Bar (bottom) ───────────────────────── */
  .statusbar {
    border-top: 2px solid var(--border);
    padding: 3px 12px;
    display: flex;
    gap: 2px;
    font-size: 11px;
    background: var(--bg);
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 100;
  }
  .statusbar-cell {
    padding: 1px 10px;
    border-right: 1px solid var(--border);
  }
  .statusbar-cell:last-child { border-right: none; }

  /* ── Negotiation Feed ──────────────────────────── */
  .nego-card { max-height: 420px; overflow-y: auto; }
  .nego-entry {
    border: 2px solid var(--border);
    margin-bottom: 10px;
    padding: 0;
  }
  .nego-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    font-weight: bold;
    border-bottom: 2px solid var(--border);
    padding: 6px 10px;
    background: var(--bg-alt);
  }
  .nego-pair {
    font-weight: bold;
    font-size: 16px;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .nego-meta { font-size: 11px; font-weight: normal; }
  .nego-verdict {
    font-weight: bold;
    font-size: 12px;
    padding: 2px 10px;
    border: 2px solid;
    letter-spacing: 1px;
  }
  .nego-approved { border-color: var(--profit); color: var(--profit); }
  .nego-rejected { background: var(--loss); color: var(--bg); border-color: var(--loss); }
  .nego-vetoed { background: var(--loss); color: var(--bg); border-color: var(--loss); text-decoration: line-through; }
  .nego-body { padding: 8px 10px; }
  .nego-section-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: bold;
    padding: 4px 0 2px;
    margin-top: 6px;
    border-top: 1px solid var(--border);
    color: var(--fg);
    opacity: 0.5;
  }
  .nego-section-label:first-child { border-top: none; margin-top: 0; }
  .nego-opinion {
    font-size: 12px;
    padding: 4px 0;
    border-bottom: var(--dot);
  }
  .nego-opinion:last-child { border-bottom: none; }
  .agent-name { font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
  .nego-timestamp { font-size: 10px; padding: 4px 10px; border-top: 1px solid var(--border); opacity: 0.4; }

  /* Agent colors */
  .agent-alpha { color: var(--alpha); }
  .agent-beta { color: var(--beta); }
  .agent-gamma { color: var(--gamma); }
  .agent-delta { color: var(--delta); }

  /* Vote colors */
  .vote-approve { color: var(--profit); font-weight: bold; }
  .vote-reject { color: var(--loss); font-weight: bold; }
  .vote-veto { color: var(--delta); font-weight: bold; text-decoration: underline; }

  /* ── Model Accuracy ────────────────────────────── */
  .accuracy-bar { display: flex; align-items: center; gap: 6px; padding: 3px 0; }
  .accuracy-name { width: 80px; font-size: 12px; font-weight: bold; }
  .accuracy-fill {
    height: 16px;
    min-width: 2px;
    transition: width 0.3s;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 4px;
    font-size: 11px;
    font-weight: bold;
    background: var(--fg);
    color: var(--bg);
  }

  .strat-active { font-weight: bold; color: var(--profit); }
  .strat-active::before { content: '* '; }
  .strat-candidate { font-style: italic; color: var(--warn); }
  .strat-retired { text-decoration: line-through; }

  /* ── Equity Chart ──────────────────────────────── */
  .chart-container {
    height: 120px;
    position: relative;
    padding: 4px 0;
    border: 1px solid var(--border);
    margin-top: 4px;
  }
  .chart-bar {
    position: absolute;
    bottom: 0;
    min-width: 2px;
    transition: height 0.3s;
  }
  .chart-bar-up { background: var(--profit); }
  .chart-bar-down { background: var(--loss); }

  .scroll-area { max-height: 280px; overflow-y: auto; }

  /* ── Layout spans (kept for compat) ──────────────── */
  .span-2, .span-3 { }

  /* ── Scrollbar (brutalist) ─────────────────────── */
  ::-webkit-scrollbar { width: 14px; }
  ::-webkit-scrollbar-track { background: var(--bg-alt); border-left: 2px solid var(--border); }
  ::-webkit-scrollbar-thumb { background: var(--bg); border: 2px solid var(--border); }

  /* ── Blinking cursor for live indicator ─────────── */
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
  .blink { animation: blink 1s step-end infinite; }

  .empty-msg { font-size: 12px; font-style: italic; padding: 8px 0; }

  /* ── Splash Screen ────────────────────────────── */
  #splash {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
  }
  #splash canvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
  }
  #splash.fade-out {
    animation: splashFade 0.6s ease-out forwards;
  }
  @keyframes splashFade {
    0% { opacity: 1; }
    100% { opacity: 0; }
  }
  .dashboard-hidden { opacity: 0; }
  .dashboard-visible { opacity: 1; transition: opacity 0.4s ease-in; }

  /* ── Responsive: stack on narrow screens ────────── */
  @media (max-width: 900px) {
    .desktop { padding: 8px; }
  }
</style>
</head>
<body>

<!-- Splash Screen -->
<div id="splash">
  <canvas id="splash-canvas"></canvas>
</div>

<!-- Menu Bar -->
<div class="menubar dashboard-hidden" id="main-ui">
  <span class="menubar-logo">$</span>
  <span class="menubar-item">File</span>
  <span class="menubar-item">View</span>
  <span class="menubar-item">Agents</span>
  <span class="menubar-item">Risk</span>
  <span class="menubar-item">Special</span>
  <span class="menubar-right">
    <span id="ws-status"><span class="blink">|</span> Connecting...</span>
    <span class="menubar-item toggle-theme" onclick="toggleTheme()">DARK</span>
  </span>
</div>

<!-- Desktop -->
<div class="desktop dashboard-hidden" id="main-desktop">

  <!-- Header Window -->
  <div class="win span-3">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text main-title">Finance Bro</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body" style="display:flex;justify-content:space-between;padding:6px 12px;">
      <span><span class="blink">|</span> <strong id="status-text">IDLE</strong> &mdash; <span id="mode-text">PAPER</span></span>
      <span>Cycle: <span id="cycle-text">-</span></span>
      <span>Pairs: <span id="pairs-text">-</span></span>
      <span id="status-dot" style="display:none;"></span>
    </div>
  </div>

  <!-- Portfolio -->
  <div class="win">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text">Portfolio</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body" id="portfolio-stats">Loading...</div>
  </div>

  <!-- Performance -->
  <div class="win">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text">Performance (30d)</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body" id="perf-stats">Loading...</div>
  </div>

  <!-- Equity Curve -->
  <div class="win">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text">Equity Curve</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body">
      <div class="chart-container" id="equity-chart"></div>
    </div>
  </div>

  <!-- Negotiations -->
  <div class="win span-2">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text">Agent Negotiations</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body nego-card" id="nego-feed">
      <div class="empty-msg">Waiting for negotiations...</div>
    </div>
  </div>

  <!-- Model Accuracy + Strategy Leaderboard -->
  <div class="win">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text">Model Accuracy</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body">
      <div id="model-accuracy"><span class="empty-msg">No data yet</span></div>
      <div style="border-top:2px solid #000;margin-top:8px;padding-top:8px;">
        <div style="font-weight:bold;text-transform:uppercase;letter-spacing:1px;font-size:11px;margin-bottom:6px;">Strategy Leaderboard</div>
        <div id="strat-leaderboard"><span class="empty-msg">No evolved strategies</span></div>
      </div>
    </div>
  </div>

  <!-- Trades -->
  <div class="win span-2">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text">Recent Trades</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body scroll-area">
      <table>
        <thead><tr><th>Time</th><th>Pair</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Status</th><th>Strategy</th></tr></thead>
        <tbody id="trades-body"><tr><td colspan="8" class="empty-msg">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Signals -->
  <div class="win">
    <div class="win-title">
      <span class="win-close"></span>
      <span class="win-stripes"></span>
      <span class="win-title-text">Recent Signals</span>
      <span class="win-stripes"></span>
    </div>
    <div class="win-body scroll-area">
      <table>
        <thead><tr><th>Time</th><th>Pair</th><th>Dir</th><th>Conf</th><th>Strategy</th><th>Acted</th></tr></thead>
        <tbody id="signals-body"><tr><td colspan="6" class="empty-msg">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

</div><!-- /desktop -->

<!-- Bottom Status Bar -->
<div class="statusbar dashboard-hidden" id="main-statusbar">
  <span class="statusbar-cell" id="sb-status"><span class="blink">|</span> LIVE</span>
  <span class="statusbar-cell" id="sb-mode">Paper Mode</span>
  <span class="statusbar-cell" id="sb-pairs">-</span>
  <span class="statusbar-cell" style="margin-left:auto;">Hyperliquid DEX &mdash; USDC Settlement</span>
</div>

<div style="height:28px;"></div><!-- spacer for fixed statusbar -->

<script>
const WS_URL = `ws://${window.location.host}/ws`;
let ws;
let reconnectTimer;

// ── Theme Toggle ──
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.querySelector('.toggle-theme').textContent = isDark ? 'DARK' : 'LIGHT';
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
}
// Restore saved theme
(function() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  if (saved === 'dark') {
    document.addEventListener('DOMContentLoaded', () => {
      const btn = document.querySelector('.toggle-theme');
      if (btn) btn.textContent = 'LIGHT';
    });
  }
})();

// [SEC] HTML-escape to prevent XSS from LLM-generated content
function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }
function pnlClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : 'neutral'; }
function fmt(v, d=2) { return v != null ? Number(v).toFixed(d) : '-'; }
function fmtPct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '-'; }
function fmtTime(t) {
  if (!t) return '-';
  const d = new Date(t);
  return d.toLocaleDateString('en-GB', {day:'2-digit',month:'short'}) + ' ' +
         d.toLocaleTimeString('en-GB', {hour:'2-digit',minute:'2-digit'});
}

function stat(label, value, cls='') {
  return '<div class="stat"><span class="stat-label">' + label + '</span><span class="stat-value ' + cls + '">' + value + '</span></div>';
}

function agentColor(name) {
  const m = { Alpha: 'agent-alpha', Beta: 'agent-beta', Gamma: 'agent-gamma', Delta: 'agent-delta' };
  return m[name] || '';
}

function agentIcon(name) {
  const icons = { Alpha: '\u03B1', Beta: '\u03B2', Gamma: '\u03B3', Delta: '\u03B4' };
  return icons[name] || '?';
}

function voteClass(op) {
  return op.vetoed ? 'vote-veto' : op.approved ? 'vote-approve' : 'vote-reject';
}

function connectWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    document.getElementById('ws-status').innerHTML = '<span class="blink">|</span> LIVE';
    clearTimeout(reconnectTimer);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderAll(data);
  };

  ws.onclose = () => {
    document.getElementById('ws-status').innerHTML = 'DISCONNECTED &mdash; reconnecting...';
    reconnectTimer = setTimeout(connectWS, 3000);
  };

  ws.onerror = () => { ws.close(); };
}

function renderAll(data) {
  renderStatus(data.status);
  renderPortfolio(data.portfolio);
  renderPerformance();
  renderTrades(data.trades);
  renderSignals(data.signals);
  renderNegotiations(data.negotiations);
  renderModelAccuracy(data.model_accuracy);
  renderLeaderboard(data.evolved_strategies);
}

function renderStatus(s) {
  if (!s) return;
  document.getElementById('status-text').textContent = s.status;
  document.getElementById('mode-text').textContent = s.mode.toUpperCase();
  document.getElementById('cycle-text').textContent = s.cycle_sec + 's (#' + s.cycle_count + ')';
  document.getElementById('pairs-text').textContent = (s.pairs || []).length;
  document.getElementById('sb-status').innerHTML = '<span class="blink">|</span> ' + esc(s.status);
  document.getElementById('sb-mode').textContent = s.mode.toUpperCase() + ' Mode';
  document.getElementById('sb-pairs').textContent = (s.pairs || []).length + ' pairs active';
}

function renderPortfolio(p) {
  if (!p) return;
  document.getElementById('portfolio-stats').innerHTML = [
    stat('Total Value', '$' + fmt(p.total_value), pnlClass(p.total_pnl)),
    stat('Cash', '$' + fmt(p.cash)),
    stat('Positions', '$' + fmt(p.positions_value)),
    stat('Drawdown', fmtPct(p.drawdown_pct)),
    stat('Daily P&L', '$' + fmt(p.daily_pnl), pnlClass(p.daily_pnl)),
    stat('Total P&L', '$' + fmt(p.total_pnl) + ' (' + fmtPct(p.total_pnl_pct) + ')', pnlClass(p.total_pnl)),
    stat('Bot Pool', '$' + fmt(p.bot_balance)),
    stat('User Pool', '$' + fmt(p.user_balance)),
  ].join('');
}

async function renderPerformance() {
  try {
    const perf = await (await fetch('/api/performance')).json();
    if (perf && perf.total_trades > 0) {
      document.getElementById('perf-stats').innerHTML = [
        stat('Total Trades', perf.total_trades),
        stat('Win Rate', fmtPct(perf.win_rate)),
        stat('Wins / Losses', perf.wins + ' / ' + perf.losses),
        stat('Total P&L', '$' + fmt(perf.total_pnl), pnlClass(perf.total_pnl)),
        stat('Profit Factor', fmt(perf.profit_factor)),
      ].join('');
    }
  } catch(e) {}
}

function renderTrades(trades) {
  if (!trades || trades.length === 0) {
    document.getElementById('trades-body').innerHTML = '<tr><td colspan="8" class="empty-msg">No trades yet</td></tr>';
    return;
  }
  document.getElementById('trades-body').innerHTML = trades.map(t => {
    const dirBadge = '<span class="badge badge-' + t.direction + '">' + t.direction.toUpperCase() + '</span>';
    const sBadge = '<span class="badge badge-' + (t.status === 'open' ? 'open' : t.pnl >= 0 ? 'closed' : 'stopped') + '">' + t.status + '</span>';
    const pnl = t.pnl != null ? '<span class="' + pnlClass(t.pnl) + '">$' + fmt(t.pnl) + '</span>' : '-';
    return '<tr><td>' + fmtTime(t.entry_time) + '</td><td>' + esc(t.pair.split('/')[0]) + '</td><td>' + dirBadge + '</td><td>' + fmt(t.entry_price) + '</td><td>' + (t.exit_price ? fmt(t.exit_price) : '-') + '</td><td>' + pnl + '</td><td>' + sBadge + '</td><td>' + esc(t.strategy) + '</td></tr>';
  }).join('');
}

function renderSignals(signals) {
  if (!signals || signals.length === 0) return;
  document.getElementById('signals-body').innerHTML = signals.slice(0, 15).map(s => {
    const dirBadge = '<span class="badge badge-' + s.direction + '">' + s.direction.toUpperCase() + '</span>';
    const acted = s.acted_on ? '<span class="positive">Y</span>' : '<span class="negative">N</span>';
    return '<tr><td>' + fmtTime(s.timestamp) + '</td><td>' + esc(s.pair.split('/')[0]) + '</td><td>' + dirBadge + '</td><td>' + fmt(s.confidence) + '</td><td>' + esc(s.strategy) + '</td><td>' + acted + '</td></tr>';
  }).join('');
}

function renderNegotiations(records) {
  if (!records || records.length === 0) {
    document.getElementById('nego-feed').innerHTML = '<div class="empty-msg">No negotiations yet &mdash; watching the market...</div>';
    return;
  }
  document.getElementById('nego-feed').innerHTML = records.slice(0, 10).map(r => {
    const verdictClass = r.vetoed_by ? 'nego-vetoed' : r.final_approved ? 'nego-approved' : 'nego-rejected';
    const verdictText = r.vetoed_by ? 'VETOED' : r.final_approved ? 'APPROVED' : 'REJECTED';

    const opinions = (r.opinions || []).map(op => {
      const vote = op.vetoed ? 'VETO' : op.approved ? 'APPROVE' : 'REJECT';
      const vc = voteClass(op);
      const ac = agentColor(op.agent);
      return '<div class="nego-opinion">' +
        '<span class="agent-name ' + ac + '">' + agentIcon(op.agent) + ' ' + esc(op.agent) + '</span> ' +
        '<span class="' + vc + '">' + vote + '</span> (risk: ' + op.risk_score + '/10, size: ' + fmt(op.size_modifier, 1) + 'x) &mdash; ' + esc(op.reasoning || '') +
      '</div>';
    }).join('');

    const roundsData = (r.rounds || []).filter(rd => rd.round > 0);
    let rounds = '';
    if (roundsData.length > 0) {
      // Group rounds by round number
      const byRound = {};
      roundsData.forEach(rd => {
        if (!byRound[rd.round]) byRound[rd.round] = [];
        byRound[rd.round].push(rd);
      });
      rounds = Object.entries(byRound).map(([rn, rds]) => {
        return '<div class="nego-section-label">Round ' + rn + '</div>' +
          rds.map(rd => {
            const ac = agentColor(rd.speaker);
            return '<div class="nego-opinion" style="padding-left:10px;border-left:2px solid var(--border);">' +
              '<span class="agent-name ' + ac + '">' + agentIcon(rd.speaker) + ' ' + esc(rd.speaker) + '</span> ' +
              esc(rd.action).toUpperCase() + ' &mdash; ' + esc(rd.content) +
            '</div>';
          }).join('');
      }).join('');
    }

    const pairName = r.signal_pair ? r.signal_pair.split('/')[0] : '?';
    const dir = (r.signal_direction || 'long');

    return '<div class="nego-entry">' +
      '<div class="nego-header">' +
        '<span class="nego-pair">' + pairName + ' <span class="badge badge-' + dir + '">' + dir.toUpperCase() + '</span></span>' +
        '<span class="nego-meta">conf: ' + fmt(r.signal_confidence) + ' | ' + (r.consensus_type || '') + ' | ' + (r.total_rounds || 0) + ' rnds</span>' +
        '<span class="nego-verdict ' + verdictClass + '">' + verdictText + '</span>' +
      '</div>' +
      '<div class="nego-body">' +
        '<div class="nego-section-label">Initial Opinions</div>' +
        opinions +
        rounds +
      '</div>' +
      '<div class="nego-timestamp">' + (r.timestamp || '') + '</div>' +
    '</div>';
  }).join('');
}

function renderModelAccuracy(data) {
  if (!data || !data.models || Object.keys(data.models).length === 0) {
    document.getElementById('model-accuracy').innerHTML = '<span class="empty-msg">No accuracy data yet</span>';
    return;
  }
  const models = data.models;
  const weights = data.weights || {};
  const best = data.best_model;

  document.getElementById('model-accuracy').innerHTML = Object.entries(models).map(([name, s]) => {
    const pct = (s.accuracy * 100);
    const isBest = name === best;
    const w = weights[name] || 1.0;
    return '<div class="accuracy-bar">' +
      '<span class="accuracy-name">' + (isBest ? '> ' : '  ') + name + '</span>' +
      '<div style="flex:1;border:1px solid var(--border);height:16px;position:relative;background:var(--bg);">' +
        '<div class="accuracy-fill" style="width:' + Math.max(3, pct) + '%;">' + pct.toFixed(0) + '%</div>' +
      '</div>' +
      '<span style="font-size:11px;width:45px;text-align:right;">' + w.toFixed(2) + 'x</span>' +
    '</div>' +
    '<div style="font-size:10px;margin-left:86px;margin-bottom:2px;">' +
      s.total_votes + ' votes | solo: $' + (s.solo_pnl_estimate > 0 ? '+' : '') + s.solo_pnl_estimate + ' | contrarian: ' + (s.contrarian_accuracy * 100).toFixed(0) + '%' +
    '</div>';
  }).join('');
}

function renderLeaderboard(strategies) {
  if (!strategies || strategies.length === 0) {
    document.getElementById('strat-leaderboard').innerHTML = '<span class="empty-msg">No evolved strategies</span>';
    return;
  }
  document.getElementById('strat-leaderboard').innerHTML = '<table>' +
    '<thead><tr><th>Name</th><th>Gen</th><th>Status</th><th>W/L</th><th>P&L</th></tr></thead>' +
    '<tbody>' + strategies.map(s => {
      const sc = s.status === 'active' ? 'strat-active' : s.status === 'candidate' ? 'strat-candidate' : 'strat-retired';
      const pnl = s.total_trades > 0 ? '<span class="' + pnlClass(s.total_pnl) + '">$' + fmt(s.total_pnl) + '</span>' : '-';
      return '<tr><td>' + s.name + '</td><td>' + s.generation + '</td><td class="' + sc + '">' + s.status + '</td><td>' + s.wins + '/' + s.losses + '</td><td>' + pnl + '</td></tr>';
    }).join('') + '</tbody></table>';
}

async function fetchEquity() {
  try {
    const equity = await (await fetch('/api/equity?days=7')).json();
    if (equity && equity.length > 0) {
      const container = document.getElementById('equity-chart');
      const values = equity.map(e => e.value);
      const min = Math.min(...values) * 0.995;
      const max = Math.max(...values) * 1.005;
      const range = max - min || 1;
      const w = container.clientWidth;
      const barW = Math.max(2, (w / values.length) - 1);
      container.innerHTML = values.map((v, i) => {
        const h = ((v - min) / range) * 110;
        const barClass = i > 0 && v >= values[i-1] ? 'chart-bar-up' : 'chart-bar-down';
        return '<div class="chart-bar ' + barClass + '" style="left:' + (i*(barW+1)) + 'px;height:' + h + 'px;width:' + barW + 'px;"></div>';
      }).join('');
    }
  } catch(e) {}
}

// ── Splash Animation: Image → Darken → Text Reveal ──
(function() {
  const canvas = document.getElementById('splash-canvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = window.innerWidth;
  const H = window.innerHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const bg = isDark ? '#1a1a1e' : '#ffffff';
  const fg = isDark ? '#c8c8c8' : '#000000';
  const PIXEL = 4;
  const cols = Math.ceil(W / PIXEL);
  const rows = Math.ceil(H / PIXEL);

  // Phase timing (frames at ~60fps)
  const P1_IMAGE_IN = 30;      // image pixel-reveals in
  const P2_HOLD = 40;          // hold image
  const P3_DARKEN = 25;        // image darkens out
  const P4_TEXT_REVEAL = 45;   // text pixel-reveals in
  const P5_HOLD_TEXT = 25;     // hold final text
  const TOTAL = P1_IMAGE_IN + P2_HOLD + P3_DARKEN + P4_TEXT_REVEAL + P5_HOLD_TEXT;
  let frame = 0;

  // Load image + fonts in parallel
  const img = new Image();
  img.src = '/static/splash.png';
  const fontItalic = new FontFace('Redaction35', "url('/static/fonts/Redaction35Italic-rgDrB.otf')", { style: 'italic' });
  const fontNormal = new FontFace('Redaction35', "url('/static/fonts/Redaction35-BWXjw.otf')", { style: 'normal' });

  Promise.all([
    new Promise(res => { img.onload = res; img.onerror = res; }),
    fontItalic.load().then(f => document.fonts.add(f)),
    fontNormal.load().then(f => document.fonts.add(f)),
  ]).then(() => {
    // Draw image to offscreen canvas (cover, centered)
    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = W; imgCanvas.height = H;
    const ictx = imgCanvas.getContext('2d');
    const scale = Math.max(W / img.width, H / img.height);
    const iw = img.width * scale;
    const ih = img.height * scale;
    ictx.drawImage(img, (W - iw) / 2, (H - ih) / 2, iw, ih);
    const imgData = ictx.getImageData(0, 0, W, H);

    // Sample image pixels
    const imgPixels = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const sx = Math.min(c * PIXEL + PIXEL / 2, W - 1);
        const sy = Math.min(r * PIXEL + PIXEL / 2, H - 1);
        const idx = (Math.floor(sy) * W + Math.floor(sx)) * 4;
        const lum = imgData.data[idx];
        imgPixels.push({ c, r, dark: lum < 128 });
      }
    }
    // Shuffle for reveal order
    const imgOrder = [...Array(imgPixels.length).keys()];
    for (let i = imgOrder.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [imgOrder[i], imgOrder[j]] = [imgOrder[j], imgOrder[i]];
    }

    // Draw text to offscreen canvas
    const txtCanvas = document.createElement('canvas');
    txtCanvas.width = W; txtCanvas.height = H;
    const tctx = txtCanvas.getContext('2d');
    tctx.fillStyle = bg;
    tctx.fillRect(0, 0, W, H);
    tctx.fillStyle = fg;
    tctx.textAlign = 'center';
    tctx.textBaseline = 'middle';
    tctx.font = 'italic 52px Redaction35';
    tctx.fillText('Finance Bro', W / 2, H / 2 - 20);
    tctx.font = '16px Redaction35';
    tctx.fillText('Your personal hedge fund', W / 2, H / 2 + 30);
    const txtData = tctx.getImageData(0, 0, W, H);

    // Sample text pixels
    const textPixels = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const sx = Math.min(c * PIXEL + PIXEL / 2, W - 1);
        const sy = Math.min(r * PIXEL + PIXEL / 2, H - 1);
        const idx = (Math.floor(sy) * W + Math.floor(sx)) * 4;
        const lum = txtData.data[idx];
        const isText = isDark ? lum > 100 : lum < 200;
        if (isText) textPixels.push({ c, r });
      }
    }
    // Shuffle text reveal order
    for (let i = textPixels.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [textPixels[i], textPixels[j]] = [textPixels[j], textPixels[i]];
    }

    const revealed = new Set();
    const textRevealed = new Set();

    function render() {
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      if (frame < P1_IMAGE_IN) {
        // Phase 1: Image pixels appear from noise
        const progress = frame / P1_IMAGE_IN;
        const count = Math.floor(progress * progress * imgOrder.length);
        for (let i = 0; i < count; i++) revealed.add(imgOrder[i]);
        // Draw noise for unrevealed
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const hash = (c * 7919 + r * 104729 + frame * 17) % 1000;
            if (hash < (1 - progress) * 80) {
              ctx.fillStyle = fg;
              ctx.fillRect(c * PIXEL, r * PIXEL, PIXEL, PIXEL);
            }
          }
        }
        // Draw revealed image pixels
        revealed.forEach(i => {
          const p = imgPixels[i];
          ctx.fillStyle = p.dark ? fg : bg;
          ctx.fillRect(p.c * PIXEL, p.r * PIXEL, PIXEL, PIXEL);
        });

      } else if (frame < P1_IMAGE_IN + P2_HOLD) {
        // Phase 2: Hold full image
        imgPixels.forEach(p => {
          ctx.fillStyle = p.dark ? fg : bg;
          ctx.fillRect(p.c * PIXEL, p.r * PIXEL, PIXEL, PIXEL);
        });

      } else if (frame < P1_IMAGE_IN + P2_HOLD + P3_DARKEN) {
        // Phase 3: Image darkens / dissolves out
        const progress = (frame - P1_IMAGE_IN - P2_HOLD) / P3_DARKEN;
        const ease = progress * progress;
        imgPixels.forEach(p => {
          // Randomly dissolve pixels away
          const hash = (p.c * 3571 + p.r * 7919) % 1000;
          if (hash < ease * 1000) {
            // dissolved — show bg
          } else {
            const alpha = 1 - ease * 0.8;
            if (p.dark) {
              ctx.fillStyle = isDark
                ? 'rgba(200,200,200,' + alpha + ')'
                : 'rgba(0,0,0,' + alpha + ')';
              ctx.fillRect(p.c * PIXEL, p.r * PIXEL, PIXEL, PIXEL);
            }
          }
        });

      } else if (frame < P1_IMAGE_IN + P2_HOLD + P3_DARKEN + P4_TEXT_REVEAL) {
        // Phase 4: Text pixel-reveals from noise
        const progress = (frame - P1_IMAGE_IN - P2_HOLD - P3_DARKEN) / P4_TEXT_REVEAL;
        const count = Math.floor(progress * textPixels.length);
        for (let i = textRevealed.size; i < count; i++) textRevealed.add(i);
        // Scattered noise
        const noiseDensity = 0.12 * (1 - progress * progress);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const hash = (c * 7919 + r * 104729 + frame * 13) % 1000;
            if (hash < noiseDensity * 1000) {
              ctx.fillStyle = fg;
              ctx.fillRect(c * PIXEL, r * PIXEL, PIXEL, PIXEL);
            }
          }
        }
        // Draw text pixels
        ctx.fillStyle = fg;
        textRevealed.forEach(i => {
          const p = textPixels[i];
          ctx.fillRect(p.c * PIXEL, p.r * PIXEL, PIXEL, PIXEL);
        });

      } else {
        // Phase 5: Hold clean text
        ctx.fillStyle = fg;
        textPixels.forEach(p => {
          ctx.fillRect(p.c * PIXEL, p.r * PIXEL, PIXEL, PIXEL);
        });
      }

      frame++;
      if (frame < TOTAL) {
        requestAnimationFrame(render);
      } else {
        const splash = document.getElementById('splash');
        splash.classList.add('fade-out');
        document.getElementById('main-ui').classList.replace('dashboard-hidden', 'dashboard-visible');
        document.getElementById('main-desktop').classList.replace('dashboard-hidden', 'dashboard-visible');
        document.getElementById('main-statusbar').classList.replace('dashboard-hidden', 'dashboard-visible');
        setTimeout(() => { splash.style.display = 'none'; }, 600);
      }
    }

    requestAnimationFrame(render);
  });
})();

connectWS();
fetchEquity();
setInterval(fetchEquity, 30000);

// Fallback: if WS fails, poll REST
setInterval(async () => {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  try {
    const [portfolio, trades, signals, status, negotiations, accuracy, strategies] = await Promise.all([
      fetch('/api/portfolio').then(r => r.json()),
      fetch('/api/trades?n=10').then(r => r.json()),
      fetch('/api/signals').then(r => r.json()),
      fetch('/api/status').then(r => r.json()),
      fetch('/api/negotiations').then(r => r.json()),
      fetch('/api/model-accuracy').then(r => r.json()),
      fetch('/api/evolved-strategies').then(r => r.json()),
    ]);
    renderAll({portfolio, trades, signals, status, negotiations, model_accuracy: accuracy, evolved_strategies: strategies});
  } catch(e) {}
}, 10000);
</script>
</body>
</html>"""


def main():
    global db, config, negotiation_engine

    parser = argparse.ArgumentParser(description="Finance Agent Live Dashboard")
    parser.add_argument("--port", type=int, default=8421, help="Port (default: 8421)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    config = load_config()
    db = Database()

    # Initialize negotiation engine for demo triggers
    providers = build_providers(config)
    if providers:
        negotiation_engine = NegotiationEngine(providers, db)
        print(f"Negotiation engine ready: {len(providers)} provider(s)")
        print(f"Trigger: curl -X POST 'http://{args.host}:{args.port}/api/trigger?pair=BTC&direction=long'")
    else:
        print("WARNING: No AI providers — demo trigger disabled")

    print(f"\nLive Dashboard: http://{args.host}:{args.port}")
    print(f"WebSocket: ws://{args.host}:{args.port}/ws")
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
