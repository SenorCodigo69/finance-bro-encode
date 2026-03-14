"""Lightweight web dashboard — serves portfolio state, trades, equity curve via HTTP."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.database import Database

PORT = 8420
db: Database | None = None
config = None


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/portfolio":
            self._json_response(self._get_portfolio())
        elif path == "/api/trades":
            n = int(params.get("n", ["20"])[0])
            self._json_response(self._get_trades(n))
        elif path == "/api/equity":
            days = int(params.get("days", ["7"])[0])
            self._json_response(self._get_equity(days))
        elif path == "/api/signals":
            self._json_response(self._get_signals())
        elif path == "/api/performance":
            self._json_response(self._get_performance())
        elif path == "/api/status":
            self._json_response(self._get_status())
        else:
            self.send_error(404)

    def _json_response(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        body = DASHBOARD_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _get_portfolio(self):
        curve = db.get_equity_curve(days=1)
        if not curve:
            return {"total_value": config.risk.starting_capital, "cash": config.risk.starting_capital,
                    "positions_value": 0, "drawdown_pct": 0, "daily_pnl": 0, "total_pnl": 0, "total_pnl_pct": 0}
        snap = curve[-1]
        return {
            "total_value": snap.total_value, "cash": snap.cash,
            "positions_value": snap.positions_value, "open_positions": snap.open_positions,
            "drawdown_pct": snap.drawdown_pct, "high_water_mark": snap.high_water_mark,
            "daily_pnl": snap.daily_pnl, "total_pnl": snap.total_pnl,
            "total_pnl_pct": snap.total_pnl_pct, "timestamp": snap.timestamp,
        }

    def _get_trades(self, n=20):
        trades = db.get_recent_trades(n)
        return [
            {"id": t.id, "pair": t.pair, "direction": t.direction,
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

    def _get_equity(self, days=7):
        curve = db.get_equity_curve(days)
        return [{"timestamp": s.timestamp, "value": s.total_value, "drawdown": s.drawdown_pct}
                for s in curve]

    def _get_signals(self):
        rows = db.conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT 30"
        ).fetchall()
        return [
            {"pair": r["pair"], "direction": r["direction"], "confidence": r["confidence"],
             "strategy": r["strategy_name"], "acted_on": bool(r["acted_on"]),
             "reason": r["block_reason"] or "", "timestamp": r["timestamp"]}
            for r in rows
        ]

    def _get_performance(self):
        from src.journal import Journal
        j = Journal(db)
        return j.get_performance_summary(days=30)

    def _get_status(self):
        status = db.get_state("agent_status") or "RUNNING"
        trades_since_review = db.get_state("trades_since_review") or "0"
        return {
            "status": status,
            "mode": config.agent.mode,
            "trades_since_review": int(trades_since_review),
            "pairs": config.agent.pairs,
            "cycle_sec": config.agent.cycle_interval_sec,
        }

    def log_message(self, format, *args):
        pass  # Suppress access logs


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finance Agent Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', monospace; padding: 20px; }
  h1 { color: #00ffa3; font-size: 1.5rem; margin-bottom: 20px; }
  h2 { color: #00d4ff; font-size: 1rem; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 2px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
  .card { background: #12121a; border: 1px solid #1e1e2e; border-radius: 8px; padding: 16px; }
  .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1a2a; }
  .stat:last-child { border-bottom: none; }
  .stat-label { color: #888; }
  .stat-value { font-weight: bold; }
  .positive { color: #00ffa3; }
  .negative { color: #ff4444; }
  .neutral { color: #ffaa00; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 8px; color: #00d4ff; border-bottom: 2px solid #1e1e2e; }
  td { padding: 8px; border-bottom: 1px solid #1a1a2a; }
  tr:hover { background: #1a1a2a; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
  .badge-long { background: #00ffa322; color: #00ffa3; }
  .badge-short { background: #ff444422; color: #ff4444; }
  .badge-open { background: #ffaa0022; color: #ffaa00; }
  .badge-closed { background: #00ffa322; color: #00ffa3; }
  .badge-stopped { background: #ff444422; color: #ff4444; }
  .status-bar { display: flex; gap: 20px; margin-bottom: 20px; padding: 10px 16px; background: #12121a; border-radius: 8px; border: 1px solid #1e1e2e; font-size: 0.85rem; }
  .status-item { display: flex; gap: 8px; align-items: center; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-green { background: #00ffa3; box-shadow: 0 0 6px #00ffa3; }
  .dot-yellow { background: #ffaa00; box-shadow: 0 0 6px #ffaa00; }
  .dot-red { background: #ff4444; box-shadow: 0 0 6px #ff4444; }
  .chart-container { height: 200px; position: relative; padding: 10px 0; }
  .chart-bar { position: absolute; bottom: 0; background: #00d4ff33; border-top: 2px solid #00d4ff; min-width: 3px; }
  .refresh-note { color: #555; font-size: 0.75rem; text-align: center; margin-top: 20px; }
  .signals-table td { font-size: 0.8rem; }
</style>
</head>
<body>

<h1>FINANCE AGENT</h1>

<div class="status-bar" id="status-bar">
  <div class="status-item"><div class="dot dot-green" id="status-dot"></div><span id="status-text">Loading...</span></div>
  <div class="status-item">Mode: <strong id="mode-text">-</strong></div>
  <div class="status-item">Pairs: <span id="pairs-text">-</span></div>
  <div class="status-item">Cycle: <span id="cycle-text">-</span></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Portfolio</h2>
    <div id="portfolio-stats">Loading...</div>
  </div>
  <div class="card">
    <h2>Performance (30d)</h2>
    <div id="perf-stats">Loading...</div>
  </div>
  <div class="card">
    <h2>Equity Curve</h2>
    <div class="chart-container" id="equity-chart"></div>
  </div>
</div>

<div class="card" style="margin-bottom: 20px;">
  <h2>Recent Trades</h2>
  <table>
    <thead><tr><th>Time</th><th>Pair</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Status</th><th>Strategy</th></tr></thead>
    <tbody id="trades-body"><tr><td colspan="8">Loading...</td></tr></tbody>
  </table>
</div>

<div class="card">
  <h2>Recent Signals</h2>
  <table class="signals-table">
    <thead><tr><th>Time</th><th>Pair</th><th>Dir</th><th>Conf</th><th>Strategy</th><th>Acted</th><th>Reason</th></tr></thead>
    <tbody id="signals-body"><tr><td colspan="7">Loading...</td></tr></tbody>
  </table>
</div>

<p class="refresh-note">Auto-refreshes every 30s</p>

<script>
const API = '';

function pnlClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : 'neutral'; }
function fmt(v, d=2) { return v != null ? Number(v).toFixed(d) : '-'; }
function fmtPct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '-'; }
function fmtTime(t) { if (!t) return '-'; const d = new Date(t); return d.toLocaleDateString('en-GB', {day:'2-digit',month:'short'}) + ' ' + d.toLocaleTimeString('en-GB', {hour:'2-digit',minute:'2-digit'}); }

async function fetchJSON(url) {
  try { const r = await fetch(API + url); return await r.json(); } catch(e) { return null; }
}

function renderStat(label, value, cls='') {
  return `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value ${cls}">${value}</span></div>`;
}

async function refresh() {
  const [portfolio, perf, trades, signals, status, equity] = await Promise.all([
    fetchJSON('/api/portfolio'),
    fetchJSON('/api/performance'),
    fetchJSON('/api/trades?n=15'),
    fetchJSON('/api/signals'),
    fetchJSON('/api/status'),
    fetchJSON('/api/equity?days=7'),
  ]);

  // Status bar
  if (status) {
    const dot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    dot.className = 'dot ' + (status.status === 'RUNNING' ? 'dot-green' : status.status === 'PAUSED' ? 'dot-red' : 'dot-yellow');
    statusText.textContent = status.status;
    document.getElementById('mode-text').textContent = status.mode.toUpperCase();
    document.getElementById('pairs-text').textContent = status.pairs.join(', ');
    document.getElementById('cycle-text').textContent = status.cycle_sec + 's';
  }

  // Portfolio
  if (portfolio) {
    document.getElementById('portfolio-stats').innerHTML = [
      renderStat('Total Value', '$' + fmt(portfolio.total_value), pnlClass(portfolio.total_pnl)),
      renderStat('Cash', '$' + fmt(portfolio.cash)),
      renderStat('Positions', '$' + fmt(portfolio.positions_value)),
      renderStat('Drawdown', fmtPct(portfolio.drawdown_pct), portfolio.drawdown_pct > 0.2 ? 'negative' : ''),
      renderStat('Daily P&L', '$' + fmt(portfolio.daily_pnl), pnlClass(portfolio.daily_pnl)),
      renderStat('Total P&L', '$' + fmt(portfolio.total_pnl) + ' (' + fmtPct(portfolio.total_pnl_pct) + ')', pnlClass(portfolio.total_pnl)),
    ].join('');
  }

  // Performance
  if (perf && perf.total_trades > 0) {
    document.getElementById('perf-stats').innerHTML = [
      renderStat('Total Trades', perf.total_trades),
      renderStat('Win Rate', fmtPct(perf.win_rate), perf.win_rate > 0.5 ? 'positive' : 'negative'),
      renderStat('Wins / Losses', perf.wins + ' / ' + perf.losses),
      renderStat('Total P&L', '$' + fmt(perf.total_pnl), pnlClass(perf.total_pnl)),
      renderStat('Avg Win', '$' + fmt(perf.avg_win), 'positive'),
      renderStat('Avg Loss', '$' + fmt(perf.avg_loss), 'negative'),
      renderStat('Profit Factor', fmt(perf.profit_factor)),
    ].join('');
  } else {
    document.getElementById('perf-stats').innerHTML = '<div class="stat"><span class="stat-label">No closed trades yet</span></div>';
  }

  // Equity chart
  if (equity && equity.length > 0) {
    const container = document.getElementById('equity-chart');
    const values = equity.map(e => e.value);
    const min = Math.min(...values) * 0.995;
    const max = Math.max(...values) * 1.005;
    const range = max - min || 1;
    const w = container.clientWidth;
    const barW = Math.max(3, (w / values.length) - 1);
    container.innerHTML = values.map((v, i) => {
      const h = ((v - min) / range) * 180;
      const color = v >= 200 ? '#00ffa3' : '#ff4444';
      return `<div class="chart-bar" style="left:${i * (barW + 1)}px;height:${h}px;width:${barW}px;background:${color}33;border-color:${color}"></div>`;
    }).join('');
  }

  // Trades table
  if (trades && trades.length > 0) {
    document.getElementById('trades-body').innerHTML = trades.map(t => {
      const dirBadge = `<span class="badge badge-${t.direction}">${t.direction.toUpperCase()}</span>`;
      const statusBadge = `<span class="badge badge-${t.status === 'open' ? 'open' : t.pnl >= 0 ? 'closed' : 'stopped'}">${t.status}</span>`;
      const pnl = t.pnl != null ? `<span class="${pnlClass(t.pnl)}">$${fmt(t.pnl)}</span>` : '-';
      return `<tr><td>${fmtTime(t.entry_time)}</td><td>${t.pair}</td><td>${dirBadge}</td><td>${fmt(t.entry_price)}</td><td>${t.exit_price ? fmt(t.exit_price) : '-'}</td><td>${pnl}</td><td>${statusBadge}</td><td>${t.strategy}</td></tr>`;
    }).join('');
  } else {
    document.getElementById('trades-body').innerHTML = '<tr><td colspan="8" style="color:#555">No trades yet — agent is watching the market</td></tr>';
  }

  // Signals table
  if (signals && signals.length > 0) {
    document.getElementById('signals-body').innerHTML = signals.map(s => {
      const dirBadge = `<span class="badge badge-${s.direction}">${s.direction.toUpperCase()}</span>`;
      const acted = s.acted_on ? '<span class="positive">Yes</span>' : '<span class="negative">No</span>';
      return `<tr><td>${fmtTime(s.timestamp)}</td><td>${s.pair}</td><td>${dirBadge}</td><td>${fmt(s.confidence)}</td><td>${s.strategy}</td><td>${acted}</td><td>${s.reason || '-'}</td></tr>`;
    }).join('');
  }
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


def main():
    global db, config
    config = load_config()
    db = Database()

    print(f"Dashboard running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop")

    server = HTTPServer(("127.0.0.1", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        db.close()
        server.server_close()


if __name__ == "__main__":
    main()
