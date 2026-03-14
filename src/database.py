"""SQLite database layer — trade journal, portfolio snapshots, agent state."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from src.models import PortfolioSnapshot, Trade
from src.utils import log

_SAFE_COL_RE = re.compile(r'^[a-z_]+$')


class Database:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path(__file__).resolve().parent.parent / "data" / "agent.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()
        self._pending_commits = 0

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL,
                status TEXT NOT NULL DEFAULT 'open',
                pnl REAL,
                pnl_pct REAL,
                fees REAL DEFAULT 0,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                signal_data TEXT,
                ai_reasoning TEXT,
                review_notes TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                strategy_name TEXT NOT NULL,
                indicators TEXT,
                reasoning TEXT,
                acted_on INTEGER NOT NULL DEFAULT 0,
                block_reason TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                open_positions INTEGER NOT NULL,
                drawdown_pct REAL NOT NULL,
                high_water_mark REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_pct REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                review_type TEXT NOT NULL,
                summary TEXT,
                patterns TEXT,
                suggestions TEXT,
                risk_assessment TEXT,
                raw_response TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_vote INTEGER NOT NULL,
                actual_profitable INTEGER NOT NULL,
                was_correct INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS market_data_cache (
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (pair, timeframe, timestamp)
            );

            -- [OPT-5] Indexes for frequent queries
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_trades_pair_status ON trades(pair, status);
            CREATE INDEX IF NOT EXISTS idx_signals_pair_tf_ts ON signals(pair, timeframe, timestamp);
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_model_accuracy_model_ts ON model_accuracy(model_name, timestamp);
        """)
        self.conn.commit()
        self._migrate_profit_split()
        self._migrate_phase5()

    @staticmethod
    def _validate_column_names(columns: dict[str, str]) -> None:
        """Ensure migration column names are safe identifiers."""
        for col in columns:
            if not _SAFE_COL_RE.match(col):
                raise ValueError(f"Invalid column name in migration: {col!r}")

    def _migrate_profit_split(self):
        """Add profit-split columns to existing tables (safe to run repeatedly)."""
        # Trades table — new columns for dual TP/trailing stops and profit split
        trade_columns = {
            "bot_take_profit": "REAL",
            "user_take_profit": "REAL",
            "bot_trailing_stop": "REAL",
            "user_trailing_stop": "REAL",
            "user_portion_closed": "INTEGER DEFAULT 0",
            "bot_portion_closed": "INTEGER DEFAULT 0",
            "profit_split_bot": "REAL",
            "profit_split_user": "REAL",
        }
        self._validate_column_names(trade_columns)
        for col, col_type in trade_columns.items():
            try:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Portfolio snapshots table — dual pool balances
        snap_columns = {
            "bot_balance": "REAL DEFAULT 0",
            "user_balance": "REAL DEFAULT 0",
        }
        self._validate_column_names(snap_columns)
        for col, col_type in snap_columns.items():
            try:
                self.conn.execute(f"ALTER TABLE portfolio_snapshots ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        self.conn.commit()

    def _migrate_phase5(self):
        """Add Phase 5 execution tracking fields (safe to run repeatedly)."""
        trade_columns = {
            "execution_venue": "TEXT DEFAULT 'hyperliquid'",
            "slippage_actual_bps": "REAL",
            "trigger_orders_placed": "INTEGER DEFAULT 0",
        }
        self._validate_column_names(trade_columns)
        for col, col_type in trade_columns.items():
            try:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass

        # Trigger orders table
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trigger_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                exchange_order_id TEXT,
                pair TEXT NOT NULL,
                side TEXT NOT NULL,
                trigger_price REAL NOT NULL,
                quantity REAL NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                placed_time TEXT,
                triggered_time TEXT,
                error TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            );
        """)
        self.conn.commit()

    # --- Trigger Orders ---

    def save_trigger_order(self, trigger: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO trigger_orders
               (trade_id, exchange_order_id, pair, side, trigger_price, quantity,
                order_type, status, placed_time, triggered_time, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trigger["trade_id"], trigger.get("exchange_order_id"),
                trigger["pair"], trigger["side"], trigger["trigger_price"],
                trigger["quantity"], trigger["order_type"], trigger["status"],
                trigger.get("placed_time"), trigger.get("triggered_time"),
                trigger.get("error"),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    _TRIGGER_ORDER_COLUMNS = frozenset({
        "status", "triggered_time", "error", "exchange_order_id",
    })

    def update_trigger_order(self, trigger_id: int, **kwargs):
        invalid = set(kwargs.keys()) - self._TRIGGER_ORDER_COLUMNS
        if invalid:
            raise ValueError(f"Invalid trigger_order columns: {invalid}")
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [trigger_id]
        self.conn.execute(f"UPDATE trigger_orders SET {sets} WHERE id=?", vals)
        self.conn.commit()

    def get_trigger_orders_for_trade(self, trade_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trigger_orders WHERE trade_id=? ORDER BY id", (trade_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_active_trigger_orders(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trigger_orders WHERE status IN ('pending', 'placed') ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Trades ---

    def save_trade(self, trade: Trade) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (pair, direction, entry_price, exit_price, quantity, stop_loss, take_profit,
                status, pnl, pnl_pct, fees, entry_time, exit_time, signal_data, ai_reasoning, review_notes,
                bot_take_profit, user_take_profit, bot_trailing_stop, user_trailing_stop,
                user_portion_closed, bot_portion_closed, profit_split_bot, profit_split_user,
                execution_venue, slippage_actual_bps, trigger_orders_placed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.pair, trade.direction, trade.entry_price, trade.exit_price,
                trade.quantity, trade.stop_loss, trade.take_profit, trade.status,
                trade.pnl, trade.pnl_pct, trade.fees, trade.entry_time, trade.exit_time,
                json.dumps(trade.signal_data), trade.ai_reasoning, trade.review_notes,
                trade.bot_take_profit, trade.user_take_profit,
                trade.bot_trailing_stop, trade.user_trailing_stop,
                1 if trade.user_portion_closed else 0,
                1 if trade.bot_portion_closed else 0,
                trade.profit_split_bot, trade.profit_split_user,
                trade.execution_venue, trade.slippage_actual_bps,
                1 if trade.trigger_orders_placed else 0,
            ),
        )
        self.conn.commit()
        trade.id = cur.lastrowid
        return trade.id

    def update_trade(self, trade: Trade):
        self.conn.execute(
            """UPDATE trades SET exit_price=?, status=?, pnl=?, pnl_pct=?, fees=?,
               exit_time=?, review_notes=?,
               bot_take_profit=?, user_take_profit=?,
               bot_trailing_stop=?, user_trailing_stop=?,
               user_portion_closed=?, bot_portion_closed=?,
               profit_split_bot=?, profit_split_user=?,
               execution_venue=?, slippage_actual_bps=?, trigger_orders_placed=?
               WHERE id=?""",
            (trade.exit_price, trade.status, trade.pnl, trade.pnl_pct,
             trade.fees, trade.exit_time, trade.review_notes,
             trade.bot_take_profit, trade.user_take_profit,
             trade.bot_trailing_stop, trade.user_trailing_stop,
             1 if trade.user_portion_closed else 0,
             1 if trade.bot_portion_closed else 0,
             trade.profit_split_bot, trade.profit_split_user,
             trade.execution_venue, trade.slippage_actual_bps,
             1 if trade.trigger_orders_placed else 0,
             trade.id),
        )
        self.conn.commit()

    def get_open_trades(self) -> list[Trade]:
        rows = self.conn.execute("SELECT * FROM trades WHERE status='open'").fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_recent_trades(self, n: int) -> list[Trade]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_trades_since(self, since: str) -> list[Trade]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE entry_time >= ? ORDER BY entry_time", (since,)
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_trade_count_today(self) -> int:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE entry_time >= ?", (today,)
        ).fetchone()
        return row["cnt"]

    def get_last_loss_time(self) -> str | None:
        row = self.conn.execute(
            "SELECT exit_time FROM trades WHERE pnl < 0 ORDER BY exit_time DESC LIMIT 1"
        ).fetchone()
        return row["exit_time"] if row else None

    def _row_to_trade(self, row: sqlite3.Row) -> Trade:
        sig = row["signal_data"]
        # Safely access profit-split columns (may not exist in old DBs before migration runs)
        keys = row.keys() if hasattr(row, "keys") else []
        return Trade(
            id=row["id"],
            pair=row["pair"],
            direction=row["direction"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            quantity=row["quantity"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            status=row["status"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            fees=row["fees"],
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            signal_data=json.loads(sig) if sig else {},
            ai_reasoning=row["ai_reasoning"] or "",
            review_notes=row["review_notes"],
            bot_take_profit=row["bot_take_profit"] if "bot_take_profit" in keys else None,
            user_take_profit=row["user_take_profit"] if "user_take_profit" in keys else None,
            bot_trailing_stop=row["bot_trailing_stop"] if "bot_trailing_stop" in keys else None,
            user_trailing_stop=row["user_trailing_stop"] if "user_trailing_stop" in keys else None,
            user_portion_closed=bool(row["user_portion_closed"]) if "user_portion_closed" in keys else False,
            bot_portion_closed=bool(row["bot_portion_closed"]) if "bot_portion_closed" in keys else False,
            profit_split_bot=row["profit_split_bot"] if "profit_split_bot" in keys else None,
            profit_split_user=row["profit_split_user"] if "profit_split_user" in keys else None,
            execution_venue=row["execution_venue"] if "execution_venue" in keys else "hyperliquid",
            slippage_actual_bps=row["slippage_actual_bps"] if "slippage_actual_bps" in keys else None,
            trigger_orders_placed=bool(row["trigger_orders_placed"]) if "trigger_orders_placed" in keys else False,
        )

    # --- Signals ---

    def save_signal(self, signal, acted_on: bool, reason: str = ""):
        self.conn.execute(
            """INSERT INTO signals
               (pair, timeframe, direction, confidence, strategy_name, indicators,
                reasoning, acted_on, block_reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.pair, signal.timeframe, signal.direction, signal.confidence,
                signal.strategy_name, json.dumps(signal.indicators), signal.reasoning,
                1 if acted_on else 0, reason, signal.timestamp,
            ),
        )
        self.conn.commit()

    # --- Portfolio snapshots ---

    def save_snapshot(self, snap: PortfolioSnapshot):
        self.conn.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, total_value, cash, positions_value, open_positions,
                drawdown_pct, high_water_mark, daily_pnl, total_pnl, total_pnl_pct,
                bot_balance, user_balance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snap.timestamp, snap.total_value, snap.cash, snap.positions_value,
                snap.open_positions, snap.drawdown_pct, snap.high_water_mark,
                snap.daily_pnl, snap.total_pnl, snap.total_pnl_pct,
                snap.bot_balance, snap.user_balance,
            ),
        )
        self.conn.commit()

    def get_equity_curve(self, days: int = 30, limit: int = 0) -> list[PortfolioSnapshot]:
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        query = "SELECT * FROM portfolio_snapshots WHERE datetime(timestamp) >= datetime(?) ORDER BY timestamp"
        params: list = [since]
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            keys = r.keys() if hasattr(r, "keys") else []
            results.append(PortfolioSnapshot(
                timestamp=r["timestamp"], total_value=r["total_value"],
                cash=r["cash"], positions_value=r["positions_value"],
                open_positions=r["open_positions"], drawdown_pct=r["drawdown_pct"],
                high_water_mark=r["high_water_mark"], daily_pnl=r["daily_pnl"],
                total_pnl=r["total_pnl"], total_pnl_pct=r["total_pnl_pct"],
                bot_balance=r["bot_balance"] if "bot_balance" in keys else 0.0,
                user_balance=r["user_balance"] if "user_balance" in keys else 0.0,
            ))
        return results

    # --- AI reviews ---

    def save_review(self, review: dict):
        self.conn.execute(
            """INSERT INTO ai_reviews (timestamp, review_type, summary, patterns, suggestions,
               risk_assessment, raw_response)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                review.get("timestamp", ""),
                review.get("review_type", "periodic"),
                review.get("summary", ""),
                json.dumps(review.get("patterns", [])),
                json.dumps(review.get("suggestions", [])),
                review.get("risk_assessment", ""),
                review.get("raw_response", ""),
            ),
        )
        self.conn.commit()

    # --- Agent state (key-value) ---

    def get_state(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM agent_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_state (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    # --- Market data cache ---

    def get_cached_ohlcv(self, pair: str, timeframe: str, since: int) -> list[tuple]:
        rows = self.conn.execute(
            """SELECT timestamp, open, high, low, close, volume
               FROM market_data_cache
               WHERE pair=? AND timeframe=? AND timestamp >= ?
               ORDER BY timestamp""",
            (pair, timeframe, since),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def cleanup_cache(self, max_age_hours: int = 72):
        """Remove old cached candles to prevent unbounded DB growth."""
        cutoff = int((time.time() - max_age_hours * 3600) * 1000)
        deleted = self.conn.execute("DELETE FROM market_data_cache WHERE timestamp < ?", (cutoff,)).rowcount
        self.conn.commit()
        if deleted:
            log.info(f"[OPT-2] Cleaned {deleted} stale cache rows (>{max_age_hours}h old)")

    def prune_vote_records(self, keep_latest: int = 200):
        """[OPT-7] Remove old vote_record_* batch keys, keeping only the most recent ones."""
        rows = self.conn.execute(
            "SELECT key FROM agent_state WHERE key LIKE 'vote_record_20%' ORDER BY key DESC"
        ).fetchall()
        if len(rows) <= keep_latest:
            return
        old_keys = [r["key"] for r in rows[keep_latest:]]
        for key in old_keys:
            self.conn.execute("DELETE FROM agent_state WHERE key=?", (key,))
        self.conn.commit()
        log.info(f"[OPT-7] Pruned {len(old_keys)} old vote records (kept latest {keep_latest})")

    def cache_ohlcv(self, pair: str, timeframe: str, candles: list[tuple]):
        self.conn.executemany(
            """INSERT OR REPLACE INTO market_data_cache
               (pair, timeframe, timestamp, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(pair, timeframe, c[0], c[1], c[2], c[3], c[4], c[5]) for c in candles],
        )
        self.conn.commit()

    # --- Evolved strategies ---

    def init_evolved_strategies_table(self):
        """Create the evolved_strategies table if it doesn't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS evolved_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                code_hash TEXT NOT NULL UNIQUE,
                generation INTEGER NOT NULL DEFAULT 1,
                parent_strategies TEXT,
                rationale TEXT,
                status TEXT NOT NULL DEFAULT 'candidate',
                created_at TEXT NOT NULL,
                activated_at TEXT,
                retired_at TEXT,
                total_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                total_pnl REAL NOT NULL DEFAULT 0.0,
                win_rate REAL NOT NULL DEFAULT 0.0
            );
        """)
        self.conn.commit()

    def save_evolved_strategy(self, meta) -> int:
        """Insert a new evolved strategy record. Returns the new row id."""
        from src.strategy_evolver import EvolvedStrategyMeta

        cur = self.conn.execute(
            """INSERT INTO evolved_strategies
               (name, file_path, code_hash, generation, parent_strategies, rationale,
                status, created_at, activated_at, retired_at,
                total_trades, wins, losses, total_pnl, win_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                meta.name, meta.file_path, meta.code_hash, meta.generation,
                json.dumps(meta.parent_strategies), meta.rationale,
                meta.status, meta.created_at, meta.activated_at, meta.retired_at,
                meta.total_trades, meta.wins, meta.losses, meta.total_pnl, meta.win_rate,
            ),
        )
        self.conn.commit()
        meta.id = cur.lastrowid
        return meta.id

    def update_evolved_strategy(self, meta):
        """Update an existing evolved strategy record."""
        self.conn.execute(
            """UPDATE evolved_strategies
               SET status=?, activated_at=?, retired_at=?,
                   total_trades=?, wins=?, losses=?, total_pnl=?, win_rate=?
               WHERE id=?""",
            (
                meta.status, meta.activated_at, meta.retired_at,
                meta.total_trades, meta.wins, meta.losses, meta.total_pnl, meta.win_rate,
                meta.id,
            ),
        )
        self.conn.commit()

    def get_evolved_strategies(self, status: str | None = None) -> list:
        """Get evolved strategies, optionally filtered by status."""
        from src.strategy_evolver import EvolvedStrategyMeta

        if status:
            rows = self.conn.execute(
                "SELECT * FROM evolved_strategies WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM evolved_strategies ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_evolved_meta(r) for r in rows]

    def get_evolved_strategy_by_name(self, name: str):
        """Get a single evolved strategy by name."""
        from src.strategy_evolver import EvolvedStrategyMeta

        row = self.conn.execute(
            "SELECT * FROM evolved_strategies WHERE name=? ORDER BY id DESC LIMIT 1",
            (name,),
        ).fetchone()
        return self._row_to_evolved_meta(row) if row else None

    def get_evolved_strategy_by_hash(self, code_hash: str):
        """Get a single evolved strategy by code hash."""
        from src.strategy_evolver import EvolvedStrategyMeta

        row = self.conn.execute(
            "SELECT * FROM evolved_strategies WHERE code_hash=?",
            (code_hash,),
        ).fetchone()
        return self._row_to_evolved_meta(row) if row else None

    def _row_to_evolved_meta(self, row: sqlite3.Row):
        from src.strategy_evolver import EvolvedStrategyMeta

        parents = row["parent_strategies"]
        return EvolvedStrategyMeta(
            id=row["id"],
            name=row["name"],
            file_path=row["file_path"],
            code_hash=row["code_hash"],
            generation=row["generation"],
            parent_strategies=json.loads(parents) if parents else [],
            rationale=row["rationale"] or "",
            status=row["status"],
            created_at=row["created_at"],
            activated_at=row["activated_at"],
            retired_at=row["retired_at"],
            total_trades=row["total_trades"],
            wins=row["wins"],
            losses=row["losses"],
            total_pnl=row["total_pnl"],
            win_rate=row["win_rate"],
        )

    # --- Model accuracy ---

    def save_model_outcome(
        self,
        signal_id: str,
        pair: str,
        direction: str,
        model_name: str,
        model_vote: bool,
        actual_profitable: bool,
        was_correct: bool,
        timestamp: str,
    ):
        """Record a single model's vote outcome after trade close."""
        self.conn.execute(
            """INSERT INTO model_accuracy
               (timestamp, signal_id, pair, direction, model_name,
                model_vote, actual_profitable, was_correct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, signal_id, pair, direction, model_name,
                1 if model_vote else 0,
                1 if actual_profitable else 0,
                1 if was_correct else 0,
            ),
        )
        self.conn.commit()

    def get_model_outcomes(self, limit: int = 1000) -> list[dict]:
        """Return recent model outcome rows as dicts."""
        rows = self.conn.execute(
            """SELECT * FROM model_accuracy
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "signal_id": r["signal_id"],
                "pair": r["pair"],
                "direction": r["direction"],
                "model_name": r["model_name"],
                "model_vote": bool(r["model_vote"]),
                "actual_profitable": bool(r["actual_profitable"]),
                "was_correct": bool(r["was_correct"]),
            }
            for r in rows
        ]

    def close(self):
        self.conn.close()
