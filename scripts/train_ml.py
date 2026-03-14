"""Train ML models on historical OHLCV data fetched from Hyperliquid.

Usage:
    python scripts/train_ml.py                    # Train all models, 90 days data
    python scripts/train_ml.py --days 180         # Use 180 days of history
    python scripts/train_ml.py --regime-only      # Only train regime classifier
    python scripts/train_ml.py --export-data      # Fetch data + save CSVs (no training)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ccxt
except ImportError:
    raise ImportError("Training script requires ccxt: pip install ccxt")
import pandas as pd
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Pairs to train on (crypto + synthetics — more history = better)
TRAINING_PAIRS = [
    "BTC/USDC:USDC",
    "ETH/USDC:USDC",
    "AAVE/USDC:USDC",
    "XYZ-AAPL/USDC:USDC",
    "XYZ-GOLD/USDC:USDC",
    "XYZ-SILVER/USDC:USDC",
    "XYZ-TSLA/USDC:USDC",
    "XYZ-NVDA/USDC:USDC",
]

DATA_DIR = Path("data/training")


def fetch_training_data(pair: str, timeframe: str, days: int) -> pd.DataFrame | None:
    """Fetch OHLCV data from Hyperliquid (Bybit fallback)."""
    exchanges = [
        ("Hyperliquid", ccxt.hyperliquid({"enableRateLimit": True})),
        ("Bybit", ccxt.bybit({"enableRateLimit": True})),
    ]

    base = pair.split("/")[0]
    variants = [pair]
    if ":" in pair:
        variants.append(pair.split(":")[0])
    variants.append(f"{base}/USDC")
    variants.append(f"{base}/USDT")
    # Deduplicate
    seen = set()
    variants = [p for p in variants if p not in seen and not seen.add(p)]

    for exch_name, exchange in exchanges:
        for try_pair in variants:
            try:
                console.print(f"  [cyan]Fetching {try_pair} ({timeframe}, {days}d) from {exch_name}...[/cyan]")
                since = exchange.parse8601(
                    (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat()
                )

                all_candles = []
                limit = 1000
                max_candles = 50_000  # [S5-H3] prevent unbounded memory growth
                while True:
                    candles = exchange.fetch_ohlcv(try_pair, timeframe, since=since, limit=limit)
                    if not candles:
                        break
                    all_candles.extend(candles)
                    since = candles[-1][0] + 1
                    if len(candles) < limit or len(all_candles) >= max_candles:
                        break

                if len(all_candles) < 200:
                    continue

                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                console.print(f"  [green]Got {len(df)} candles[/green]")
                return df

            except Exception as e:
                continue

    console.print(f"  [yellow]No data for {pair}[/yellow]")
    return None


def main():
    parser = argparse.ArgumentParser(description="Train ML models for the trading agent")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch (default: 90)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--regime-only", action="store_true", help="Only train regime classifier")
    parser.add_argument("--export-data", action="store_true", help="Only fetch + save data (no training)")
    args = parser.parse_args()

    console.print(Panel("[bold]ML Model Training[/bold]", style="cyan"))

    # Step 1: Fetch training data
    console.print("\n[bold]Step 1: Fetching training data...[/bold]")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for pair in TRAINING_PAIRS:
        safe_name = pair.replace("/", "_").replace(":", "_")
        csv_path = DATA_DIR / f"{safe_name}.csv"

        # Check if we already have recent data
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            if len(existing) > 200:
                console.print(f"  [dim]Using cached {safe_name} ({len(existing)} candles)[/dim]")
                dfs[pair] = existing
                continue

        df = fetch_training_data(pair, args.timeframe, args.days)
        if df is not None:
            dfs[pair] = df
            df.to_csv(csv_path, index=False)
            console.print(f"  [green]Saved {safe_name}.csv ({len(df)} candles)[/green]")

    console.print(f"\n[bold]Training data ready: {len(dfs)} pairs[/bold]")

    if args.export_data:
        console.print("[green]Data export complete (--export-data mode)[/green]")
        return

    # Step 2: Train models
    from src.ml_signals import MLRegimeClassifier, HAS_TORCH

    # 2a: Regime classifier
    console.print("\n[bold]Step 2a: Training ML Regime Classifier...[/bold]")
    regime_clf = MLRegimeClassifier()
    report = regime_clf.train(dfs)

    if "error" not in report:
        table = Table(title="Regime Classifier Results", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Samples", str(report.get("samples", 0)))
        table.add_row("Features", str(report.get("features", 0)))
        table.add_row("CV Accuracy", f"{report.get('cv_accuracy_mean', 0):.1%} ± {report.get('cv_accuracy_std', 0):.1%}")
        dist = report.get("class_distribution", {})
        table.add_row("Bulls", str(dist.get("bull", 0)))
        table.add_row("Bears", str(dist.get("bear", 0)))
        table.add_row("Sideways", str(dist.get("sideways", 0)))
        console.print(table)

        console.print("\n[bold]Top Features:[/bold]")
        for feat, imp in report.get("top_features", {}).items():
            console.print(f"  {feat}: {imp:.4f}")
    else:
        console.print(f"[red]Regime classifier failed: {report['error']}[/red]")

    if args.regime_only:
        console.print("\n[green]Done (--regime-only mode)[/green]")
        return

    # 2b: LSTM Price Predictor
    if HAS_TORCH:
        console.print("\n[bold]Step 2b: Training LSTM Price Predictor...[/bold]")
        from src.ml_signals import PricePredictor
        predictor = PricePredictor()
        lstm_report = predictor.train(dfs, epochs=50)

        if "error" not in lstm_report:
            console.print(f"  Samples: {lstm_report.get('samples', 0)}")
            console.print(f"  Best val accuracy: {lstm_report.get('best_val_accuracy', 0):.1%}")
            console.print(f"  Class dist: {lstm_report.get('class_distribution', {})}")
        else:
            console.print(f"  [yellow]{lstm_report['error']}[/yellow]")
    else:
        console.print("\n[yellow]Step 2b: Skipping LSTM (PyTorch not installed)[/yellow]")
        console.print("  Install with: pip install torch")

    console.print("\n[bold green]ML training complete![/bold green]")
    console.print("Models saved to data/models/")
    console.print("The strategy engine will auto-load trained models on next cycle.")


if __name__ == "__main__":
    main()
