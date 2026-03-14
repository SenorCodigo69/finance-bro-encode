"""Competitive intelligence scanner — auto-scans open-source trading agent repos.

Fetches recent commits, releases, README content, and strategy files from
GitHub to extract techniques and features we might be missing.

Usage:
    python scripts/competitive_intel.py
    python scripts/competitive_intel.py --repo freqtrade
    python scripts/competitive_intel.py --export
    python scripts/competitive_intel.py --repo freqtrade --export
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"

# Repos to scan: display_name -> "owner/repo"
REPOS: dict[str, str] = {
    "eliza":      "elizaOS/eliza",
    "freqtrade":  "freqtrade/freqtrade",
    "hummingbot": "hummingbot/hummingbot",
    "octobot":    "OctoBot-Trading/OctoBot",
    "finrl":      "AI4Finance-Foundation/FinRL",
}

# What strategies our agent already has
OUR_STRATEGIES: set[str] = {
    "momentum",
    "trend_following",
    "mean_reversion",
    "breakout",
}

# What data sources we already use
OUR_DATA_SOURCES: set[str] = {
    "ohlcv",
    "rsi",
    "macd",
    "bollinger bands",
    "atr",
    "volume",
    "funding rate",
    "open interest",
    "long short ratio",
    "taker buy sell",
}

# What features we already have
OUR_FEATURES: set[str] = {
    "paper trading",
    "stop loss",
    "take profit",
    "position sizing",
    "drawdown protection",
    "multi-pair",
    "ai signal vetting",
    "trade journal",
    "backtesting",
    "risk manager",
}

# Keywords to find strategies in commit messages / READMEs
STRATEGY_KEYWORDS: list[str] = [
    "strategy", "signal", "entry", "exit", "momentum", "mean reversion",
    "breakout", "trend", "arbitrage", "market making", "scalping", "swing",
    "grid", "dca", "dollar cost", "pairs trading", "statistical arbitrage",
    "reinforcement learning", "machine learning", "lstm", "transformer",
    "neural", "xgboost", "lightgbm", "random forest", "genetic algorithm",
    "evolutionary", "hyperopt", "optuna", "bayesian",
]

DATA_SOURCE_KEYWORDS: list[str] = [
    "order book", "orderbook", "depth", "tick data", "level 2", "l2",
    "sentiment", "social", "twitter", "reddit", "news", "nlp",
    "on-chain", "onchain", "whale", "exchange flows",
    "funding", "open interest", "oi", "liquidation", "basis",
    "options", "implied volatility", "iv", "put call",
    "macro", "fed", "interest rate", "economic",
    "alternative data", "satellite", "credit card",
    "fear greed", "fear and greed",
    "cme gap", "futures premium",
]

RISK_KEYWORDS: list[str] = [
    "kelly criterion", "kelly", "volatility targeting", "var", "value at risk",
    "cvar", "conditional value", "expected shortfall",
    "regime detection", "regime", "hidden markov", "hmm",
    "correlation", "portfolio optimization", "markowitz",
    "trailing stop", "dynamic stop", "atr stop",
    "hedging", "hedge", "delta neutral",
    "circuit breaker", "drawdown", "max drawdown",
]

ARCHITECTURE_KEYWORDS: list[str] = [
    "microservices", "plugin", "webhook", "telegram", "discord",
    "backtesting", "paper trading", "simulation",
    "multi-exchange", "cross exchange", "aggregator",
    "async", "asyncio", "concurrent", "parallel",
    "database", "timescale", "influx", "redis",
    "api", "rest api", "websocket", "streaming",
    "docker", "kubernetes", "containerize",
    "dashboard", "ui", "frontend", "grafana",
    "hyperopt", "optimization", "parameter tuning",
    "walk forward", "walk-forward", "rolling window",
]

ML_AI_KEYWORDS: list[str] = [
    "reinforcement learning", "rl", "deep learning", "neural network",
    "lstm", "gru", "transformer", "attention",
    "classification", "regression", "prediction",
    "feature engineering", "feature importance",
    "ensemble", "stacking", "boosting",
    "backtesting optimization", "hyperparameter",
    "online learning", "adaptive",
    "llm", "gpt", "claude", "openai", "anthropic",
    "embedding", "rag", "retrieval",
]


# ---------------------------------------------------------------------------
# GitHub API client (no auth, rate-limit aware)
# ---------------------------------------------------------------------------

class GitHubClient:
    """Minimal GitHub REST API client using urllib (stdlib only)."""

    REQUEST_DELAY = 1.0  # seconds between requests

    def __init__(self):
        self._rate_remaining: int = 60
        self._rate_reset: int = 0

    def _get(self, url: str, accept: str = "application/vnd.github.v3+json") -> dict | list | None:
        """Make a GET request to the GitHub API.

        Returns parsed JSON or None on error.
        Respects rate limiting via X-RateLimit-Remaining header.
        """
        # Back off if we're nearly out of rate limit
        if self._rate_remaining < 3:
            now = int(time.time())
            wait = max(0, self._rate_reset - now) + 5
            console.print(f"[yellow]Rate limit nearly exhausted — waiting {wait}s...[/yellow]")
            time.sleep(wait)

        req = urllib.request.Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": "finance-agent-competitive-intel/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                # Update rate-limit state from headers
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                if remaining:
                    self._rate_remaining = int(remaining)
                if reset:
                    self._rate_reset = int(reset)

                body = resp.read().decode("utf-8")
                return json.loads(body)

        except urllib.error.HTTPError as e:
            if e.code == 403:
                console.print(f"[red]Rate limited (403) on {url}[/red]")
                time.sleep(60)
            elif e.code == 404:
                console.print(f"[dim]Not found (404): {url}[/dim]")
            else:
                console.print(f"[yellow]HTTP {e.code} for {url}: {e.reason}[/yellow]")
            return None
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            console.print(f"[yellow]Request failed for {url}: {e}[/yellow]")
            return None
        finally:
            time.sleep(self.REQUEST_DELAY)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def get_repo_meta(self, repo: str) -> dict | None:
        return self._get(f"{GITHUB_API}/repos/{repo}")  # type: ignore[return-value]

    def get_releases(self, repo: str, n: int = 3) -> list[dict]:
        data = self._get(f"{GITHUB_API}/repos/{repo}/releases?per_page={n}")
        return data if isinstance(data, list) else []

    def get_tags(self, repo: str, n: int = 3) -> list[dict]:
        data = self._get(f"{GITHUB_API}/repos/{repo}/tags?per_page={n}")
        return data if isinstance(data, list) else []

    def get_commits(self, repo: str, n: int = 20) -> list[dict]:
        data = self._get(f"{GITHUB_API}/repos/{repo}/commits?per_page={n}")
        return data if isinstance(data, list) else []

    def get_readme(self, repo: str) -> str | None:
        """Return decoded README text or None."""
        data = self._get(
            f"{GITHUB_API}/repos/{repo}/readme",
            accept="application/vnd.github.v3.raw",
        )
        # Raw accept returns a string-like bytes blob via the API
        # but urllib returns bytes; if _get returns the raw string already we're fine
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            # v3+json fallback — content is base64
            import base64
            content = data.get("content", "")
            encoding = data.get("encoding", "base64")
            if encoding == "base64":
                try:
                    return base64.b64decode(content.replace("\n", "")).decode("utf-8", errors="replace")
                except Exception:
                    return None
        return None

    def search_strategy_files(self, repo: str) -> list[dict]:
        """Search for files with 'strategy' in the filename."""
        data = self._get(
            f"{GITHUB_API}/search/code?q=filename:strategy+repo:{repo}&per_page=10"
        )
        if isinstance(data, dict):
            return data.get("items", [])
        return []


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RepoIntel:
    name: str
    repo_slug: str
    stars: int = 0
    forks: int = 0
    language: str = ""
    description: str = ""
    last_release: str = ""
    last_release_date: str = ""
    recent_releases: list[str] = field(default_factory=list)
    commit_count_scanned: int = 0
    recent_commits: list[str] = field(default_factory=list)
    strategy_files: list[str] = field(default_factory=list)
    # Extracted features
    strategies_found: list[str] = field(default_factory=list)
    data_sources_found: list[str] = field(default_factory=list)
    risk_techniques_found: list[str] = field(default_factory=list)
    architecture_patterns_found: list[str] = field(default_factory=list)
    ml_ai_features_found: list[str] = field(default_factory=list)
    # Raw text for analysis
    readme_snippet: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _find_keywords(text: str, keywords: list[str]) -> list[str]:
    """Case-insensitive keyword match. Returns deduplicated list of hits."""
    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        if kw in text_lower and kw not in seen:
            found.append(kw)
            seen.add(kw)
    return found


def _deduplicate_strategies(raw: list[str]) -> list[str]:
    """Remove trivially generic hits like bare 'strategy' or 'signal'."""
    generics = {"strategy", "signal", "entry", "exit"}
    result = [s for s in raw if s not in generics]
    # If nothing specific, keep the generics so we don't produce an empty list
    return result if result else raw


def extract_intel(
    text: str,
    commit_messages: list[str],
    strategy_filenames: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Extract strategy, data, risk, architecture, ML keywords from text."""
    # Combine all text sources
    combined = " ".join([text] + commit_messages + strategy_filenames)

    strategies = _deduplicate_strategies(_find_keywords(combined, STRATEGY_KEYWORDS))
    data_sources = _find_keywords(combined, DATA_SOURCE_KEYWORDS)
    risk_techniques = _find_keywords(combined, RISK_KEYWORDS)
    architecture = _find_keywords(combined, ARCHITECTURE_KEYWORDS)
    ml_ai = _find_keywords(combined, ML_AI_KEYWORDS)

    return strategies, data_sources, risk_techniques, architecture, ml_ai


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_repo(client: GitHubClient, name: str, slug: str) -> RepoIntel:
    """Scan a single GitHub repo and return structured intelligence."""
    intel = RepoIntel(name=name, repo_slug=slug)

    console.print(f"\n[bold cyan]Scanning {name} ({slug})...[/bold cyan]")

    # 1. Repo metadata
    meta = client.get_repo_meta(slug)
    if meta is None:
        intel.error = "Failed to fetch repo metadata"
        console.print(f"  [red]Could not reach {slug}[/red]")
        return intel

    intel.stars = meta.get("stargazers_count", 0)
    intel.forks = meta.get("forks_count", 0)
    intel.language = meta.get("language") or ""
    intel.description = meta.get("description") or ""
    console.print(f"  [dim]{intel.stars:,} stars, {intel.forks:,} forks, {intel.language}[/dim]")

    # 2. Releases / tags
    console.print("  [dim]Fetching releases...[/dim]")
    releases = client.get_releases(slug, n=3)
    if releases:
        for rel in releases:
            tag = rel.get("tag_name", "?")
            date = rel.get("published_at", "")[:10]
            intel.recent_releases.append(f"{tag} ({date})")
        intel.last_release = releases[0].get("tag_name", "")
        intel.last_release_date = releases[0].get("published_at", "")[:10]
    else:
        # Fall back to tags if no formal releases
        tags = client.get_tags(slug, n=3)
        for t in tags:
            intel.recent_releases.append(t.get("name", "?"))
        if tags:
            intel.last_release = tags[0].get("name", "")

    # 3. Recent commits
    console.print("  [dim]Fetching commits...[/dim]")
    commits = client.get_commits(slug, n=20)
    intel.commit_count_scanned = len(commits)
    commit_messages: list[str] = []
    for c in commits:
        msg = c.get("commit", {}).get("message", "")
        short = msg.split("\n")[0][:100]
        intel.recent_commits.append(short)
        commit_messages.append(msg)

    # 4. README
    console.print("  [dim]Fetching README...[/dim]")
    readme = client.get_readme(slug) or ""
    intel.readme_snippet = readme[:500].replace("\n", " ")

    # 5. Strategy files (search API — note: counted against separate search rate limit)
    console.print("  [dim]Searching strategy files...[/dim]")
    strategy_items = client.search_strategy_files(slug)
    time.sleep(2)  # Extra pause: search API is stricter (10 req/min unauth)
    intel.strategy_files = [item.get("name", "") for item in strategy_items]

    # 6. Extract keywords
    (
        intel.strategies_found,
        intel.data_sources_found,
        intel.risk_techniques_found,
        intel.architecture_patterns_found,
        intel.ml_ai_features_found,
    ) = extract_intel(
        text=readme,
        commit_messages=commit_messages,
        strategy_filenames=intel.strategy_files,
    )

    console.print(
        f"  [green]Done.[/green] "
        f"Strategies: {len(intel.strategies_found)}, "
        f"Data sources: {len(intel.data_sources_found)}, "
        f"ML/AI: {len(intel.ml_ai_features_found)}"
    )
    return intel


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def compute_gaps(all_intel: list[RepoIntel]) -> dict[str, list[tuple[str, str]]]:
    """Compare competitor features against our known feature set.

    Returns a dict of category -> list of (feature, source_repo) tuples.
    """
    gaps: dict[str, list[tuple[str, str]]] = {
        "strategies": [],
        "data_sources": [],
        "risk_techniques": [],
        "architecture": [],
        "ml_ai": [],
    }

    seen_strategies: set[str] = set(OUR_STRATEGIES)
    seen_data: set[str] = set(OUR_DATA_SOURCES)
    seen_features: set[str] = set(OUR_FEATURES)

    for intel in all_intel:
        for s in intel.strategies_found:
            if s not in seen_strategies and s not in {"strategy", "signal", "entry", "exit"}:
                gaps["strategies"].append((s, intel.name))
                seen_strategies.add(s)

        for d in intel.data_sources_found:
            if d not in seen_data:
                gaps["data_sources"].append((d, intel.name))
                seen_data.add(d)

        for r in intel.risk_techniques_found:
            if r not in seen_features and r not in {"drawdown", "stop loss", "trailing stop"}:
                gaps["risk_techniques"].append((r, intel.name))
                seen_features.add(r)

        for a in intel.architecture_patterns_found:
            if a not in seen_features:
                gaps["architecture"].append((a, intel.name))
                seen_features.add(a)

        for m in intel.ml_ai_features_found:
            if m not in seen_features:
                gaps["ml_ai"].append((m, intel.name))
                seen_features.add(m)

    return gaps


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def print_repo_summary(all_intel: list[RepoIntel]):
    """Print per-repo summary table."""
    console.print()
    table = Table(title="Competitor Repo Overview", show_lines=True)
    table.add_column("Repo", style="bold", min_width=12)
    table.add_column("Stars", justify="right", min_width=8)
    table.add_column("Last Release", min_width=18)
    table.add_column("Commits (last 20)", min_width=18, justify="center")
    table.add_column("Strategy Files", min_width=14, justify="center")
    table.add_column("Key Language", min_width=12)
    table.add_column("Description", min_width=30)

    for intel in all_intel:
        if intel.error:
            table.add_row(
                intel.name,
                "-",
                "-",
                "-",
                "-",
                "-",
                f"[red]{intel.error}[/red]",
            )
            continue

        release = intel.last_release or "[dim]no releases[/dim]"
        if intel.last_release_date:
            release += f"\n[dim]{intel.last_release_date}[/dim]"

        table.add_row(
            intel.name,
            f"{intel.stars:,}",
            release,
            str(intel.commit_count_scanned),
            str(len(intel.strategy_files)),
            intel.language or "-",
            (intel.description[:60] + "...") if len(intel.description) > 60 else intel.description,
        )

    console.print(table)


def print_feature_matrix(all_intel: list[RepoIntel]):
    """Print a matrix of features found per repo."""
    console.print()

    categories = [
        ("Strategies / Signals", "strategies_found"),
        ("Data Sources",         "data_sources_found"),
        ("Risk Techniques",      "risk_techniques_found"),
        ("Architecture",         "architecture_patterns_found"),
        ("ML / AI",              "ml_ai_features_found"),
    ]

    for cat_label, attr in categories:
        table = Table(title=f"Feature Matrix: {cat_label}", show_lines=True)
        table.add_column("Feature", style="bold", min_width=24)
        for intel in all_intel:
            table.add_column(intel.name, justify="center", min_width=12)

        # Collect all unique features across repos
        all_features: set[str] = set()
        for intel in all_intel:
            all_features.update(getattr(intel, attr, []))

        for feat in sorted(all_features):
            row = [feat]
            for intel in all_intel:
                has_it = feat in getattr(intel, attr, [])
                row.append("[green]YES[/green]" if has_it else "[dim]-[/dim]")
            table.add_row(*row)

        if all_features:
            console.print(table)
        else:
            console.print(f"[dim]No {cat_label} found across any repo.[/dim]")


def print_gaps(gaps: dict[str, list[tuple[str, str]]]):
    """Print features competitors have that we don't."""
    console.print()
    console.print(
        Panel(
            "[bold]Features We're Missing[/bold]  "
            "(found in competitors, not in our agent)",
            style="yellow",
        )
    )

    category_labels = {
        "strategies":     "Strategies / Signal Types",
        "data_sources":   "Data Sources",
        "risk_techniques": "Risk Management Techniques",
        "architecture":   "Architecture Patterns",
        "ml_ai":          "ML / AI Features",
    }

    for cat, items in gaps.items():
        if not items:
            continue
        table = Table(title=category_labels.get(cat, cat), show_lines=False, box=None)
        table.add_column("Feature", style="bold yellow", min_width=28)
        table.add_column("First seen in", style="dim", min_width=14)

        for feat, source in items:
            table.add_row(feat, source)

        console.print(table)
        console.print()


def print_strategy_ideas(all_intel: list[RepoIntel]):
    """Print strategy ideas extracted from competitor commit messages."""
    console.print()
    console.print(
        Panel(
            "[bold]Strategy Ideas from Competitor Commits[/bold]",
            style="magenta",
        )
    )

    STRATEGY_COMMIT_RE = re.compile(
        r"\b(strategy|signal|indicator|feature|add|implement|new|improve)\b",
        re.IGNORECASE,
    )

    for intel in all_intel:
        if not intel.recent_commits:
            continue

        interesting = [
            msg for msg in intel.recent_commits
            if STRATEGY_COMMIT_RE.search(msg)
               and len(msg) > 15
        ][:8]

        if not interesting:
            continue

        table = Table(title=f"{intel.name} — notable commits", show_lines=False, box=None)
        table.add_column("Commit Message", style="dim", min_width=60)

        for msg in interesting:
            table.add_row(msg[:100])

        console.print(table)
        console.print()


def print_recent_releases(all_intel: list[RepoIntel]):
    """Print recent release tags for each repo."""
    console.print()
    table = Table(title="Recent Releases (last 3 per repo)", show_lines=True)
    table.add_column("Repo", style="bold", min_width=12)
    table.add_column("Releases", min_width=40)

    for intel in all_intel:
        releases_str = "\n".join(intel.recent_releases) if intel.recent_releases else "[dim]none[/dim]"
        table.add_row(intel.name, releases_str)

    console.print(table)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def build_markdown_report(
    all_intel: list[RepoIntel],
    gaps: dict[str, list[tuple[str, str]]],
    scanned_at: str,
) -> str:
    lines: list[str] = []

    lines.append("# Competitive Intelligence Report")
    lines.append(f"\nGenerated: {scanned_at}\n")
    lines.append("Repos scanned: " + ", ".join(i.repo_slug for i in all_intel))
    lines.append("")

    # Per-repo summary
    lines.append("## Repo Summaries\n")
    lines.append("| Repo | Stars | Forks | Language | Last Release | Strategy Files |")
    lines.append("|------|------:|------:|----------|--------------|:--------------:|")
    for intel in all_intel:
        lines.append(
            f"| {intel.name} | {intel.stars:,} | {intel.forks:,} | "
            f"{intel.language or '-'} | {intel.last_release or '-'} | "
            f"{len(intel.strategy_files)} |"
        )
    lines.append("")

    # Features matrix
    lines.append("## Feature Matrix\n")

    category_map = [
        ("Strategies / Signals", "strategies_found"),
        ("Data Sources",         "data_sources_found"),
        ("Risk Techniques",      "risk_techniques_found"),
        ("Architecture Patterns","architecture_patterns_found"),
        ("ML / AI Features",     "ml_ai_features_found"),
    ]

    for cat_label, attr in category_map:
        all_features: set[str] = set()
        for intel in all_intel:
            all_features.update(getattr(intel, attr, []))

        if not all_features:
            continue

        lines.append(f"### {cat_label}\n")
        header = "| Feature | " + " | ".join(i.name for i in all_intel) + " |"
        sep = "|---------|" + "---|" * len(all_intel)
        lines.append(header)
        lines.append(sep)

        for feat in sorted(all_features):
            row = f"| {feat} |"
            for intel in all_intel:
                has_it = feat in getattr(intel, attr, [])
                row += " YES |" if has_it else " - |"
            lines.append(row)
        lines.append("")

    # Gaps section
    lines.append("## Features We're Missing\n")
    category_labels = {
        "strategies":      "Strategies / Signal Types",
        "data_sources":    "Data Sources",
        "risk_techniques": "Risk Management Techniques",
        "architecture":    "Architecture Patterns",
        "ml_ai":           "ML / AI Features",
    }

    for cat, items in gaps.items():
        if not items:
            continue
        lines.append(f"### {category_labels.get(cat, cat)}\n")
        for feat, source in items:
            lines.append(f"- **{feat}** (first seen in: {source})")
        lines.append("")

    # Strategy ideas from commits
    lines.append("## Strategy Ideas from Competitor Commits\n")
    STRATEGY_COMMIT_RE = re.compile(
        r"\b(strategy|signal|indicator|feature|add|implement|new|improve)\b",
        re.IGNORECASE,
    )
    for intel in all_intel:
        interesting = [
            msg for msg in intel.recent_commits
            if STRATEGY_COMMIT_RE.search(msg) and len(msg) > 15
        ][:8]
        if not interesting:
            continue
        lines.append(f"### {intel.name}\n")
        for msg in interesting:
            lines.append(f"- {msg[:120]}")
        lines.append("")

    # Recent releases
    lines.append("## Recent Releases\n")
    for intel in all_intel:
        if intel.recent_releases:
            lines.append(f"**{intel.name}**: " + " | ".join(intel.recent_releases))
        else:
            lines.append(f"**{intel.name}**: no releases found")
    lines.append("")

    return "\n".join(lines)


def export_report(report_md: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_md, encoding="utf-8")
    console.print(f"\n[green]Report saved to {path.resolve()}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan open-source trading repos for competitive intelligence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/competitive_intel.py
  python scripts/competitive_intel.py --repo freqtrade
  python scripts/competitive_intel.py --export
  python scripts/competitive_intel.py --repo freqtrade --export
        """,
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        choices=list(REPOS.keys()),
        metavar="NAME",
        help=f"Scan a specific repo only. Choices: {', '.join(REPOS.keys())}",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Save markdown report to data/competitive_intel_report.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which repos to scan
    if args.repo:
        repos_to_scan = {args.repo: REPOS[args.repo]}
    else:
        repos_to_scan = REPOS

    console.print(
        Panel(
            f"[bold]Competitive Intelligence Scanner[/bold]\n"
            f"Scanning {len(repos_to_scan)} repo(s): {', '.join(repos_to_scan.keys())}\n"
            f"[dim]Rate limit: ~1s between requests. Search API: +2s extra.[/dim]",
            style="cyan",
        )
    )

    client = GitHubClient()
    all_intel: list[RepoIntel] = []
    scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for name, slug in repos_to_scan.items():
        intel = scan_repo(client, name, slug)
        all_intel.append(intel)

    # --- Display results ---
    print_repo_summary(all_intel)
    print_recent_releases(all_intel)
    print_feature_matrix(all_intel)

    gaps = compute_gaps(all_intel)
    print_gaps(gaps)
    print_strategy_ideas(all_intel)

    # Summary panel
    total_strategies = sum(len(i.strategies_found) for i in all_intel)
    total_data = sum(len(i.data_sources_found) for i in all_intel)
    total_gaps = sum(len(v) for v in gaps.values())

    console.print()
    console.print(
        Panel(
            f"[bold]Scan Complete[/bold]\n"
            f"Repos scanned: {len(all_intel)}  |  "
            f"Strategy signals found: {total_strategies}  |  "
            f"Data sources found: {total_data}  |  "
            f"[yellow]Gaps identified: {total_gaps}[/yellow]",
            style="green",
        )
    )

    # --- Export ---
    if args.export:
        report_md = build_markdown_report(all_intel, gaps, scanned_at)
        export_path = Path(__file__).resolve().parent.parent / "data" / "competitive_intel_report.md"
        export_report(report_md, export_path)


if __name__ == "__main__":
    main()
