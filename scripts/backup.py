#!/usr/bin/env python3
"""Python wrapper for backup.sh — call from agent before risky operations."""

import subprocess
import sys
from pathlib import Path


def run_backup() -> bool:
    """Run the backup script. Returns True on success."""
    script = Path(__file__).parent / "backup.sh"
    if not script.exists():
        print(f"ERROR: backup script not found at {script}", file=sys.stderr)
        return False
    try:
        result = subprocess.run(
            [str(script)], capture_output=False, text=True, timeout=60
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("ERROR: backup timed out after 60s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: backup failed: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    sys.exit(0 if run_backup() else 1)
