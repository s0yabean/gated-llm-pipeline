"""Paths, env loading and small JSON helpers. No hard-coded machine paths: everything is relative
to the repo root or set by an environment variable (see .env.example)."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env without overriding the real environment or printing values."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        m = re.match(r"^(?:export\s+)?([A-Z0-9_]+)=(.*)$", line.strip())
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


load_dotenv()

DATA = Path(os.environ.get("GLP_DATA_DIR", ROOT / "data"))
RUNS = Path(os.environ.get("GLP_RUNS_DIR", ROOT / "runs"))
CACHE = Path(os.environ.get("GLP_CACHE_DIR", RUNS / ".cache"))

POSTS = DATA / "sample_posts.json"
CATALOG = DATA / "evidence_cards.json"
LEDGER = DATA / "seen_ledger.csv"
BLOCKLIST = DATA / "blocklist.json"


def run_dir(name: str | None = None) -> Path:
    name = name or dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def load(path):
    return json.loads(Path(path).read_text())
