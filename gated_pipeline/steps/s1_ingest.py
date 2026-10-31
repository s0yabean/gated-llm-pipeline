"""Step 1: ingest a batch of posts and dedup it in code. No model is asked anything here.

Drops: post id already answered (seen ledger) or seen in an earlier run within --prior-days
(0 by default so the demo can be re-run under new names; use 3 on a live feed),
normalised text already answered, blocklisted author, replies, reposts, posts older than
--hours, and posts under 30 characters. Per-source raw/kept counts are recorded so yield can be
measured per source (in the original, per search query).

Input rows: {id, url, handle, author_name, text, source, age_hours, followers, views,
is_reply, is_repost}. The bundled data/sample_posts.json is synthetic.
"""
from __future__ import annotations

import collections
import csv
import time
from pathlib import Path

from ..config import BLOCKLIST, LEDGER, POSTS, RUNS, dump, load
from ..text import norm_text


def ledger_rows(path: Path = LEDGER) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def blocklist(path: Path = BLOCKLIST) -> set[str]:
    if not path.exists():
        return set()
    return {h.strip().lstrip("@").lower() for h in load(path).get("handles", [])}


def prior_ids(run: Path, days: float) -> set[str]:
    ids, cutoff = set(), time.time() - days * 86400
    if days <= 0:
        return ids
    for f in RUNS.glob("*/ingest.json"):
        if f.parent != run and f.stat().st_mtime >= cutoff:
            ids |= {str(r["id"]) for r in load(f)}
    return ids


def run(d: Path, input_path: Path | None = None, hours: float = 24, prior_days: float = 0) -> list[dict]:
    posts = load(input_path or POSTS)
    try:
        ledger = ledger_rows()
    except FileNotFoundError:
        ledger = []
    seen = {str(r.get("id")) for r in ledger} | prior_ids(d, prior_days)
    seen_text = {norm_text(r.get("source_text", ""))[:200] for r in ledger if r.get("source_text")}
    blocked = blocklist()

    kept, batch_ids = [], set()
    drops = collections.Counter()
    per_source = collections.defaultdict(collections.Counter)
    for p in posts:
        src = per_source[p.get("source", "?")]
        src["raw"] += 1
        pid, text = str(p["id"]), p.get("text", "")
        if pid in seen or pid in batch_ids:
            reason = "seen"
        elif p.get("handle", "").lower() in blocked:
            reason = "blocklist"
        elif p.get("is_reply"):
            reason = "reply"
        elif p.get("is_repost") or text.startswith("RT @"):
            reason = "repost"
        elif (p.get("age_hours") or 0) > hours + 2:
            reason = "too_old"
        elif len(text.strip()) < 30:
            reason = "short"
        elif norm_text(text)[:200] in seen_text:
            reason = "seen_text"
        else:
            reason = None
        if reason:
            drops[reason] += 1
            continue
        batch_ids.add(pid)
        kept.append(p)
        src["kept"] += 1

    dump(d / "ingest.json", kept)
    stats = {"rows": len(kept), "raw": len(posts), "drops": dict(drops),
             "per_source": {k: dict(v) for k, v in per_source.items()}}
    dump(d / "ingest_stats.json", stats)
    print(f"[1 ingest] {len(posts)} raw -> {len(kept)} kept; drops {dict(drops)}")
    return kept
