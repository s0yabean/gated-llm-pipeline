"""Backtest: compare classifier triage with the old single-LLM relevance gate on past posts.

Input rows need `baseline_relevance` (0-18 from the old one-prompt LLM scorer; pass at 10+).
Runs step 2 on them with no LLM scores (classifier only), then cross-tabulates:

                  classifier pass | escalate | drop
  baseline pass          .              .        .
  baseline reject        .              .        .

"New finds" are classifier passes the baseline rejected and that were never answered. They are the
rows to hand-check. Writes backtest_summary.json, new_finds.csv, baseline_pass_classifier_drop.csv.
"""
from __future__ import annotations

import collections
import csv
from pathlib import Path

from ..config import POSTS, dump, load
from ..steps import s2_triage
from ..steps.s1_ingest import ledger_rows

BASELINE_PASS = 10


def run(d: Path, input_path: Path | None = None) -> dict:
    posts = [p for p in load(input_path or POSTS) if p.get("baseline_relevance") is not None]
    tri = s2_triage.run(d, posts=posts, llm_scores={})
    try:
        answered = {r.get("url") for r in ledger_rows()}
    except FileNotFoundError:
        answered = set()
    base = {p["url"]: p for p in posts}

    grid = collections.Counter()
    new_finds, base_only = [], []
    for t in tri:
        b_pass = base[t["url"]]["baseline_relevance"] >= BASELINE_PASS
        grid[("baseline_pass" if b_pass else "baseline_reject", t["classifier_status"])] += 1
        if t["classifier_status"] == "pass" and not b_pass and t["url"] not in answered:
            new_finds.append(t)
        if b_pass and t["classifier_status"] == "drop":
            base_only.append(t)
    n = len(tri)
    summary = {"posts": n, "grid": {f"{a}|{b}": v for (a, b), v in sorted(grid.items())},
               "escalate_share": round(sum(t["classifier_status"] == "escalate" for t in tri) / max(1, n), 3),
               "new_finds": len(new_finds),
               "new_find_link_kinds": dict(collections.Counter(t["link_kind"] for t in new_finds)),
               "baseline_pass_classifier_drop": len(base_only),
               "baseline_pass_classifier_drop_reasons": dict(collections.Counter(
                   r for t in base_only for r in t["drop_reasons"]).most_common())}
    dump(d / "backtest_summary.json", summary)
    for name, rows in (("new_finds.csv", new_finds), ("baseline_pass_classifier_drop.csv", base_only)):
        with open(d / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["handle", "url", "text", "link", "link_kind", "link_tag", "theme", "drop_reasons",
                        "baseline_relevance"])
            for t in rows:
                w.writerow(["@" + t["handle"], t["url"], t["text"], t["link"], t["link_kind"], t["link_tag"],
                            t["theme"], "; ".join(t["drop_reasons"]), base[t["url"]]["baseline_relevance"]])

    print(f"[backtest] {n} posts")
    print("                   pass  escalate  drop")
    for row in ("baseline_pass", "baseline_reject"):
        print(f"  {row:16} " + "  ".join(f"{grid[(row, c)]:5}" for c in ("pass", "escalate", "drop")))
    print(f"  escalate share {summary['escalate_share']:.0%}; new finds {len(new_finds)}; "
          f"baseline pass but classifier drop {len(base_only)} {summary['baseline_pass_classifier_drop_reasons']}")
    return summary
