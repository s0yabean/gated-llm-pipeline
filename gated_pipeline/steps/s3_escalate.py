"""Step 3: send only the unsure subset to the reasoning model, then re-decide.

The LLM scores each post in escalate.json (0-18 relevance). Scores go to llm_scores.json and
triage is re-run: every classifier answer comes from the cache, so the second pass makes zero
live classifier calls. Unsafe and sensitive posts never reach this step.
"""
from __future__ import annotations

from pathlib import Path

from ..catalog import Catalog
from ..classifier import get_classifier
from ..config import dump, load
from ..llm import get_llm
from . import s2_triage


def run(d: Path) -> list[dict]:
    esc = load(d / "escalate.json")
    triage_n = len(load(d / "triage.json"))
    llm = get_llm()
    cat = Catalog()
    scores = load(d / "llm_scores.json") if (d / "llm_scores.json").exists() else {}
    for r in esc:
        if r["url"] not in scores:
            cards = [cat.by_tag[x["tag"]] for x in r["links_top3"] if x["tag"] in cat.by_tag] or cat.pool[:5]
            scores[r["url"]] = llm.relevance(r["text"], cards)
    dump(d / "llm_scores.json", scores)
    share = len(esc) / max(1, triage_n)
    print(f"[3 escalate] {len(esc)} of {triage_n} posts ({share:.0%}) sent to LLM {llm.name}")
    clf = get_classifier()
    live_before = clf.live
    out = s2_triage.run(d, llm_scores=scores)
    print(f"             re-decide pass: {clf.live - live_before} new live classifier calls (rest from cache)")
    return out
