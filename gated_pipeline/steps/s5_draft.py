"""Step 5: the reasoning model drafts a response for each matched (post, card) pair.

This is the only open-ended generation in the pipeline, and it only runs on posts that passed
triage and have a card at fit >= 2.0. Output: drafts.json (template routes are added by step 6).
"""
from __future__ import annotations

from pathlib import Path

from ..config import dump, load
from ..llm import get_llm


def run(d: Path) -> list[dict]:
    llm = get_llm()
    drafts = [x for x in (load(d / "drafts.json") if (d / "drafts.json").exists() else [])
              if x["route"] != "evidence_reply"]
    n = 0
    for e in load(d / "evidence.json"):
        if e["decision"] != "use":
            continue
        c = e["card"]
        drafts.append({"url": e["url"], "handle": e["handle"], "source": e["text"], "route": "evidence_reply",
                       "reply": llm.draft(e["text"], c), "card_tag": c["tag"], "card_claim": c["claim"],
                       "card_note": c.get("note", ""), "weak_fit": e["weak_fit"], "drafted_by": llm.name})
        n += 1
    dump(d / "drafts.json", drafts)
    print(f"[5 draft] {n} evidence replies drafted by {llm.name}")
    return drafts
