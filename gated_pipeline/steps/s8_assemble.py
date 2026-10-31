"""Step 8: build the final batch from QA results plus reviewer decisions. Nothing is posted.

decisions.json (optional, in the run folder):
  {"approve": [urls moved from review to ship], "hold": [urls held back even though QA passed],
   "why": {url: reason}}
Revisions win: qa_r1.json over qa.json. Output: final_batch.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..config import load


def run(d: Path, decisions: str = "decisions.json") -> list[dict]:
    dec = load(d / decisions) if (d / decisions).exists() else {}
    drafts = {x["url"]: x for x in load(d / "drafts.json")}
    qa = {x["url"]: x for x in load(d / "qa.json")}
    for f in sorted(d.glob("qa_r*.json")):
        for x in load(f):
            qa[x["url"]] = x | {"revised": True}
    ship, held = [], []
    for url, q in qa.items():
        ok = (q["status"] == "pass" and url not in dec.get("hold", [])) or \
             (q["status"] == "review" and url in dec.get("approve", []))
        (ship if ok else held).append(q)
    with open(d / "final_batch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["handle", "url", "source_text", "reply", "card", "route", "total_12", "note"])
        for q in ship:
            note = ("approved from review: " + "; ".join(q["review"])) if q["status"] == "review" else \
                ("revised once" if q.get("revised") else "")
            w.writerow(["@" + q["handle"], q["url"], drafts[q["url"]]["source"], q["reply"], q["card_tag"],
                        q["route"], q["total_12"], note])
    print(f"[8 assemble] ship {len(ship)}, held {len(held)} -> runs/{d.name}/final_batch.csv")
    return ship
