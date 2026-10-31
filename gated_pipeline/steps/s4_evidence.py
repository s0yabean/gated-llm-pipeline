"""Step 4: pick one evidence card per target, or record a gap.

1. Code: verified cards only; cards the triage link check already tied to this post come first,
   then a BM25 shortlist; lifetime use caps (from the seen ledger) and a per-batch cap.
2. Classifier: one request per post, one Score question (0-3 fit) per (post, card) pair.
3. Code policy: best fit >= REUSE -> use it; nothing fits -> gap. A gap is never "use the closest
   card anyway": it goes to evidence_gaps.json for a person to fill with a new, verified card.
Only targets routed to evidence_reply are handled here.
"""
from __future__ import annotations

import collections
from pathlib import Path

from ..catalog import Catalog
from ..classifier import ask_many, get_classifier, score
from ..config import dump, load
from ..text import plain, toks
from .s1_ingest import ledger_rows

REUSE = 2.0         # expected fit on 0-3 needed to draft with a card
WEAK = 2.3          # 2.0-2.3 is drafted but flagged weak_fit for the drafter and reviewer
BATCH_SHARE = 0.10  # one card may back at most 10% of a batch (min 1)

FIT_LEVELS = [
    "Unrelated: the card has nothing to do with the post's argument, or only shares a word.",
    "Stretch: related topic, but using it would need a forced bridge or wordplay.",
    "Relevant: the card gives the team a first-hand example that bears on the post's actual argument.",
    "Direct: the card supports a sharp, arguable reply to the post's main point.",
]


def fit_questions(cands: list[dict]) -> dict:
    return {f"r{i}": score({"card": {"claim": plain(c["claim"]), "caution": plain(c.get("note", ""))[:300]},
                            "question": "How well could the team use `card` as the first-hand evidence in a reply "
                                        "that adds a new angle to `post`?"}, FIT_LEVELS)
            for i, c in enumerate(cands)}


def pick(ranked, uses, batch_uses, batch_cap):
    """Walk cards best-first; skip capped ones; stop below REUSE. Pure function, unit-tested."""
    skips = []
    for s, c in ranked:
        if s < REUSE:
            break
        cap = int(c["cap_uses"]) if str(c.get("cap_uses") or "").isdigit() else None
        if cap is not None and uses[c["tag"]] >= cap:
            skips.append(c["tag"])
            continue
        if batch_uses[c["tag"]] >= batch_cap:
            skips.append(c["tag"] + " (batch cap)")
            continue
        return (s, c), skips
    return None, skips


def run(d: Path, k: int = 8) -> list[dict]:
    posts = [t for t in load(d / "targets.json") if t["route"] == "evidence_reply"]
    cat = Catalog()
    try:
        ledger = ledger_rows()
    except FileNotFoundError:
        ledger = []
    uses = collections.Counter(r.get("card", "").strip() for r in ledger)
    prior = collections.defaultdict(list)
    for r in ledger:
        prior[r.get("card", "").strip()].append(r.get("reply", ""))

    jobs, shortlists = [], []
    for p in posts:
        linked = [cat.by_tag[x["tag"]] for x in p.get("links_top3", []) if x["tag"] in cat.by_tag and x["strength"] >= 0.5]
        cands = (linked + [cat.pool[i] for i in cat.bm.top(toks(p["text"]), k) if cat.pool[i] not in linked])[:k]
        shortlists.append(cands)
        jobs.append(({"post": p["text"]}, fit_questions(cands)))
    answers = ask_many(get_classifier(), jobs)

    results, gaps = [], []
    batch_uses = collections.Counter()
    batch_cap = max(1, round(len(posts) * BATCH_SHARE))
    for p, cands, ans in zip(posts, shortlists, answers):
        ranked = sorted(((ans[f"r{i}"]["score"], c) for i, c in enumerate(cands)), key=lambda x: -x[0])
        chosen, skips = pick(ranked, uses, batch_uses, batch_cap)
        row = {"url": p["url"], "handle": p["handle"], "text": p["text"],
               "top5": [{"tag": c["tag"], "fit": round(s, 2), "uses": uses[c["tag"]]} for s, c in ranked[:5]],
               "cap_skipped": skips}
        if chosen:
            s, c = chosen
            batch_uses[c["tag"]] += 1
            ptoks = set(toks(p["text"]))
            near = sorted(prior[c["tag"]], key=lambda r: -len(ptoks & set(toks(r))))[:5]
            row |= {"decision": "use", "weak_fit": s < WEAK, "prior_replies_nearest": near,
                    "card": {"tag": c["tag"], "claim": c["claim"], "note": c.get("note", ""), "fit": round(s, 2)}}
        else:
            row |= {"decision": "gap", "best_fit": round(ranked[0][0], 2) if ranked else None}
            gaps.append(row)
        results.append(row)

    dump(d / "evidence.json", results)
    dump(d / "evidence_gaps.json", gaps)
    print(f"[4 evidence] {len(results)} evidence_reply targets: "
          f"{sum(r['decision'] == 'use' for r in results)} matched, {len(gaps)} gaps")
    return results
