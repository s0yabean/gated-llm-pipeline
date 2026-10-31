"""Step 6: fill the fixed template for direct_help and founder_intro targets. Select, don't generate.

Code cuts candidate phrases out of the post. The classifier picks the one that is the most
specific detail about what the author is doing (one Choice), then checks the top picks read
naturally inside the opener (Nouls). Code fills the template, rotates openers in order and routes
the link by area. No LLM call. Appends rows to drafts.json.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..classifier import ask_many, choice, get_classifier, noul
from ..config import dump, load

TEMPLATE = "{opener} We write free guides for owner-led teams on {area}: {link}"
GENERIC_AREA = "pricing, operations and hiring"
SITE = "https://example.com/guides"
LINKS = {"pricing": SITE + "/pricing", "operations": SITE + "/operations", "hiring": SITE + "/hiring",
         "career": SITE + "/career"}
AREA_PHRASE = {"pricing": "pricing", "operations": "support and operations", "hiring": "hiring",
               "career": "going independent"}
OPENERS = ["Your {d} caught my eye.", "Saw your {d}, liked it.", "Your {d} stood out to me.", "Noticed your {d}.",
           "Your {d} got my attention.", "Came across your {d}.", "Spotted your {d}.", "Your {d} is a solid one.",
           "That {d} of yours caught my attention.", "Your {d} made me stop scrolling."]
EDGE_STOP = set("a an the and or but of to in on at by for with from as is are was were be it its this that my our "
                "your i we you they so just very really also".split())
AREA_MIN_P = 0.6    # below this the generic area phrase and site link are used


def candidate_phrases(text: str, limit: int = 150) -> list[str]:
    body = re.sub(r"https?://\S+|@\w+|#\w+|&\w+;", " ", text)
    out, seen = [], set()
    for clause in re.split(r"[\n.!?,:;()\"]+", body):
        words = re.findall(r"[A-Za-z0-9$+'\-]+", clause)
        for n in range(2, 6):
            for i in range(len(words) - n + 1):
                span = words[i:i + n]
                if span[0].lower() in EDGE_STOP or span[-1].lower() in EDGE_STOP:
                    continue
                p = " ".join(span)
                if p.lower() not in seen:
                    seen.add(p.lower())
                    out.append(p)
    return out[:limit]


def fill(i: int, detail: str | None, area: str, area_p: float) -> str:
    use_area = area in LINKS and area_p >= AREA_MIN_P
    return TEMPLATE.format(opener=OPENERS[i % len(OPENERS)].format(d=detail or "post"),
                           area=AREA_PHRASE[area] if use_area else GENERIC_AREA, link=LINKS[area] if use_area else SITE)


def run(d: Path) -> list[dict]:
    clf = get_classifier()
    posts = [t for t in load(d / "targets.json") if t["route"] in ("direct_help", "founder_intro")]
    cands = [candidate_phrases(p["text"]) for p in posts]
    picks = ask_many(clf, [({"post": p["text"]}, {"detail": choice(
        "Which phrase, copied from `post`, is the most specific detail about what the author is building or doing "
        "(a product, project, milestone, city or industry) that someone could mention back to them?",
        {c: None for c in cs} | {"none": "No phrase names something specific."})}) for p, cs in zip(posts, cands)])
    tops = [[c for c, pr in sorted(pk["detail"]["probabilities"].items(), key=lambda kv: -kv[1]) if c != "none"][:5]
            for pk in picks]
    natural = ask_many(clf, [({"post": p["text"]}, {f"n{i}": noul(
        f"Is 'Your {c} caught my eye.' a natural, grammatical English sentence that correctly refers to "
        f"something in `post`?") for i, c in enumerate(top)}) for p, top in zip(posts, tops)])

    drafts = [x for x in (load(d / "drafts.json") if (d / "drafts.json").exists() else [])
              if x["route"] not in ("direct_help", "founder_intro")]
    for i, (p, top, nat) in enumerate(zip(posts, tops, natural)):
        ok = [(nat[f"n{j}"]["noul"], c) for j, c in enumerate(top)]
        # first natural phrase in the classifier's own ranking order
        detail = next((c for v, c in ok if v >= 0.5), None)
        area = p["area"] if p["route"] == "direct_help" else "none"
        drafts.append({"url": p["url"], "handle": p["handle"], "source": p["text"], "route": p["route"],
                       "reply": fill(i, detail, area, p.get("area_p", 0)), "card_tag": "template",
                       "card_claim": "Fixed template", "card_note": "", "weak_fit": detail is None,
                       "detail": detail, "detail_candidates": top, "drafted_by": "template+classifier"})
    dump(d / "drafts.json", drafts)
    print(f"[6 template] {len(posts)} template replies filled (no LLM)")
    return drafts
