"""Step 7: QA gates for every draft. Code first, then one classifier request per draft.

Tier A (code, blocks): length (links count as 23), dashes, question ending, hashtags/emoji/mentions,
banned phrases and opener shapes, numbers not found in the card or post, the fixed template
(approved opener, area phrase, matching link), 6-gram clone check against every shipped reply in
the seen ledger and against the rest of this batch.
Tier A2 (classifier swap check, blocks): each reply is shown with its own post and up to 3 similar
posts from the batch; the classifier picks which post it answers. Picking the wrong post means the
reply is on the wrong row or does not answer its post.
Tier B (classifier rubric and risks, queue only): recorded and used only to send drafts to the LLM
or human review queue, sorted by severity. In the original benchmark the rubric did not separate
drafts reviewers fixed from drafts they passed (AUC 0.65), so it never blocks on its own.

Ledger provenance: `clone_ledger` is a number only when the ledger was really loaded. A missing file
(NO_LEDGER_CONFIGURED) and a file with no usable rows (LEDGER_LOADED_EMPTY) carry their status name
instead, so "never checked" can't read as "checked and clean".
"""
from __future__ import annotations

import collections
import csv
import re
from pathlib import Path

from ..classifier import ask_many, choice, get_classifier, noul, score
from ..config import LEDGER, dump, load
from ..text import grams, jaccard, plain
from . import s6_template
from .s1_ingest import ledger_rows

BANNED = ["was never the hard part", "wearing a productivity costume", "i learned that the expensive way",
          "game changer", "let that sink in"]
OPENER_SHAPES = [r"^the (real|actual) \w+ (isn't|is not)", r"^this flips", r"^nobody calls it", r"only works if",
                 r"sounds \w+ until", r"than most admit", r"^[A-Z]\w+ing \w+ (is|was) "]
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|my|me|mine|we|our|us)\b", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
TEMPLATE_ROUTES = ("direct_help", "founder_intro")

NO_LEDGER_CONFIGURED = "NO_LEDGER_CONFIGURED"
LEDGER_LOADED = "LEDGER_LOADED"
LEDGER_LOADED_EMPTY = "LEDGER_LOADED_EMPTY"

SWAP_HARD, SWAP_REVIEW = 0.2, 0.5   # probability the classifier gives the reply's own post
SENSITIVE_HARD = 0.4                # same line as triage's "sensitive" drop


def x_len(s: str) -> int:
    return len(re.sub(r"https?://\S+", "x" * 23, s))


_FIXED = re.compile(r"We write free guides for owner-led teams on .*$", re.S)


def variable_part(route: str, s: str) -> str:
    """Template routes share a fixed skeleton by design; clone-check only the part that varies."""
    return _FIXED.sub("", s).strip() if route in TEMPLATE_ROUTES else s


def load_ledger():
    """Returns (status, source, entry_count, [(url, grams)])."""
    try:
        rows = ledger_rows()
    except FileNotFoundError:
        return NO_LEDGER_CONFIGURED, str(LEDGER), 0, []
    lg = [(r.get("url"), grams(variable_part("evidence_reply", r.get("reply", "")))) for r in rows if r.get("reply")]
    if not lg:
        return LEDGER_LOADED_EMPTY, str(LEDGER), 0, []
    return LEDGER_LOADED, str(LEDGER), len(lg), lg


_OPENER_RE = [re.compile("^" + re.escape(o).replace(re.escape("{d}"), r"(.{2,60}?)") + "$")
              for o in s6_template.OPENERS]
_TEMPLATE_RE = re.compile(r"^(?P<opener>.+?) We write free guides for owner-led teams on (?P<area>.+?): "
                          r"(?P<link>https://example\.com/guides\S*)$", re.S)


def template_check(r: str) -> list[str]:
    m = _TEMPLATE_RE.match(r.strip())
    if not m:
        return ["template: fixed wording changed"]
    fails = []
    if not any(rx.match(m["opener"].strip()) for rx in _OPENER_RE):
        fails.append("template: opener not in the approved list")
    area, link = m["area"], m["link"].rstrip("/")
    if area == s6_template.GENERIC_AREA:
        if link != s6_template.SITE:
            fails.append("template: generic area needs the site link")
    elif area in s6_template.AREA_PHRASE.values():
        key = next(k for k, v in s6_template.AREA_PHRASE.items() if v == area)
        if link != s6_template.LINKS[key]:
            fails.append(f"template: {area} needs its own link")
    else:
        fails.append(f"template: area phrase '{area}'")
    return fails


def lint(d: dict, ledger_status: str, ledger_grams: list, batch: list):
    r, fails = d["reply"], []
    if x_len(r) >= 280:
        fails.append(f"length {x_len(r)}")
    if "—" in r or "–" in r:
        fails.append("em/en dash")
    if d["route"] in TEMPLATE_ROUTES:
        fails += template_check(r)
    if d["route"] == "evidence_reply":
        if r.rstrip().endswith("?"):
            fails.append("ends on a question")
        if re.search(r"#\w|@\w|https?://", r) or EMOJI.search(r):
            fails.append("hashtag/mention/link/emoji")
        low = r.lower()
        fails += [f"banned phrase: {b}" for b in BANNED if b in low]
        fails += [f"opener shape: {p}" for p in OPENER_SHAPES if re.search(p, r if p.startswith("^[A-Z]") else low)]
        src = (d.get("card_claim", "") + " " + d.get("card_note", "") + " " + d.get("source", "")).lower()
        for num in re.findall(r"\d[\d,.]*", r):
            if num.strip(".,") not in src:
                fails.append(f"number not in card or post: {num}")
    g = grams(variable_part(d["route"], r))
    # Evidence replies: 0.35 Jaccard on 6-grams. Template openers rotate through fixed lines, so only a
    # near-identical opener + detail counts as a clone there.
    clone_line = 0.35 if d["route"] == "evidence_reply" else 0.9
    if ledger_status == LEDGER_LOADED:
        best_ledger = max((jaccard(g, lg) for u, lg in ledger_grams if u != d["url"]), default=0)
        if best_ledger >= clone_line:
            fails.append(f"clone of a shipped reply ({best_ledger:.2f})")
        clone_ledger = round(best_ledger, 2)
    else:
        clone_ledger = ledger_status
    best_batch = max((jaccard(g, grams(variable_part(o["route"], o["reply"]))) for o in batch
                      if o is not d and o["route"] == d["route"]), default=0)
    if best_batch >= clone_line:
        fails.append(f"clone within batch ({best_batch:.2f})")
    return fails, clone_ledger, round(best_batch, 2)


def _tokset(s: str) -> set:
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def swap_jobs(drafts: list[dict], k: int = 3):
    """One Choice per draft: which of these posts was this reply written for?"""
    toks = [_tokset(x["source"]) for x in drafts]
    jobs, keys = [], []
    for i, x in enumerate(drafts):
        if len(x["source"].strip()) < 30:          # no post text to match against: skip, never guess
            jobs.append(None)
            keys.append(None)
            continue
        pool = [j for j in range(len(drafts)) if j != i and drafts[j].get("batch") == x.get("batch")
                and len(drafts[j]["source"].strip()) >= 30 and drafts[j]["source"][:200] != x["source"][:200]]
        near = sorted(pool, key=lambda j: -len(toks[i] & toks[j]) / max(1, len(toks[i] | toks[j])))[:k]
        if not near:
            jobs.append(None)
            keys.append(None)
            continue
        order = near[:]
        order.insert(sum(map(ord, x["url"])) % (len(near) + 1), i)   # own post at a fixed, url-derived slot
        labels = "ABCD"[:len(order)]
        posts = {labels[n]: plain(drafts[j]["source"])[:700] for n, j in enumerate(order)}
        jobs.append(({"reply": plain(x["reply"]), "posts": posts}, {"which": choice(
            "`reply` was written as a reply to exactly one of `posts`. Which one does it answer?",
            {lab: None for lab in labels})}))
        keys.append(labels[order.index(i)])
    return jobs, keys


def _levels(a, b, c):
    return [a, b, c]


def rubric_questions(d: dict) -> dict:
    qs = {
        "claim": score("Does `reply` add an arguable angle to `post` that the post did not already make?", _levels(
            "No: it agrees with or restates the post.", "Partly: a small twist on the post's own point.",
            "Yes: a clear, arguable angle the post did not cover.")),
        "receipt": score("Does `reply` back its point with a first-person, specific experience (a number, name, tool, "
                         "date or event from the team's own work)?", _levels(
            "No personal experience, or only a vague 'in my experience'.", "A personal experience, but generic.",
            "A specific first-hand detail.")),
        "caught": score("Would the author of `post` feel their situation was precisely seen by `reply`?", _levels(
            "No, it could be a reply to any post on the topic.", "Somewhat.", "Yes, it lands on something unsaid.")),
        "hook": score("Does the first sentence of `reply` create tension with or a sharp take on `post`?", _levels(
            "No: a generic opener or restates the post.", "Mildly.", "Yes: it grabs attention on its own.")),
        "pace": score("Does every phrase in `reply` earn its place?", _levels(
            "No: filler, repetition or throat-clearing.", "Mostly tight with some slack.", "Tight throughout.")),
        "close": score("Is the last sentence of `reply` its sharpest line?", _levels(
            "No: it trails off, asks a question or has a call to action.", "Fine but not the sharpest.",
            "Yes: it lands the point.")),
        "persona_misfit": noul("Does `reply` assume something about the author of `post` that the post does not "
                               "support (their role, situation or feelings)?"),
        "third_party": noul("Does `reply` state a fact about a named person, company or product that appears in "
                            "neither `post` nor `card`?"),
        "risk_guru": noul("Does `reply` sound preachy, lecturing or like a motivational platitude?"),
        "risk_mystical": noul("Does `reply` make a certain prediction about the future?"),
        "risk_generic": noul("Could `reply` be posted under many different posts with no change?"),
        "risk_salesy": noul("Does `reply` pitch a product or service?"),
        "risk_offensive": noul("Could `reply` read as rude, dismissive, mocking or insensitive to the author of `post`?"),
        "sensitive_post": noul(
            "Is `post` about death, dying or grief, or about mental health (depression, anxiety, burnout as an "
            "illness, a breakdown, trauma, therapy, self-harm or suicide)?",
            "A death, grief or a mental-health struggle is the subject of the post or central to the author's story.",
            "Neither appears, or only as a figure of speech with no real loss or health struggle behind it."),
    }
    for i, sent in enumerate(sentences(d["reply"])):
        # Only first-person sentences can make claims about the team; the classifier reads others too literally.
        if d["route"] != "evidence_reply" or not FIRST_PERSON.search(sent):
            continue
        qs[f"fact{i}"] = choice({"sentence": plain(sent), "question": "How does `card` relate to `sentence`, which is "
                                                                      "part of the team's reply?"},
                                {"supported": "The sentence states something about the team that `card` says.",
                                 "opinion": "The sentence is a view, argument or advice, not a fact about the team.",
                                 "about_post": "The sentence only refers to what the post said.",
                                 "not_stated": "The sentence states a fact about the team that `card` does not say.",
                                 "contradicted": "The sentence conflicts with `card`."})
    return qs


def sentences(s: str) -> list[str]:
    s = re.sub(r"https?://\S+", "", s)
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if len(x.strip()) > 3]


def verdict(d: dict, fails: list, a: dict, swap: dict | None) -> dict:
    """Code and the swap check can block a draft. Rubric and risk scores only queue it for review."""
    stan = {k: a[k]["score"] for k in ("claim", "receipt", "caught", "hook", "pace", "close")}
    total = sum(stan.values())
    facts = {k: a[k] for k in a if k.startswith("fact")}
    bad_facts = [(c, round(v["probabilities"].get(c, 0), 2)) for v in facts.values()
                 for c in [v["choice"]] if c in ("not_stated", "contradicted")]
    risks = {k: a[k]["noul"] for k in a if k.startswith("risk_")}

    hard = list(fails)
    if a["sensitive_post"]["noul"] >= SENSITIVE_HARD:
        hard.append(f"sensitive post ({a['sensitive_post']['noul']:.2f}), do not reply")
    own_p = None
    if swap:
        own_p = round(swap["probabilities"].get(swap["own"], 0), 2)
        if own_p < SWAP_HARD:
            hard.append(f"swap check: reply fits post {swap['choice']}, not its own ({own_p})")

    review = []   # (severity 0-1, reason)
    if own_p is not None and SWAP_HARD <= own_p < SWAP_REVIEW:
        review.append((1 - own_p, f"swap check unsure ({own_p})"))
    if d["route"] == "evidence_reply":
        review += [(p, f"fact {c} ({p})") for c, p in bad_facts if p >= 0.6]
        if a["third_party"]["noul"] >= 0.6:
            review.append((a["third_party"]["noul"], "unsourced third-party claim"))
        if total < 6.0:
            review.append(((6.0 - total) / 6.0 + 0.5, f"rubric {total:.1f}/12"))
        for k, line in (("risk_guru", 0.8), ("risk_salesy", 0.8), ("risk_offensive", 0.7), ("risk_mystical", 0.7)):
            if risks[k] >= line:
                review.append((risks[k], f"{k} {risks[k]:.2f}"))
    else:
        if d.get("weak_fit"):
            hard.append("no specific detail from the post")
        if risks["risk_offensive"] >= 0.7:
            review.append((risks["risk_offensive"], "risk_offensive"))
    if a["persona_misfit"]["noul"] >= 0.8:
        review.append((a["persona_misfit"]["noul"], "persona misfit"))
    status = "rewrite" if hard else ("review" if review else "pass")
    return {"status": status, "hard": hard, "review": [r for _, r in sorted(review, reverse=True)],
            "review_severity": round(max((sv for sv, _ in review), default=0), 2), "swap_own_p": own_p,
            "rubric": {k: round(v, 2) for k, v in stan.items()}, "total_12": round(total, 1),
            "facts": {k: v["choice"] for k, v in facts.items()}, "risks": {k: round(v, 2) for k, v in risks.items()}}


def run(d: Path, drafts_file: str = "drafts.json") -> list[dict]:
    clf = get_classifier()
    drafts = load(d / drafts_file)
    status, source, count, lg = load_ledger()
    lints = [lint(x, status, lg, drafts) for x in drafts]
    jobs = [({"post": x["source"][:3000], "reply": x["reply"],
              "card": plain(x.get("card_claim", "") + " Caution: " + x.get("card_note", ""))[:1500]},
             rubric_questions(x)) for x in drafts]
    answers = ask_many(clf, jobs)
    sjobs, own = swap_jobs(drafts)
    sans = iter(ask_many(clf, [j for j in sjobs if j]))
    swaps = [None if j is None else (lambda w: {"choice": w["choice"], "own": o, "probabilities": w["probabilities"]})(
        next(sans)["which"]) for j, o in zip(sjobs, own)]

    out = []
    for x, (fails, cl, cb), ans, sw in zip(drafts, lints, answers, swaps):
        out.append({"handle": x["handle"], "url": x["url"], "route": x["route"], "reply": x["reply"],
                    "card_tag": x.get("card_tag"), "clone_ledger": cl, "clone_batch": cb,
                    "ledger_status": status, "ledger_entry_count": count, **verdict(x, fails, ans, sw)})
    qa_name = drafts_file.replace("drafts", "qa")
    dump(d / qa_name, out)
    rank = {"rewrite": 0, "review": 1, "pass": 2}
    with open(d / qa_name.replace(".json", ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["status", "review_severity", "handle", "route", "total_12", "hard", "review", "reply", "card", "url"])
        for o in sorted(out, key=lambda o: (rank[o["status"]], -o["review_severity"])):
            w.writerow([o["status"], o["review_severity"], "@" + o["handle"], o["route"], o["total_12"],
                        "; ".join(o["hard"]), "; ".join(o["review"]), o["reply"], o["card_tag"], o["url"]])
    print(f"[7 qa] ledger {status} ({count} entries); {dict(collections.Counter(o['status'] for o in out))}")
    for o in out:
        if o["status"] != "pass":
            print(f"           {o['status']:7} @{o['handle']:22} {o['hard'] or o['review']}")
    print(f"           {clf.usage_line()}")
    return out
