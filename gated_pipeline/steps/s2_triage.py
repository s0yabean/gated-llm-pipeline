"""Step 2: classifier triage. Is this post worth a considered response, and which kind?

A post is worth a response when it links to something the team can stand behind: it supports a
position in the evidence catalog (agree and add evidence), refutes one (push back from experience),
or shares a theme a card covers (share a related experience). It also needs substance.

Two classifier requests per post; every question in a request runs in parallel:
  1. filters + reply room + theme + routing questions          (every post)
  2. belief link: one Choice per shortlisted card               (posts that survive the filters)
     supports | refutes | same_theme | unrelated
Code owns the shortlist, the thresholds, routing and ranking. Policy changes never re-ask the
classifier: answers are cached, so re-running with new thresholds is free.

Outputs: triage.json, escalate.json (the unsure subset for step 3), targets.json, yield.json.
"""
from __future__ import annotations

import collections
import math
import re
from pathlib import Path

from ..catalog import Catalog
from ..classifier import ask_many, choice, get_classifier, noul
from ..config import dump, load
from ..text import plain

# Broad on purpose. The "words" seed the card shortlist, so a post about football can still reach
# a card about captaining a team.
THEMES = {
    "startups_business": ("Building a company or product, startups, founders, customers, sales, fundraising, pricing, "
                          "proposals, quitting a job to build.",
                          "startup founder customers sales pricing clients proposals revenue"),
    "ai_automation": ("AI, agents, LLMs, automation, evals, workflow tools, the future of work.",
                      "agents automation model eval workflow inbox"),
    "operations_support": ("Operations, customer support, onboarding, processes, response times, churn.",
                           "support onboarding churn reply process"),
    "teams_hiring": ("Hiring, managing, remote work, meetings, delegation, team culture.",
                     "hiring team trial standups delegation remote"),
    "careers_money": ("Careers, jobs, pay, freelancing, contracts, personal finance.",
                      "career job contract freelance pay"),
    "health_habits": ("Sleep, fitness, focus, energy, working hours, habits.", "sleep focus habits energy hours"),
    "sports_teams": ("Sports and team play: football, volleyball, coaching, captaincy.",
                     "football captain team coaching"),
    "none": ("None of these: gaming, celebrity gossip, crypto price talk, or nothing to engage with.", ""),
}

LINK = {"supports": "`post` argues or describes something that `card` backs up, so the team could agree and add "
                    "its own evidence.",
        "refutes": "`post` argues something that `card` goes against, so the team could push back from its own "
                   "experience.",
        "same_theme": "`post` is about a theme `card` covers, so the team could share a related experience, though "
                      "`card` neither backs nor contradicts the post.",
        "unrelated": "No real connection. This includes posts that only use a shared word in passing about "
                     "something else, like a celebrity, food or a joke."}

AREAS = {"pricing": "Pricing, quotes, proposals, invoices, what to charge.",
         "operations": "Customer support, onboarding, inboxes, processes, response times.",
         "hiring": "Hiring, trials, first employees, contractors.",
         "career": "Keeping a job, quitting, contracts, freelancing.",
         "none": "No personal decision or struggle of the author's own."}


def filter_questions() -> dict:
    return {
        "unsafe": noul(
            "Does `post.text` involve child abuse or exploitation, graphic violence or gore, explicit sexual "
            "content, self-harm or suicide, a recent death or tragedy, or a conspiracy claim about named private "
            "individuals?"),
        "sensitive": noul(
            "Is `post.text` about death, dying or grief, or about mental health (depression, anxiety, panic "
            "attacks, burnout as an illness, a breakdown, trauma, therapy, self-harm or suicide)?",
            "A death, grief or a mental-health struggle is the subject of the post or a central part of the "
            "author's story, even if the post also draws a career or business lesson from it.",
            "Neither appears, or only as a figure of speech ('this deadline is killing me', 'dead tired') with no "
            "real loss or health struggle behind it."),
        "promo": noul(
            "Is `post.text` mainly an advertisement or self-promotion: selling or plugging a product, course, "
            "newsletter, service, token, event, link or giveaway?",
            "The main purpose is to sell, plug, or drive clicks or sign-ups.",
            "The main purpose is to share a view, story, question or information. A passing mention of the "
            "author's own work does not count."),
        "bait": noul(
            "Is `post.text` engagement bait, written mainly to farm replies, likes or follows ('comment X below', "
            "'agree?', 'who else...', a generic list of platitudes) rather than sharing the author's own view?"),
        "institutional": noul(
            "Is the author (`post.author_name`, `post.author_handle`) a company, brand, media outlet, news or "
            "aggregator account, government body or politician, rather than an individual person?"),
        "political": noul("Is `post.text` about partisan politics, a named politician, an election or a culture-war "
                          "fight?"),
        "shill": noul(
            "Is `post.text` a campaign or shill post for a crypto token, trading bot, airdrop or AI product "
            "(tagging a project account, a ticker, 'this changes everything', hype threads) rather than the "
            "author's own first-hand experience or opinion?"),
        "reply_room": noul(
            "Could a thoughtful stranger add a new, arguable angle to `post.text` from their own experience?",
            "The post shares a view, claim, dilemma, question or personal story that invites a considered reply.",
            "There is nothing to add beyond agreeing or reacting: announcements, links, jokes, quotes with no view."),
        "substance": noul(
            "Does `post.text` make a real point, claim, question or personal story that an adult professional "
            "could engage with seriously?",
            "It argues something, shares an experience or lesson, or asks a real question.",
            "It is a passing remark, fan comment, vent with no point, meme, joke or list of links."),
        "theme": choice({"themes": {k: v[0] for k, v in THEMES.items()},
                         "question": "Which one of `themes` is `post.text` mainly about?"},
                        {k: None for k in THEMES}),
        # routing
        "two_paths": noul("Does the author of `post.text` name two or more concrete options that they personally "
                          "are choosing between (for example keep the job or go freelance)?"),
        "in_motion": noul("Is the author of `post.text` facing a decision in their own work right now, rather than "
                          "giving advice, a general lesson or a hypothetical?"),
        "stuck": noul("Does the author of `post.text` say that they themselves are stuck, overthinking or unable to "
                      "decide right now?"),
        "life_area": choice({"areas": AREAS, "question": "Which area is the author's own decision or struggle in "
                                                         "`post.text` about?"}, {k: None for k in AREAS}),
        "founder": noul("Is the author of `post.text` an early-stage founder or indie builder talking about their own "
                        "company or product?"),
    }


def link_questions(cands: list[dict]) -> dict:
    return {f"r{i}": choice({"card": {"claim": plain(c["claim"])[:600]},
                             "question": "`card` is one of the team's positions or first-hand experiences. How "
                                         "does `post` relate to it?"}, LINK)
            for i, c in enumerate(cands)}


def followers_bucket(n) -> str:
    if not n:
        return "unknown"
    for lim, name in ((1_000, "under 1k"), (10_000, "1k-10k"), (100_000, "10k-100k"), (1_000_000, "100k-1M")):
        if n < lim:
            return name
    return "over 1M"


def state_for(p: dict) -> dict:
    return {"post": {"author_handle": "@" + p["handle"].lstrip("@"), "author_name": p.get("author_name", ""),
                     "author_followers": followers_bucket(p.get("followers")), "text": p["text"]}}


# ------------------------------------------------------------ policy (code, not prompts)
# Low sensitive line on purpose: a wrongly skipped post costs nothing, a reply under someone's grief
# costs trust.
DROP = {"sensitive": 0.4, "unsafe": 0.5, "promo": 0.7, "bait": 0.7, "shill": 0.7, "institutional": 0.6,
        "political": 0.6}
REVIEW_FLOOR = 0.35    # a filter between this and its DROP line sends the post to review
LINK_PASS = 0.65       # best card link strength (1 - P(unrelated)) needed to pass
LINK_REVIEW = 0.45     # between this and LINK_PASS goes to review
THEME_ONLY_PASS = 0.9  # a link that only shares a theme must be this strong to pass (in the original
                       # backtest most false passes were weak theme-only links on personal stories)
LLM_PASS = 10          # LLM relevance (0-18) needed when an LLM score is present


def code_spam_signals(text: str) -> list[str]:
    """Deterministic spam tells the classifier should not have to judge."""
    sig = []
    if sum(1 for ch in text if 0x1D400 <= ord(ch) <= 0x1D7FF) >= 8:
        sig.append("fancy_unicode")          # math-alphabet bold text used by shill campaigns
    if re.findall(r"(?<![\w$])\$[A-Za-z]{2,10}\b", text):
        sig.append("cashtag")
    if len(re.findall(r"#\w+", text)) >= 3:
        sig.append("hashtags")
    if re.search(r"0x[a-fA-F0-9]{20,}", text):
        sig.append("contract_address")
    if len(re.sub(r"https?://\S+|@\w+", "", text).strip()) < 40:
        sig.append("mostly_links_or_mentions")
    return sig


def filter_stage(p: dict, a: dict):
    n = {k: a[k]["noul"] for k in a if a[k]["type"] == "noul"}
    spam = code_spam_signals(p["text"])
    hard = [k for k, t in DROP.items() if n[k] >= t]
    if len(spam) >= 2 or "contract_address" in spam:
        hard.append("code_spam")
    if n["reply_room"] < REVIEW_FLOOR:
        hard.append("no_reply_room")
    if n["substance"] < 0.4:
        hard.append("no_substance")
    review = [k for k, t in DROP.items() if REVIEW_FLOOR <= n[k] < t]
    review += ["code_" + s for s in spam if len(spam) == 1]
    if REVIEW_FLOOR <= n["reply_room"] < 0.65:
        review.append("reply_room_borderline")
    if 0.4 <= n["substance"] < 0.6:
        review.append("substance_borderline")
    return n, hard, review


def best_links(cands: list[dict], la: dict) -> list[dict]:
    links = []
    for i, c in enumerate(cands):
        pr = la[f"r{i}"]["probabilities"]
        strength = 1 - pr.get("unrelated", 0)
        kind = max((k for k in pr if k != "unrelated"), key=lambda k: pr[k])
        links.append({"tag": c["tag"], "strength": round(strength, 3), "kind": kind,
                      "p": {k: round(v, 3) for k, v in pr.items()}})
    return sorted(links, key=lambda x: -x["strength"])


def decide(p: dict, a: dict, n: dict, hard: list, review: list, links: list) -> dict:
    hard, review = list(hard), list(review)
    theme = a["theme"]["choice"]
    best = links[0] if links else {"strength": 0.0, "kind": None, "tag": None}
    if not hard:
        if best["strength"] < LINK_REVIEW:
            hard.append("no_belief_link")
        elif best["strength"] < LINK_PASS:
            review.append("belief_link_borderline")
        if theme == "none" and a["theme"]["probabilities"].get("none", 0) > 0.6:
            review.append("no_theme")
        if best["kind"] == "same_theme" and LINK_PASS <= best["strength"] < THEME_ONLY_PASS:
            review.append("theme_only_link")

    # A filter only just over its line is soft. A strong link makes promo/shill soft up to 0.9, since
    # builders sharing their own work often trip those lines.
    def is_soft(r):
        if r in ("promo", "shill") and best["strength"] >= 0.85:
            return n[r] < 0.9
        return r in DROP and n[r] < 0.75

    soft_only = hard and all(is_soft(r) for r in hard)
    if "unsafe" in hard or "sensitive" in hard:
        status, review = "drop", []   # never spend a reasoning call to rescue unsafe or sensitive content
    elif hard and not soft_only:
        status = "drop"
    elif hard or review:
        status = "escalate"
        review = review or ["soft_" + r for r in hard]
    else:
        status = "pass"
    clf_status = status

    # Hybrid with an LLM relevance score, when step 3 has added one. The classifier is the first pass;
    # the LLM settles what it was unsure about; a disagreement goes to human review; unsafe always drops.
    # "review" is the human queue; "escalate" means "needs the LLM".
    llm = p.get("llm_relevance")
    if llm is not None and "unsafe" not in hard and "sensitive" not in hard:
        llm_pass = llm >= LLM_PASS
        if clf_status == "escalate":
            status = "pass" if llm_pass else "drop"
        elif (clf_status == "pass") != llm_pass:
            status = "review"
            review = review + ["classifier_llm_disagree"]

    area = a["life_area"]["choice"]
    route = "evidence_reply"
    plug = n["two_paths"] * n["in_motion"] * (1 - n["bait"])
    if (plug >= 0.5 or n["stuck"] * n["in_motion"] >= 0.5) and area != "none":
        route = "direct_help"
    elif n["founder"] >= 0.7 and theme in ("startups_business", "ai_automation"):
        route = "founder_intro"

    views = p.get("views") or 0
    age = max(p.get("age_hours") or 24, 1)
    velocity = views / age
    rank = best["strength"] * n["reply_room"] * (1 - n["bait"]) * (1 + 0.15 * math.log10(1 + velocity))
    return {"status": status, "classifier_status": clf_status, "llm_relevance": llm, "drop_reasons": hard,
            "escalate_reasons": review, "route": route, "theme": theme, "area": area,
            "area_p": round(a["life_area"]["probabilities"].get(area, 0), 3),
            "link": best["strength"], "link_kind": best["kind"], "link_tag": best["tag"], "links_top3": links[:3],
            "reply_room": round(n["reply_room"], 2), "plug": round(plug, 2), "views_per_hour": round(velocity, 1),
            "rank": round(rank, 4), "nouls": {k: round(v, 3) for k, v in n.items()}}


def run(d: Path, posts: list[dict] | None = None, llm_scores: dict | None = None, workers: int = 4) -> list[dict]:
    clf = get_classifier()
    posts = [p for p in (posts if posts is not None else load(d / "ingest.json")) if p.get("text")]
    llm_scores = llm_scores if llm_scores is not None else (
        load(d / "llm_scores.json") if (d / "llm_scores.json").exists() else {})
    fq = filter_questions()
    first = ask_many(clf, [(state_for(p), fq) for p in posts], workers)

    cat = Catalog()
    staged, link_jobs = [], []
    for p, ans in zip(posts, first):
        n, hard, review = filter_stage(p, ans)
        cands = []
        if "unsafe" not in hard and "no_substance" not in hard and \
                not [h for h in hard if h in DROP and n.get(h, 0) >= 0.75]:
            cands = cat.shortlist(p["text"], THEMES[ans["theme"]["choice"]][1].split())
            if cands:
                link_jobs.append(({"post": p["text"][:3000]}, link_questions(cands)))
        staged.append((p, ans, n, hard, review, cands))
    second = iter(ask_many(clf, link_jobs, workers))

    out = []
    for p, ans, n, hard, review, cands in staged:
        links = best_links(cands, next(second)) if cands else []
        p = p | ({"llm_relevance": llm_scores[p["url"]]} if p["url"] in llm_scores else {})
        out.append({**{k: p.get(k) for k in ("id", "url", "handle", "text", "source", "views", "age_hours",
                                             "followers", "baseline_relevance")},
                    **decide(p, ans, n, hard, review, links), "answers": ans})
    out.sort(key=lambda r: -r["rank"])
    dump(d / "triage.json", out)
    dump(d / "escalate.json", [r for r in out if r["status"] == "escalate"])
    dump(d / "targets.json", [r for r in out if r["status"] == "pass"])

    per_src = collections.defaultdict(collections.Counter)
    for r in out:
        c = per_src[r.get("source") or "?"]
        c["total"] += 1
        c[r["status"]] += 1
    dump(d / "yield.json", {s: dict(c) | {"pass_rate": round(c["pass"] / c["total"], 2)} for s, c in per_src.items()})

    counts = dict(collections.Counter(r["status"] for r in out))
    print(f"[2 triage] {len(out)} posts: {counts}; drop reasons "
          f"{dict(collections.Counter(x for r in out for x in r['drop_reasons']))}")
    print(f"           {clf.usage_line()}")
    return out
