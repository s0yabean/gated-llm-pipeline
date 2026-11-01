"""The cheap classifier behind every gate, as a pluggable backend.

The request shape follows TypeSafe System One (the API behind Jev): one request carries a `state`
(JSON the questions can point into with `backticks`) and a dict of named questions. Three question
types:

  noul    -> {"type": "noul", "noul": P(true)}
  choice  -> {"type": "choice", "choice": best, "probabilities": {option: p}}
  score   -> {"type": "score", "score": expected level, "probabilities": [p per level]}

All questions in one request are answered in parallel. Backends:

  MockClassifier  local and deterministic (keyword and word-overlap heuristics). Needs no key. It
                  exists so the control flow, caching and tests run anywhere; its probabilities are
                  not a model of anything and its numbers say nothing about Jev.
  JevClassifier   TypeSafe System One over HTTP. TYPESAFE_API_KEY pins jev-1.13.0; otherwise
                  AI_GATEWAY_API_KEY routes through Vercel AI Gateway (unversioned typesafe-ai/jev).

Whatever the backend, `Cached` stores every answer on disk by request hash, so changing a threshold
in code and re-running costs nothing.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import CACHE
from .text import toks


# ------------------------------------------------------------------ question builders

def noul(instructions, criteria_true: str | None = None, criteria_false: str | None = None) -> dict:
    q = {"type": "noul", "instructions": instructions}
    if criteria_true or criteria_false:
        q["criteria"] = {"true": criteria_true or "", "false": criteria_false or ""}
    return q


def choice(instructions, options: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def score(instructions, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


# ------------------------------------------------------------------ backends

class Classifier:
    name = "base"

    def ask(self, state: dict, questions: dict) -> dict:
        raise NotImplementedError


class JevClassifier(Classifier):
    def __init__(self):
        if os.environ.get("TYPESAFE_API_KEY"):
            self.url = "https://api.typesafe.ai/v1/systemone"
            self.key = os.environ["TYPESAFE_API_KEY"]
            self.model = os.environ.get("JEV_MODEL", "jev-1.13.0")
        elif os.environ.get("AI_GATEWAY_API_KEY"):
            self.url = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
            self.key = os.environ["AI_GATEWAY_API_KEY"]
            self.model = "typesafe-ai/jev"
        else:
            raise SystemExit("CLASSIFIER=jev needs TYPESAFE_API_KEY or AI_GATEWAY_API_KEY")
        self.name = "jev:" + self.model
        self.input_tokens = 0

    def ask(self, state, questions, retries: int = 7):
        body = {"model": self.model, "state": state, "questions": questions}
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.key}",
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                last = f"{e.code} {e.read().decode()[:300]}"
                if e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(last) from e
            except Exception as e:  # network or timeout
                last = str(e)
            time.sleep(min(30, 2 ** attempt))
        else:
            raise RuntimeError(f"jev failed: {last}")
        answers = data["answers"]
        conf = (data.get("provider_metadata") or {}).get("typesafe", {}).get("confidence") or {}
        for k, a in answers.items():
            a.setdefault("type", questions[k]["type"])
            if "confidence" not in a and k in conf:
                a["confidence"] = conf[k]
        self.input_tokens += (data.get("usage") or {}).get("input_tokens", 0)
        return answers


# Mock heuristics. Keyword lists stand in for a model so the pipeline is runnable offline.
KW = {
    "sensitive": ["passed away", "funeral", "grief", "grieving", "depression", "anxiety", "panic attack",
                  "therapy", "therapist", "breakdown", "trauma", "mourning", "diagnosed with burnout"],
    "unsafe": ["gore", "graphic violence", "self-harm", "suicide"],
    "promo": ["link in bio", "sign up", "use code", "% off", "waitlist", "register now", "limited seats",
              "buy now", "my course", "my newsletter", "dm me for", "early-bird"],
    "bait": ["agree?", "comment below", "follow for more", "who else", "reply with", "like if", "drop a",
             "retweet if"],
    "political": ["election", "senator", "vote for", "parliament", "candidate", "partisan", "the president"],
    "shill": ["airdrop", "100x", "to the moon", "this changes everything", "presale", "whitelist", "not financial advice"],
}
INSTITUTIONAL = ["news", "official", "daily", "media", "inc", "hq", "agency", "weekly", "team"]
CONTRARIAN = ["overrated", "myth", "doesn't work", "is dead", "wrong", "never works", "waste of", "stop ", "nobody needs"]
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|we|we're|we've|my|our)\b", re.I)


def _post_text(state) -> str:
    p = state.get("post", "")
    return p.get("text", "") if isinstance(p, dict) else str(p)


def _hits(text: str, words) -> int:
    low = text.lower()
    return sum(1 for w in words if w in low)


def _p(hits: int, base: float = 0.05, step: float = 0.4) -> float:
    return round(min(0.97, base + step * hits), 3)


def _overlap(a: str, b: str) -> float:
    ta, tb = set(toks(a)), set(toks(b))
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _flatten(x) -> str:
    if isinstance(x, dict):
        return " ".join(_flatten(v) for v in x.values())
    if isinstance(x, list):
        return " ".join(_flatten(v) for v in x)
    return str(x or "")


def _choice(probs: dict) -> dict:
    tot = sum(probs.values()) or 1.0
    probs = {k: round(v / tot, 3) for k, v in probs.items()}
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}


def _score(expected: float, n_levels: int) -> dict:
    expected = max(0.0, min(n_levels - 1.0, expected))
    lo = int(expected)
    hi = min(lo + 1, n_levels - 1)
    probs = [0.0] * n_levels
    probs[hi] += expected - lo
    probs[lo] += 1 - (expected - lo)
    return {"type": "score", "score": round(expected, 3), "probabilities": [round(p, 3) for p in probs]}


class MockClassifier(Classifier):
    """Deterministic stand-in. Dispatches on question id and type; never calls the network."""
    name = "mock"

    def ask(self, state, questions):
        return {qid: self._one(qid, q, state) for qid, q in questions.items()}

    # -- noul
    def _noul(self, qid, q, state):
        text = _post_text(state)
        low = text.lower()
        words = len(text.split())
        fp = bool(FIRST_PERSON.search(text))
        if qid in KW or qid == "sensitive_post":
            p = _p(_hits(text, KW["sensitive" if qid == "sensitive_post" else qid]))
            if qid == "shill":
                p = _p(_hits(text, KW["shill"]) + len(re.findall(r"(?<![\w$])\$[A-Za-z]{2,10}\b", text)))
        elif qid == "institutional":
            post = state.get("post", {})
            who = (post.get("author_handle", "") + " " + post.get("author_name", "")).lower() if isinstance(post, dict) else ""
            p = 0.85 if set(re.split(r"[^a-z]+", who)) & set(INSTITUTIONAL) else 0.08
        elif qid == "reply_room":
            p = 0.82 if fp and words >= 20 else (0.5 if fp or words >= 25 else 0.2)
            p -= 0.3 * min(1, _hits(text, KW["bait"]) + _hits(text, KW["promo"]))
        elif qid == "substance":
            p = 0.82 if words >= 25 else (0.5 if words >= 14 else 0.2)
        elif qid == "two_paths":
            p = 0.8 if " or " in low and _hits(text, ["should i", "deciding", "choosing between", "torn between"]) else 0.1
        elif qid == "in_motion":
            p = 0.8 if _hits(text, ["should i", "deciding", "this week i", "right now", "considering", "torn"]) else 0.15
        elif qid == "stuck":
            p = 0.8 if _hits(text, ["stuck", "can't decide", "overthinking", "torn"]) else 0.1
        elif qid == "founder":
            p = 0.82 if _hits(text, ["we launched", "i'm building", "my startup", "our startup", "we just shipped",
                                     "i built", "our product", "we're building"]) else 0.15
        elif qid.startswith("n"):        # template naturalness check
            m = re.search(r"'Your (.{3,60}?) caught my eye\.'", _flatten(q["instructions"]))
            words = m.group(1).lower().split() if m else []
            dangling = {"would", "could", "should", "will", "keep", "keeps", "is", "are", "was", "feels", "between"}
            p = 0.8 if words and words[-1] not in dangling and not words[-1].endswith("ing") \
                and words[0] not in dangling else 0.3
        elif qid == "fit":               # old yes/no topic fit, kept for the QA eval
            p = min(0.95, 0.15 + 1.6 * _overlap(state.get("reply", ""), text))
        elif qid == "risk_salesy":
            p = 0.85 if "http" in state.get("reply", "") else 0.05
        elif qid == "risk_guru":
            p = 0.6 if _hits(state.get("reply", ""), ["always", "the secret", "you must", "never forget"]) else 0.15
        elif qid == "risk_generic":
            p = 0.44
        elif qid in ("persona_misfit", "third_party", "risk_offensive", "risk_mystical"):
            p = 0.08
        else:
            p = 0.5
        return {"type": "noul", "noul": round(max(0.01, min(0.99, p)), 3)}

    # -- choice
    def _choice_ans(self, qid, q, state):
        opts = q["criteria"]
        ins = q["instructions"]
        text = _post_text(state)
        if qid == "which":                                   # swap check
            reply = state.get("reply", "")
            return _choice({k: (_overlap(reply, v) + 0.02) ** 2 for k, v in state["posts"].items()})
        if qid.startswith("r") and isinstance(ins, dict) and "card" in ins:   # belief link
            ov = _overlap(text, _flatten(ins["card"]))
            strength = min(0.95, 0.1 + 1.4 * ov)
            kind = "refutes" if _hits(text, CONTRARIAN) else ("supports" if ov >= 0.3 else "same_theme")
            probs = {k: strength * (0.7 if k == kind else 0.15) for k in opts if k != "unrelated"}
            probs["unrelated"] = 1 - strength
            return _choice(probs)
        if qid.startswith("fact"):
            sent = ins.get("sentence", "") if isinstance(ins, dict) else ""
            card = _flatten(state.get("card", ""))
            nums = re.findall(r"\d[\d,.]*", sent)
            if nums and any(n.strip(".,") not in card for n in nums):
                return _choice({k: (0.7 if k == "not_stated" else 0.075) for k in opts})
            if _overlap(sent, card) >= 0.5:
                return _choice({k: (0.8 if k == "supported" else 0.05) for k in opts})
            return _choice({k: (0.7 if k == "opinion" else 0.075) for k in opts})
        if qid == "detail":                                  # pick the most specific phrase
            numwords = {"two", "three", "four", "five", "six", "ten", "first", "second"}

            def spec(p):
                words = p.split()
                content = len(toks(p)) == len(words)
                return (2 * bool(re.search(r"\d", p)) + sum(w[:1].isupper() for w in words[1:])
                        + 1.5 * any(w.lower() in numwords for w in words) + 1.0 * content
                        + 0.5 * (len(words) in (2, 3)) + 0.02 * len(p))
            cands = [k for k in opts if k != "none"]
            if not cands:
                return _choice({"none": 1.0})
            best = max(cands, key=spec)
            return _choice({k: (0.6 if k == best else 0.4 / max(1, len(opts) - 1)) for k in opts})
        # topical choice (theme, area, ...): word overlap with each option's description
        descs = next((v for v in ins.values() if isinstance(v, dict)), opts) if isinstance(ins, dict) else opts
        scores = {}
        for k in opts:
            desc = (descs.get(k) if isinstance(descs, dict) else None) or k
            scores[k] = 0.05 + len(set(toks(text)) & set(toks(desc + " " + k)))
        if "none" in scores:
            scores["none"] = 0.6 if max(v for k, v in scores.items() if k != "none") < 1 else 0.05
        return _choice(scores)

    # -- score
    def _score_ans(self, qid, q, state):
        n = len(q["criteria"])
        ins = q["instructions"]
        reply = state.get("reply", "")
        if isinstance(ins, dict) and "card" in ins:          # card fit, 0-3
            return _score(6 * _overlap(_post_text(state), _flatten(ins["card"])), n)
        if qid == "bears":
            return _score(6 * _overlap(_post_text(state), state.get("passage", "")), n)
        base = {"claim": 1.0 + 0.5 * bool(re.search(r"\b(but|actually|instead)\b", reply, re.I)),
                "receipt": 1.6 if FIRST_PERSON.search(reply) and re.search(r"\d", reply) else 0.8,
                "caught": 0.9, "hook": 1.0, "pace": 1.5 if len(reply) < 220 else 1.0, "close": 1.0}
        return _score(base.get(qid, 1.0), n)

    def _one(self, qid, q, state):
        t = q["type"]
        if t == "noul":
            return self._noul(qid, q, state)
        if t == "choice":
            return self._choice_ans(qid, q, state)
        return self._score_ans(qid, q, state)


class Cached(Classifier):
    """Disk cache keyed by backend name + full request. Counts live vs cached calls."""

    def __init__(self, inner: Classifier, cache_dir: Path | None = None):
        self.inner = inner
        self.name = inner.name
        self.dir = Path(cache_dir or CACHE) / re.sub(r"[^a-z0-9.-]+", "_", inner.name.lower())
        self.live = 0
        self.cached = 0

    def key(self, state, questions) -> str:
        body = {"backend": self.inner.name, "state": state, "questions": questions}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def ask(self, state, questions):
        f = self.dir / f"{self.key(state, questions)}.json"
        if f.exists():
            self.cached += 1
            return json.loads(f.read_text())
        ans = self.inner.ask(state, questions)
        self.live += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(ans))
        return ans

    def usage_line(self) -> str:
        line = f"classifier {self.name}: {self.live} live requests, {self.cached} cached"
        tok = getattr(self.inner, "input_tokens", 0)
        if tok:
            line += f", {tok:,} input tokens (~${tok / 1e6 * 0.042:.4f})"
        return line


_DEFAULT: Cached | None = None


def get_classifier() -> Cached:
    """CLASSIFIER=mock (default) or CLASSIFIER=jev."""
    global _DEFAULT
    if _DEFAULT is None:
        kind = os.environ.get("CLASSIFIER", "mock").lower()
        _DEFAULT = Cached(JevClassifier() if kind == "jev" else MockClassifier())
    return _DEFAULT


def ask_many(clf: Classifier, jobs: list, workers: int = 4) -> list:
    """jobs: list of (state, questions). Answers come back in the same order. Items that still fail
    after the backend's own retries are retried one at a time at the end."""
    out = [None] * len(jobs)
    failed = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(clf.ask, s, q): i for i, (s, q) in enumerate(jobs)}
        for f in cf.as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except RuntimeError as e:
                failed.append(futs[f])
                print(f"  item {futs[f]} failed, will retry: {str(e)[:80]}", flush=True)
    for i in failed:
        time.sleep(5)
        out[i] = clf.ask(*jobs[i])
    return out
