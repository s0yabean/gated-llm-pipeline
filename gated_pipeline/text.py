"""Deterministic text helpers: tokens, BM25 shortlist, character 6-grams for clone checks."""
from __future__ import annotations

import collections
import math
import re

STOP = set("""a an the and or but if then so of to in on at by for with from as is are was were be been being it its
this that these those i you he she we they me my your our their his her them us not no do does did have has had
can could would should will just about into over more most very than too also only what which who whom how why
when where there here all any some such own same other each few after before up down out off again once
s t don isn doesn it's i'm we're you're get got really thing things one""".split())


def toks(s: str) -> list[str]:
    s = re.sub(r"https?://\S+|@\w+", " ", s or "")
    return [w for w in re.findall(r"[a-z][a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2]


def norm_text(s: str) -> str:
    s = re.sub(r"^RT @\w+:\s*", "", s or "")
    s = re.sub(r"https?://\S+", "", s).lower()
    return re.sub(r"\W+", " ", s).strip()


def plain(s: str) -> str:
    """Backticks inside a System One question are read as pointers into the state, and a bad pointer
    fails the request. Strip them from any free text placed inside a question."""
    return (s or "").replace("`", "'")


def grams(s: str, n: int = 6) -> set[str]:
    s = re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", (s or "").lower())).strip()
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs = [collections.Counter(d) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = collections.Counter(w for d in self.docs for w in d)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def top(self, q: list[str], k: int) -> list[int]:
        q = set(q)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for w in q & d.keys():
                f = d[w]
                s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scores.append((s, i))
        return [i for s, i in sorted(scores, reverse=True)[:k] if s > 0]
