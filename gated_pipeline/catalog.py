"""Evidence catalog: first-hand facts the team is allowed to cite in a response.

Each card: {tag, claim, note (caution), cap_uses (optional lifetime cap), status}.
Only cards with status "verified" are usable. Code owns the shortlist (BM25 over post words plus
theme words); the classifier only judges the shortlisted pairs.
"""
from __future__ import annotations

from .config import CATALOG, load
from .text import BM25, toks


def card_doc(c: dict) -> list[str]:
    return toks(" ".join([c["tag"].replace("/", " ").replace("-", " "), c["claim"], c.get("note", "")]))


class Catalog:
    def __init__(self, cards: list[dict] | None = None):
        cards = cards if cards is not None else load(CATALOG)["cards"]
        self.pool = [c for c in cards if c.get("status", "verified") == "verified"]
        self.by_tag = {c["tag"]: c for c in self.pool}
        self.bm = BM25([card_doc(c) for c in self.pool])

    def shortlist(self, text: str, extra_words=(), k_text: int = 6, k_theme: int = 3) -> list[dict]:
        """Cards matching the post's own words, plus cards matching its theme's words."""
        seen, out = set(), []
        idx = self.bm.top(toks(text), k_text) + (self.bm.top(list(extra_words), k_theme) if extra_words else [])
        for i in idx:
            if i not in seen:
                seen.add(i)
                out.append(self.pool[i])
        return out
