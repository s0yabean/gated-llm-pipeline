"""The reasoning model, used for two narrow jobs only:

  1. relevance(post) -> 0-18, called only on the items the classifier was unsure about (step 3)
  2. draft(post, card) -> a short response that uses the card as its evidence (step 5)

LLM=mock (default) is deterministic and offline. LLM=claude uses the official Anthropic SDK
(`pip install anthropic`, ANTHROPIC_API_KEY or an `ant auth login` profile); CLAUDE_MODEL overrides
the model id.
"""
from __future__ import annotations

import os
import re

from .text import toks

RELEVANCE_PROMPT = """You score inbound social posts for a small team that answers a few posts a day.
Score 0-18: how worth a considered reply is this post, given the team's evidence cards below?
High: a real point, story or question where one card lets the team agree with evidence, push back, or
share a related first-hand experience. Low: promotion, bait, news, gossip, or nothing to add.
Answer with the number only.

Evidence cards:
{cards}

Post:
{post}"""

DRAFT_PROMPT = """Write one reply (under 260 characters) to the post below, as the team.
Use the evidence card as the only first-hand fact; do not invent numbers, names or events.
Add an angle the post did not make. No questions at the end, no hashtags, mentions, links, emoji or dashes.
Return the reply text only.

Post:
{post}

Evidence card:
{claim}
Caution: {note}"""


class MockLLM:
    name = "mock"

    def relevance(self, post: str, cards: list[dict]) -> int:
        pt = set(toks(post))
        best = max((len(pt & set(toks(c["claim"]))) for c in cards), default=0)
        words = len(post.split())
        fp = bool(re.search(r"\b(i|we|my|our)\b", post, re.I))
        return int(min(18, 3 * best + (4 if words >= 25 else 0) + (3 if fp else 0)))

    def draft(self, post: str, card: dict) -> str:
        return f"Seen this from the other side. {card['claim']}"


class ClaudeLLM:
    def __init__(self):
        try:
            import anthropic
        except ImportError as e:  # optional dependency
            raise SystemExit("LLM=claude needs `pip install anthropic`") from e
        self.client = anthropic.Anthropic()
        self.model = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
        self.name = "claude:" + self.model

    def _ask(self, prompt: str, effort: str = "low") -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=2048, output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}])
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def relevance(self, post: str, cards: list[dict]) -> int:
        cards_txt = "\n".join(f"- {c['claim']}" for c in cards)
        m = re.search(r"\d+", self._ask(RELEVANCE_PROMPT.format(cards=cards_txt, post=post)))
        return min(18, int(m.group())) if m else 0

    def draft(self, post: str, card: dict) -> str:
        return self._ask(DRAFT_PROMPT.format(post=post, claim=card["claim"], note=card.get("note", "")), "medium")


def get_llm():
    return ClaudeLLM() if os.environ.get("LLM", "mock").lower() == "claude" else MockLLM()
