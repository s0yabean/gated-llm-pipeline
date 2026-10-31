"""QA eval: the swap check against the old yes/no topic-fit question, on real vs swapped pairs.

Takes the drafts of a finished run. Builds synthetic swaps: each draft's reply is paired with the
most similar other post in the batch (the hardest mismatch). A good post-fit check blocks or flags
the swapped pairs and leaves the real pairs alone. Writes eval_swap.json.

In the original this ran on 250 real untouched pairs and 250 synthetic swaps; the swap check
replaced topic fit because it blocked more swaps (84% vs 63%) and far fewer real pairs (3% vs 20%).
On the bundled synthetic batch with the mock classifier the numbers only show the harness works.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..classifier import ask_many, get_classifier, noul
from ..config import dump, load
from ..steps import s7_qa


def old_fit(x: dict):
    return ({"post": x["source"][:3000], "reply": x["reply"]},
            {"fit": noul("Does `reply` respond to the actual argument of `post`, not just a shared keyword?")})


def make_swaps(drafts: list[dict]) -> list[dict]:
    toks = [set(re.findall(r"[a-z]{4,}", x["source"].lower())) for x in drafts]
    out = []
    for i, x in enumerate(drafts):
        pool = [j for j in range(len(drafts)) if j != i and drafts[j]["route"] == x["route"]
                and drafts[j]["source"][:200] != x["source"][:200]]
        if not pool:
            continue
        j = max(pool, key=lambda j: len(toks[i] & toks[j]) / max(1, len(toks[i] | toks[j])))
        out.append(x | {"reply": drafts[j]["reply"], "url": x["url"] + "#swap"})
    return out


def run(d: Path) -> dict:
    clf = get_classifier()
    drafts = [x for x in load(d / "drafts.json") if len(x["source"].strip()) >= 30]
    swapped = make_swaps(drafts)

    def own_p(items):
        pool = items + [y for y in drafts if y["url"] not in {i["url"] for i in items}]
        jobs, own = s7_qa.swap_jobs(pool)
        jobs, own = jobs[: len(items)], own[: len(items)]
        it = iter(ask_many(clf, [j for j in jobs if j]))
        return [None if j is None else next(it)["which"]["probabilities"].get(o, 0) for j, o in zip(jobs, own)]

    res = {}
    for name, items in (("real", drafts), ("synthetic_swap", swapped)):
        p = own_p(items)
        fit = [a["fit"]["noul"] for a in ask_many(clf, [old_fit(x) for x in items])]
        res[name] = {"n": len(items),
                     "swap_block": sum(v is not None and v < s7_qa.SWAP_HARD for v in p),
                     "swap_flag": sum(v is not None and v < s7_qa.SWAP_REVIEW for v in p),
                     "old_fit_block": sum(v < 0.4 for v in fit),
                     "old_fit_flag": sum(v < 0.6 for v in fit)}
    dump(d / "eval_swap.json", res)
    for k, v in res.items():
        n = max(1, v["n"])
        print(f"[eval-qa] {k:15} n={v['n']:3}  swap check block {v['swap_block']/n:.0%} flag {v['swap_flag']/n:.0%}"
              f"   old topic fit block {v['old_fit_block']/n:.0%} flag {v['old_fit_flag']/n:.0%}")
    return res
