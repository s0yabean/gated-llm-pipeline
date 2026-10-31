"""CLI.

  python -m gated_pipeline all --run demo          # steps 1-8
  python -m gated_pipeline triage --run demo       # one step
  python -m gated_pipeline backtest --run bt
  python -m gated_pipeline eval-qa --run demo      # after a run has drafts
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import backtest, eval_qa
from .classifier import get_classifier
from .config import run_dir
from .steps import (s1_ingest, s2_triage, s3_escalate, s4_evidence, s5_draft, s6_template, s7_qa,
                    s8_assemble)

STEPS = {
    "ingest": lambda d, a: s1_ingest.run(d, Path(a.input) if a.input else None, a.hours, a.prior_days),
    "triage": lambda d, a: s2_triage.run(d),
    "escalate": lambda d, a: s3_escalate.run(d),
    "evidence": lambda d, a: s4_evidence.run(d),
    "draft": lambda d, a: s5_draft.run(d),
    "template": lambda d, a: s6_template.run(d),
    "qa": lambda d, a: s7_qa.run(d),
    "assemble": lambda d, a: s8_assemble.run(d),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gated_pipeline")
    ap.add_argument("command", choices=["all", *STEPS, "backtest", "eval-qa"])
    ap.add_argument("--run", default=None, help="run folder name under runs/")
    ap.add_argument("--input", help="posts JSON (default data/sample_posts.json)")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--prior-days", type=float, default=0, help="also dedup against runs from the last N days")
    a = ap.parse_args(argv)
    d = run_dir(a.run)
    if a.command == "all":
        for fn in STEPS.values():
            fn(d, a)
    elif a.command == "backtest":
        backtest.run(d, Path(a.input) if a.input else None)
    elif a.command == "eval-qa":
        eval_qa.run(d)
    else:
        STEPS[a.command](d, a)
    print(get_classifier().usage_line())


if __name__ == "__main__":
    main()
