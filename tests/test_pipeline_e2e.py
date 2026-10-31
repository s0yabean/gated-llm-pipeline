"""End to end on the synthetic sample with the mock classifier and mock LLM. Offline, ~1 s."""
import csv
import tempfile
import unittest
from pathlib import Path

from gated_pipeline import classifier
from gated_pipeline.analysis import backtest, eval_qa
from gated_pipeline.config import load
from gated_pipeline.steps import (s1_ingest, s2_triage, s3_escalate, s4_evidence, s5_draft, s6_template, s7_qa,
                                  s8_assemble)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.d = root / "run"
        self.d.mkdir()
        classifier._DEFAULT = classifier.Cached(classifier.MockClassifier(), root / "cache")

    def tearDown(self):
        classifier._DEFAULT = None
        self.tmp.cleanup()

    def test_all_steps(self):
        kept = s1_ingest.run(self.d)
        self.assertLess(len(kept), len(load(s1_ingest.POSTS)))      # code dedup removed some rows
        first = s2_triage.run(self.d)
        live_after_triage = classifier._DEFAULT.live
        s3_escalate.run(self.d)
        self.assertEqual(classifier._DEFAULT.live, live_after_triage)  # re-decide is fully cached
        tri = load(self.d / "triage.json")
        self.assertEqual(len(tri), len(first))
        self.assertFalse(any(r["status"] == "escalate" for r in tri))
        for r in tri:                                                  # sensitive posts never pass
            if "sensitive" in r["drop_reasons"]:
                self.assertEqual(r["status"], "drop")
        s4_evidence.run(self.d)
        s5_draft.run(self.d)
        s6_template.run(self.d)
        qa = s7_qa.run(self.d)
        self.assertTrue(any("clone of a shipped reply" in h for q in qa for h in q["hard"]))
        ship = s8_assemble.run(self.d)
        with open(self.d / "final_batch.csv") as f:
            self.assertEqual(len(list(csv.DictReader(f))), len(ship))
        res = eval_qa.run(self.d)
        self.assertGreater(res["synthetic_swap"]["swap_flag"], res["real"]["swap_flag"])

    def test_backtest(self):
        s = backtest.run(self.d)
        self.assertEqual(s["posts"], sum(s["grid"].values()))


if __name__ == "__main__":
    unittest.main()
