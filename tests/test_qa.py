"""QA gate tests. No network and no real ledger: `ledger_rows` is patched where s7_qa imports it."""
import unittest
from unittest import mock

from gated_pipeline.steps import s6_template, s7_qa as qa


def draft(reply="A plain reply with nothing special going on here at all.", url="https://example.com/posts/1",
          route="evidence_reply", source="", claim=""):
    return {"reply": reply, "url": url, "route": route, "handle": "sample_u", "card_claim": claim,
            "card_note": "", "source": source}


class TestLedgerProvenance(unittest.TestCase):
    def test_missing_file_is_not_configured(self):
        with mock.patch.object(qa, "ledger_rows", side_effect=FileNotFoundError):
            status, _, count, grams = qa.load_ledger()
        self.assertEqual((status, count, grams), (qa.NO_LEDGER_CONFIGURED, 0, []))

    def test_file_with_no_usable_rows_is_loaded_empty_and_distinct(self):
        with mock.patch.object(qa, "ledger_rows", return_value=[{"url": "u1", "reply": ""}, {"url": "u2"}]):
            status, *_ = qa.load_ledger()
        self.assertEqual(status, qa.LEDGER_LOADED_EMPTY)
        self.assertNotEqual(status, qa.NO_LEDGER_CONFIGURED)

    def test_loaded_counts_only_rows_with_a_reply(self):
        rows = [{"url": "u1", "reply": "A real shipped reply with content."}, {"url": "u2", "reply": ""}]
        with mock.patch.object(qa, "ledger_rows", return_value=rows):
            status, _, count, _ = qa.load_ledger()
        self.assertEqual((status, count), (qa.LEDGER_LOADED, 1))

    def test_clone_score_is_a_status_name_unless_loaded(self):
        for status in (qa.NO_LEDGER_CONFIGURED, qa.LEDGER_LOADED_EMPTY):
            fails, cl, _ = qa.lint(draft(), status, [], [])
            self.assertEqual(cl, status)
            self.assertFalse(any(f.startswith("clone of a shipped reply") for f in fails))

    def test_real_clone_blocks_when_loaded(self):
        text = "Running a business alone teaches you things a course never will teach you."
        fails, cl, _ = qa.lint(draft(text), qa.LEDGER_LOADED, [("https://example.com/posts/2", qa.grams(text))], [])
        self.assertTrue(any(f.startswith("clone of a shipped reply") for f in fails))
        self.assertGreaterEqual(cl, 0.35)

    def test_batch_clone_fires_without_a_ledger(self):
        text = "Running a business alone teaches you things a course never will teach you."
        d1, d2 = draft(text, "u1"), draft(text, "u2")
        fails, _, _ = qa.lint(d1, qa.NO_LEDGER_CONFIGURED, [], [d1, d2])
        self.assertTrue(any(f.startswith("clone within batch") for f in fails))


class TestLint(unittest.TestCase):
    def lint(self, reply, **kw):
        return qa.lint(draft(reply, **kw), qa.NO_LEDGER_CONFIGURED, [], [])[0]

    def test_rules(self):
        self.assertIn("ends on a question", self.lint("Is that really the whole story here?"))
        self.assertIn("em/en dash", self.lint("Fixed pricing — it worked for us."))
        self.assertIn("hashtag/mention/link/emoji", self.lint("Fixed pricing worked #pricing"))
        self.assertTrue(any(f.startswith("number not in card") for f in self.lint("We cut it by 70 percent.")))
        self.assertEqual(self.lint("We cut it by 70 percent.", claim="cut by 70 percent"), [])
        self.assertTrue(any(f.startswith("length") for f in self.lint("word " * 70)))

    def test_template_check(self):
        good = s6_template.fill(0, "three tiers", "pricing", 0.9)
        self.assertEqual(qa.template_check(good), [])
        generic = s6_template.fill(1, "day job", "career", 0.3)   # low area confidence -> generic phrase + site
        self.assertEqual(qa.template_check(generic), [])
        wrong_link = good.replace("/pricing", "/hiring")
        self.assertIn("template: pricing needs its own link", qa.template_check(wrong_link))
        self.assertIn("template: fixed wording changed", qa.template_check("Hi! Check out our guides."))
        bad_opener = good.replace("caught my eye", "is amazing")
        self.assertIn("template: opener not in the approved list", qa.template_check(bad_opener))


class TestVerdict(unittest.TestCase):
    def ans(self, **over):
        a = {k: {"type": "score", "score": 1.5} for k in ("claim", "receipt", "caught", "hook", "pace", "close")}
        for k in ("persona_misfit", "third_party", "risk_guru", "risk_mystical", "risk_generic", "risk_salesy",
                  "risk_offensive", "sensitive_post"):
            a[k] = {"type": "noul", "noul": 0.05}
        a.update(over)
        return a

    def test_swap_check_blocks_below_0_2_and_queues_below_0_5(self):
        sw = lambda p: {"choice": "B", "own": "A", "probabilities": {"A": p, "B": 1 - p}}  # noqa: E731
        self.assertEqual(qa.verdict(draft(), [], self.ans(), sw(0.1))["status"], "rewrite")
        self.assertEqual(qa.verdict(draft(), [], self.ans(), sw(0.3))["status"], "review")
        self.assertEqual(qa.verdict(draft(), [], self.ans(), sw(0.9))["status"], "pass")

    def test_rubric_never_blocks_only_queues(self):
        low = self.ans(**{k: {"type": "score", "score": 0.2} for k in ("claim", "receipt", "caught", "hook")})
        v = qa.verdict(draft(), [], low, None)
        self.assertEqual(v["status"], "review")
        self.assertEqual(v["hard"], [])

    def test_sensitive_post_blocks(self):
        v = qa.verdict(draft(), [], self.ans(sensitive_post={"type": "noul", "noul": 0.45}), None)
        self.assertEqual(v["status"], "rewrite")


if __name__ == "__main__":
    unittest.main()
