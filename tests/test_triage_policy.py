import unittest

from gated_pipeline.steps import s2_triage as t
from tests.helpers import POST, answers, link


def decide(a, links, post=POST):
    n, hard, review = t.filter_stage(post, a)
    return t.decide(post, a, n, hard, review, links)


class TestTriagePolicy(unittest.TestCase):
    def test_clean_post_with_strong_link_passes(self):
        self.assertEqual(decide(answers(), link(0.8))["status"], "pass")

    def test_unsafe_always_drops_and_never_escalates(self):
        r = decide(answers(unsafe=0.6), link(0.95))
        self.assertEqual(r["status"], "drop")
        self.assertEqual(r["escalate_reasons"], [])

    def test_sensitive_drops_at_low_line_and_llm_cannot_rescue(self):
        r = decide(answers(sensitive=0.42), link(0.95), POST | {"llm_relevance": 18})
        self.assertEqual(r["status"], "drop")

    def test_filter_just_over_line_is_soft_and_escalates(self):
        r = decide(answers(promo=0.72), link(0.5))
        self.assertEqual(r["status"], "escalate")

    def test_filter_well_over_line_drops(self):
        self.assertEqual(decide(answers(promo=0.8), link(0.5))["status"], "drop")

    def test_strong_link_softens_promo_up_to_0_9(self):
        self.assertEqual(decide(answers(promo=0.85), link(0.9))["status"], "escalate")
        self.assertEqual(decide(answers(promo=0.92), link(0.9))["status"], "drop")

    def test_weak_link_drops_borderline_link_escalates(self):
        self.assertIn("no_belief_link", decide(answers(), link(0.3))["drop_reasons"])
        self.assertIn("belief_link_borderline", decide(answers(), link(0.55))["escalate_reasons"])

    def test_theme_only_link_needs_0_9(self):
        r = decide(answers(), link(0.8, "same_theme"))
        self.assertEqual(r["status"], "escalate")
        self.assertIn("theme_only_link", r["escalate_reasons"])
        self.assertEqual(decide(answers(), link(0.92, "same_theme"))["status"], "pass")

    def test_llm_settles_escalated_rows(self):
        a, lk = answers(promo=0.72), link(0.5)
        self.assertEqual(decide(a, lk, POST | {"llm_relevance": 12})["status"], "pass")
        self.assertEqual(decide(a, lk, POST | {"llm_relevance": 6})["status"], "drop")

    def test_classifier_llm_disagreement_goes_to_human_review(self):
        r = decide(answers(), link(0.8), POST | {"llm_relevance": 3})
        self.assertEqual(r["status"], "review")
        self.assertIn("classifier_llm_disagree", r["escalate_reasons"])

    def test_routing(self):
        a = answers(two_paths=0.9, in_motion=0.9,
                    life_area={"type": "choice", "choice": "career", "probabilities": {"career": 0.8}})
        self.assertEqual(decide(a, link(0.8))["route"], "direct_help")
        self.assertEqual(decide(answers(founder=0.8), link(0.8))["route"], "founder_intro")
        self.assertEqual(decide(answers(), link(0.8))["route"], "evidence_reply")

    def test_code_spam_signals(self):
        self.assertIn("cashtag", t.code_spam_signals("buy $ABC now, real long text for the length check here"))
        self.assertIn("contract_address", t.code_spam_signals("0x" + "a" * 40))
        self.assertEqual(t.code_spam_signals("A normal post about pricing that is long enough to count."), [])


if __name__ == "__main__":
    unittest.main()
