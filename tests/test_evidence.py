import collections
import unittest

from gated_pipeline.steps.s4_evidence import pick


def card(tag, cap=""):
    return {"tag": tag, "claim": tag, "cap_uses": cap}


class TestPick(unittest.TestCase):
    def test_best_card_above_reuse_line(self):
        chosen, _ = pick([(2.6, card("a")), (2.1, card("b"))], collections.Counter(), collections.Counter(), 1)
        self.assertEqual(chosen[1]["tag"], "a")

    def test_below_reuse_line_is_a_gap(self):
        chosen, _ = pick([(1.9, card("a"))], collections.Counter(), collections.Counter(), 1)
        self.assertIsNone(chosen)

    def test_lifetime_and_batch_caps_skip_to_next_card(self):
        uses = collections.Counter({"a": 3})
        chosen, skips = pick([(2.9, card("a", "3")), (2.5, card("b")), (2.4, card("c"))], uses,
                             collections.Counter({"b": 1}), 1)
        self.assertEqual(chosen[1]["tag"], "c")
        self.assertEqual(skips, ["a", "b (batch cap)"])


if __name__ == "__main__":
    unittest.main()
