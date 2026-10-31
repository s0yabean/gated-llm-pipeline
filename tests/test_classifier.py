import tempfile
import unittest
from pathlib import Path

from gated_pipeline.classifier import Cached, MockClassifier, choice, noul, score


class Counting(MockClassifier):
    calls = 0

    def ask(self, state, questions):
        Counting.calls += 1
        return super().ask(state, questions)


class TestCache(unittest.TestCase):
    def test_second_identical_request_is_served_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            clf = Cached(Counting(), Path(tmp))
            q = {"promo": noul("Is `post.text` an ad?")}
            s = {"post": {"text": "Use code SAVE for 20% off, link in bio"}}
            a1 = clf.ask(s, q)
            a2 = clf.ask(s, q)
            self.assertEqual(a1, a2)
            self.assertEqual((clf.live, clf.cached, Counting.calls), (1, 1, 1))
            clf.ask(s, {"promo": noul("Is `post.text` an advert?")})   # changed question -> new key
            self.assertEqual(clf.live, 2)


class TestMockShapes(unittest.TestCase):
    def test_answer_shapes_match_system_one(self):
        a = MockClassifier().ask({"post": {"text": "We shipped a pricing page and clients picked the middle tier."}}, {
            "promo": noul("ad?"),
            "theme": choice({"themes": {"pricing": "pricing tiers clients"}, "question": "?"}, {"pricing": None, "none": None}),
            "fit": score({"card": {"claim": "clients picked the middle tier"}, "question": "?"}, ["a", "b", "c", "d"]),
        })
        self.assertEqual(a["promo"]["type"], "noul")
        self.assertTrue(0 < a["promo"]["noul"] < 1)
        self.assertEqual(a["theme"]["choice"], "pricing")
        self.assertAlmostEqual(sum(a["theme"]["probabilities"].values()), 1.0, places=2)
        self.assertEqual(len(a["fit"]["probabilities"]), 4)
        self.assertGreater(a["fit"]["score"], 1.5)


if __name__ == "__main__":
    unittest.main()
