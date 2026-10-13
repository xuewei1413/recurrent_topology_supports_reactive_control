"""Checks for explicit missing-statistic handling and unchanged GA settings."""
import unittest
from unittest.mock import patch

from local_replication.analyze_association import json_finite, partial_or_undefined
from local_replication.evolve_association import COHORTS, make_ga


class SummaryTests(unittest.TestCase):
    def test_nonfinite_does_not_remove_networks(self):
        data = {"networks": [{"seed": 4230, "value": float("nan")},
                             {"seed": 4231, "value": 0.0}], "p": float("inf")}
        actual = json_finite(data)
        self.assertEqual(len(actual["networks"]), 2)
        self.assertIsNone(actual["networks"][0]["value"])
        self.assertEqual(actual["networks"][1]["value"], 0.0)
        self.assertIsNone(actual["p"])

    def test_degenerate_partial_is_explicit(self):
        with patch("local_replication.analyze_association.partial", side_effect=ZeroDivisionError):
            result = partial_or_undefined([], ["J0"])
        self.assertIsNone(result["r"])
        self.assertIn("no network omitted", result["status"])

    def test_original_cohorts_and_budget(self):
        self.assertEqual(len(COHORTS["v3"]["seeds"]), 25)
        self.assertEqual(len(COHORTS["c2"]["seeds"]), 30)
        ga = make_ga("c2", 4260, 20000)
        self.assertEqual(ga.config.pop_size, 30)
        self.assertEqual(ga.config.n_elite, 6)
        self.assertEqual(ga.config.n_eval_episodes, 4)
        self.assertEqual(ga.config.total_eval_budget, 20000)
        self.assertEqual(ga.config.selection_seeds, tuple(range(89000, 89010)))


if __name__ == "__main__":
    unittest.main()
