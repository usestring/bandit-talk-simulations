import unittest

import blocked_control as control


class BlockedControlTests(unittest.TestCase):
    def test_control_remains_impossible_after_repeated_use_and_rest(self):
        env = control.ReactiveControl(400)
        for t in range(400):
            self.assertEqual(env.rates(t)[3], 0.0)
            env.observe(3 if t < 200 else 0)

    def test_round_robin_wastes_one_quarter_on_control(self):
        row = control.simulate("round_robin", 400, 12, "blocked_unknown")
        self.assertEqual(row["blocked_share"], 0.25)

    def test_known_blocked_pruning_preserves_original_choices_and_outcomes(self):
        for name in control.POLICIES:
            with self.subTest(name=name):
                original = control.simulate(name, 400, 12, "three_arms")
                pruned = control.simulate(name, 400, 12, "blocked_pruned")
                self.assertEqual(original, pruned)
                self.assertEqual(pruned["blocked_share"], 0.0)
