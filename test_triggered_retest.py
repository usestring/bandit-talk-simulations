import random
import unittest

from bandit_environments import Environment, Policy, simulate_with_policy


class TriggeredRetestTests(unittest.TestCase):
    def policy(self, window=100, threshold=0.70):
        return Policy("triggered_retest", 3, 2000, random.Random(7),
                      retest_window=window, retest_threshold=threshold)

    def commit(self, policy):
        for t in range(90):
            arm = policy.select(t)
            policy.update(arm, int(arm == 0))
        self.assertEqual(policy.select(90), 0)

    def test_clock_alone_never_retests(self):
        policy = self.policy()
        self.commit(policy)
        for t in range(90, 2000):
            self.assertEqual(policy.select(t), 0)
            policy.update(0, 1)
        self.assertEqual(policy.retests, 0)
        self.assertEqual(policy.test_requests, 90)

    def test_requires_full_window_of_committed_outcomes(self):
        policy = self.policy()
        self.commit(policy)
        for t in range(90, 189):
            self.assertEqual(policy.select(t), 0)
            policy.update(0, 0)
        self.assertEqual(policy.select(189), 0)
        self.assertEqual(policy.retests, 0)
        policy.update(0, 0)
        self.assertEqual(policy.select(190), 0)
        self.assertEqual(policy.retests, 1)
        self.assertIsNone(policy.committed)
        self.assertEqual(len(policy.recent_committed), 0)

    def test_threshold_is_strict_and_window_trails(self):
        policy = self.policy()
        self.commit(policy)
        for t in range(90, 190):
            policy.select(t)
            policy.update(0, int(t < 160))
        self.assertEqual(policy.select(190), 0)
        self.assertEqual(policy.retests, 0)
        policy.update(0, 0)
        policy.select(191)
        self.assertEqual(policy.retests, 1)

    def test_retest_uses_fresh_evidence_and_monitor(self):
        policy = self.policy(window=4)
        self.commit(policy)
        for t in range(90, 94):
            policy.select(t)
            policy.update(0, 0)
        for t in range(94, 184):
            arm = policy.select(t)
            policy.update(arm, int(arm == 1))
        self.assertEqual(policy.select(184), 1)
        self.assertEqual(policy.retests, 1)
        self.assertEqual(len(policy.recent_committed), 0)
        policy.update(1, 1)
        self.assertEqual(list(policy.recent_committed), [1])
        self.assertEqual(policy.test_requests, 180)

    def test_environment_change_time_is_independent_of_policy(self):
        env = Environment("shift", 2000, shift_at=637)
        self.assertEqual(env.rates(636), [0.90, 0.75, 0.60])
        self.assertEqual(env.rates(637), [0.40, 0.75, 0.60])
        _, policy, _, wins, cost = simulate_with_policy("triggered_retest", "shift", 2000, 7, 637)
        self.assertGreater(policy.retests, 0)
        self.assertEqual(cost, 2000)
        self.assertEqual(len(wins), 2000)

    def test_rejects_invalid_monitor_settings(self):
        for window, threshold in ((0, 0.7), (100, -0.1), (100, 1.1)):
            with self.assertRaises(ValueError):
                self.policy(window, threshold)


if __name__ == "__main__":
    unittest.main()
