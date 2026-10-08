import unittest

from measure_convergence import confirmation_request, censored_quantile, summarize


class ConvergenceTests(unittest.TestCase):
    def test_confirmation_requires_full_window_and_hold(self):
        self.assertIsNone(confirmation_request([0.0] * 198))
        self.assertEqual(confirmation_request([0.0] * 199), 199)

    def test_bad_prefix_does_not_count_as_converged(self):
        self.assertEqual(confirmation_request([1.0] * 100 + [0.0] * 199), 294)
        self.assertIsNone(confirmation_request([0.06] * 2000))

    def test_missing_runs_censor_population_quantiles(self):
        self.assertEqual(censored_quantile([1, 2, None, None], 0.5), 2)
        self.assertIsNone(censored_quantile([1, 2, None, None], 0.9))
        self.assertIsNone(censored_quantile([None] * 50, 0.5))

    def test_time_conversion_retains_censored_values(self):
        values = summarize([412] * 50, 1000)
        self.assertEqual(values["median_seconds"], 2060)
        self.assertEqual(values["p90_seconds"], 2060)
        self.assertEqual(values["budget_seconds"], 5000)
        self.assertIsNone(summarize([None] * 50, 1000)["median_seconds"])


if __name__ == "__main__":
    unittest.main()
