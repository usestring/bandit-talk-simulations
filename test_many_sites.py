import contextlib
import io
import math
import tempfile
import unittest
from unittest.mock import patch

import many_sites


class ManySitesTests(unittest.TestCase):
    def test_gif_handles_zero_success_oracle_windows(self):
        with tempfile.TemporaryDirectory() as output:
            args = ["many_sites.py", "--requests", "100", "--seeds", "1",
                    "--window", "1", "--gif", "--output-dir", output]
            with patch("sys.argv", args), patch.object(many_sites.be, "animate_lines") as animate:
                with contextlib.redirect_stdout(io.StringIO()):
                    many_sites.main()
            animate.assert_called_once()
            panels = animate.call_args.args[1]
            successes = panels[0]["ref"][1]
            costs = panels[1]["ref"][1]
            self.assertIn(0, successes)
            self.assertEqual(len(costs), 100)
            self.assertTrue(all(math.isfinite(c) and c >= 0 for c in costs))


if __name__ == "__main__":
    unittest.main()
