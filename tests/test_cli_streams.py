from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared.cli import configure_cli_streams  # noqa: E402


class CliStreamTests(unittest.TestCase):
    def test_reconfigures_stdout_and_stderr_to_utf8(self):
        stdout_buffer = io.BytesIO()
        stderr_buffer = io.BytesIO()
        stdout = io.TextIOWrapper(stdout_buffer, encoding="ascii")
        stderr = io.TextIOWrapper(stderr_buffer, encoding="ascii")

        with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
            configure_cli_streams()
            self.assertEqual(stdout.encoding.lower(), "utf-8")
            self.assertEqual(stderr.encoding.lower(), "utf-8")
            print("result → ✅")
            print("error → ✅", file=sys.stderr)
            stdout.flush()
            stderr.flush()

        self.assertEqual(stdout_buffer.getvalue().decode("utf-8").splitlines(), ["result → ✅"])
        self.assertEqual(stderr_buffer.getvalue().decode("utf-8").splitlines(), ["error → ✅"])

    def test_redirected_string_streams_are_left_usable(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
            configure_cli_streams()
            print("result → ✅")
            print("error → ✅", file=sys.stderr)

        self.assertEqual(stdout.getvalue(), "result → ✅\n")
        self.assertEqual(stderr.getvalue(), "error → ✅\n")


if __name__ == "__main__":
    unittest.main()
