from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline as ep  # noqa: E402


class EvalToolDispatchTests(unittest.TestCase):
    def setUp(self):
        self.csv_path = Path("table.csv")

    def test_unknown_tool_is_rejected_without_running_python(self):
        sandbox = Mock()
        sandbox.run.side_effect = AssertionError("must not be called")
        tool_log = []

        result = ep._run_eval_tool(
            "shell",
            {"code": "print('unsafe')"},
            sandbox,
            self.csv_path,
            tool_log,
            "inspect",
        )

        self.assertEqual(result, "[unknown tool: shell]")
        sandbox.run.assert_not_called()
        self.assertEqual(
            tool_log,
            [
                {
                    "tool": "shell",
                    "input": {"code": "print('unsafe')"},
                    "step_thinking": "inspect",
                }
            ],
        )

    def test_run_python_without_sandbox_returns_stable_error(self):
        tool_log = []

        result = ep._run_eval_tool(
            "run_python",
            {"code": "print(1)"},
            None,
            self.csv_path,
            tool_log,
        )

        self.assertEqual(result, "[run_python unavailable: sandbox is not initialized]")
        self.assertEqual(tool_log[0]["tool"], "run_python")

    def test_run_python_rejects_missing_or_non_string_code(self):
        for tool_input in (None, {}, {"code": None}, {"code": 1}, {"code": []}):
            with self.subTest(tool_input=tool_input):
                sandbox = Mock()
                tool_log = []

                result = ep._run_eval_tool(
                    "run_python",
                    tool_input,
                    sandbox,
                    self.csv_path,
                    tool_log,
                )

                self.assertEqual(result, "[run_python failed: code must be a string]")
                sandbox.run.assert_not_called()

    def test_valid_run_python_call_is_forwarded_unchanged(self):
        sandbox = Mock()
        sandbox.run.return_value = "42"
        tool_log = []

        result = ep._run_eval_tool(
            "run_python",
            {"code": "print(42)"},
            sandbox,
            self.csv_path,
            tool_log,
        )

        self.assertEqual(result, "42")
        sandbox.run.assert_called_once_with("print(42)", self.csv_path)


if __name__ == "__main__":
    unittest.main()
