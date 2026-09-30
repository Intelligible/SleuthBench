from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, call, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import sandbox as sandbox_module  # noqa: E402


class SandboxStartupTests(unittest.TestCase):
    def test_setup_stderr_is_ignored_when_df_postcondition_passes(self):
        sandbox = sandbox_module.PythonSandbox()

        with ExitStack() as stack:
            stack.enter_context(patch.object(sandbox, "_kill"))
            exec_raw = stack.enter_context(
                patch.object(
                    sandbox,
                    "_exec_raw",
                    side_effect=[
                        ("", "UserWarning: harmless version mismatch", False),
                        ("", "", False),
                    ],
                )
            )
            stack.enter_context(
                patch.object(sandbox_module.subprocess, "Popen", return_value=object())
            )
            with patch.object(
                sandbox_module.time,
                "monotonic",
                return_value=100.0,
            ):
                sandbox._start("data.csv")

        self.assertEqual(sandbox._current_csv, "data.csv")
        self.assertEqual(exec_raw.call_count, 2)
        self.assertEqual(
            exec_raw.call_args_list[1],
            call(
                "assert 'df' in dir() and df is not None",
                deadline=130.0,
            ),
        )
        self.assertEqual(
            {item.kwargs["deadline"] for item in exec_raw.call_args_list},
            {130.0},
        )

    def test_failed_postcondition_reports_original_setup_error(self):
        sandbox = sandbox_module.PythonSandbox()
        setup_error = "Traceback:\nFileNotFoundError: missing.csv"
        assertion_error = "Traceback:\nAssertionError"

        with ExitStack() as stack:
            kill = stack.enter_context(patch.object(sandbox, "_kill"))
            stack.enter_context(
                patch.object(
                    sandbox,
                    "_exec_raw",
                    side_effect=[
                        ("", setup_error, True),
                        ("", assertion_error, True),
                    ],
                )
            )
            stack.enter_context(
                patch.object(sandbox_module.subprocess, "Popen", return_value=object())
            )
            with self.assertRaises(RuntimeError) as raised:
                sandbox._start("missing.csv")

        self.assertIn(setup_error, str(raised.exception))
        self.assertNotIn(assertion_error, str(raised.exception))
        self.assertEqual(kill.call_count, 2)

    def test_first_run_shares_one_deadline_across_setup_and_user_code(self):
        sandbox = sandbox_module.PythonSandbox(timeout=30)
        proc = Mock()
        proc.poll.return_value = None

        with ExitStack() as stack:
            stack.enter_context(patch.object(sandbox, "_kill"))
            exec_raw = stack.enter_context(
                patch.object(
                    sandbox,
                    "_exec_raw",
                    side_effect=[
                        ("", "", False),
                        ("", "", False),
                        ("1", "", False),
                    ],
                )
            )
            stack.enter_context(
                patch.object(sandbox_module.subprocess, "Popen", return_value=proc)
            )
            stack.enter_context(
                patch.object(sandbox_module.time, "monotonic", return_value=100.0)
            )
            result = sandbox.run("print(1)", "data.csv")

        self.assertEqual(result, "1")
        self.assertEqual(exec_raw.call_count, 3)
        self.assertEqual(
            {item.kwargs["deadline"] for item in exec_raw.call_args_list},
            {130.0},
        )


class SandboxProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = REPO_ROOT / "data" / "standardized" / "bike_sharing_100.csv"
        cls.sandbox = sandbox_module.PythonSandbox(timeout=10, max_output=64)

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.close()

    def test_no_newline_stdout_and_stderr_preserve_framing(self):
        self.assertEqual(
            self.sandbox.run("print('hello', end='')", self.csv_path),
            "hello",
        )
        self.assertEqual(
            self.sandbox.run("import sys\n_ = sys.stderr.write('oops')", self.csv_path),
            "[stderr]\noops",
        )

    def test_native_output_cannot_block_or_corrupt_protocol(self):
        code = (
            "import os\n"
            "_ = os.write(2, b'x' * 1_000_000)\n"
            "_ = os.write(1, b'noise')\n"
            "print('captured')"
        )
        self.assertEqual(self.sandbox.run(code, self.csv_path), "captured")

    def test_old_sentinel_text_is_ordinary_code_and_output(self):
        code = "__END__value = 7\nprint(f'__OUT__ __ERR__ __DONE__ {__END__value}', end='')"
        self.assertEqual(
            self.sandbox.run(code, self.csv_path),
            "__OUT__ __ERR__ __DONE__ 7",
        )

    def test_timeout_kills_process_and_next_call_restarts(self):
        original_timeout = self.sandbox.timeout
        self.sandbox.timeout = 1
        try:
            self.assertEqual(
                self.sandbox.run("while True:\n    pass", self.csv_path),
                "ERROR: timed out after 1s",
            )
        finally:
            self.sandbox.timeout = original_timeout

        self.assertEqual(self.sandbox.run("print(42)", self.csv_path), "42")

    def test_large_output_is_capped_before_protocol_response(self):
        result = self.sandbox.run("print('x' * 1_000_000, end='')", self.csv_path)
        self.assertEqual(result, ("x" * 64) + "\n[output truncated]")

    def test_warning_is_reported_without_marking_execution_failed(self):
        original_max_output = self.sandbox.max_output
        self.sandbox.max_output = 4_000
        try:
            result = self.sandbox.run(
                "import warnings\nwarnings.warn('benign')\nprint(42)",
                self.csv_path,
            )
        finally:
            self.sandbox.max_output = original_max_output

        self.assertTrue(result.startswith("42\n\n[stderr]\n"))
        self.assertIn("UserWarning: benign", result)
        self.assertFalse(result.startswith("ERROR:"))

    def test_system_exit_is_an_error_without_killing_repl(self):
        original_max_output = self.sandbox.max_output
        self.sandbox.max_output = 4_000
        try:
            result = self.sandbox.run(
                "survived_value = 41\nraise SystemExit('bye')",
                self.csv_path,
            )
            follow_up = self.sandbox.run("survived_value + 1", self.csv_path)
        finally:
            self.sandbox.max_output = original_max_output

        self.assertTrue(result.startswith("ERROR:\n"))
        self.assertIn("SystemExit: bye", result)
        self.assertEqual(follow_up, "42")

    def test_keyboard_interrupt_is_an_error_without_killing_repl(self):
        original_max_output = self.sandbox.max_output
        self.sandbox.max_output = 4_000
        try:
            result = self.sandbox.run(
                "interrupt_value = 8\nraise KeyboardInterrupt()",
                self.csv_path,
            )
            follow_up = self.sandbox.run("interrupt_value + 1", self.csv_path)
        finally:
            self.sandbox.max_output = original_max_output

        self.assertTrue(result.startswith("ERROR:\n"))
        self.assertIn("KeyboardInterrupt", result)
        self.assertEqual(follow_up, "9")

    def test_regular_exception_is_still_reported_as_error(self):
        original_max_output = self.sandbox.max_output
        self.sandbox.max_output = 4_000
        try:
            result = self.sandbox.run("1 / 0", self.csv_path)
        finally:
            self.sandbox.max_output = original_max_output

        self.assertTrue(result.startswith("ERROR:\n"))
        self.assertIn("ZeroDivisionError", result)

    def test_hard_process_exit_is_cleaned_up_before_restart(self):
        result = self.sandbox.run("import os\nos._exit(3)", self.csv_path)

        self.assertEqual(result, "ERROR: Sandbox process died unexpectedly")
        self.assertEqual(self.sandbox.run("print(42)", self.csv_path), "42")


class SandboxKillTests(unittest.TestCase):
    @staticmethod
    def _live_process():
        proc = Mock()
        proc.pid = 123
        proc.poll.return_value = None
        return proc

    def test_taskkill_timeout_falls_back_to_direct_kill(self):
        sandbox = sandbox_module.PythonSandbox()
        proc = self._live_process()
        proc.wait.return_value = 0
        sandbox._proc = proc
        sandbox._current_csv = "table.csv"

        with patch.object(sandbox_module.sys, "platform", "win32"), patch.object(
            sandbox_module.subprocess,
            "run",
            side_effect=sandbox_module.subprocess.TimeoutExpired("taskkill", 5),
        ):
            sandbox._kill()

        proc.kill.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(sandbox._proc)
        self.assertIsNone(sandbox._current_csv)

    def test_taskkill_nonzero_exit_falls_back_to_direct_kill(self):
        sandbox = sandbox_module.PythonSandbox()
        proc = self._live_process()
        proc.wait.return_value = 0
        sandbox._proc = proc

        with patch.object(sandbox_module.sys, "platform", "win32"), patch.object(
            sandbox_module.subprocess,
            "run",
            return_value=Mock(returncode=1),
        ):
            sandbox._kill()

        proc.kill.assert_called_once_with()
        self.assertIsNone(sandbox._proc)

    def test_taskkill_wait_timeout_also_falls_back_to_direct_kill(self):
        sandbox = sandbox_module.PythonSandbox()
        proc = self._live_process()
        proc.wait.side_effect = [
            sandbox_module.subprocess.TimeoutExpired("wait", 5),
            0,
        ]
        sandbox._proc = proc

        with patch.object(sandbox_module.sys, "platform", "win32"), patch.object(
            sandbox_module.subprocess,
            "run",
            return_value=Mock(returncode=0),
        ):
            sandbox._kill()

        proc.kill.assert_called_once_with()
        self.assertEqual(proc.wait.call_count, 2)
        self.assertIsNone(sandbox._proc)

    def test_failed_fallback_retains_process_for_later_cleanup(self):
        sandbox = sandbox_module.PythonSandbox()
        proc = self._live_process()
        proc.kill.side_effect = OSError("access denied")
        proc.wait.side_effect = sandbox_module.subprocess.TimeoutExpired("wait", 5)
        sandbox._proc = proc
        sandbox._current_csv = "table.csv"

        with patch.object(sandbox_module.sys, "platform", "win32"), patch.object(
            sandbox_module.subprocess,
            "run",
            side_effect=sandbox_module.subprocess.TimeoutExpired("taskkill", 5),
        ):
            sandbox._kill()

        self.assertIs(sandbox._proc, proc)
        self.assertIsNone(sandbox._current_csv)
        proc.stdin.close.assert_not_called()
        proc.stdout.close.assert_not_called()

    def test_stale_timeout_cannot_kill_replacement_process(self):
        sandbox = sandbox_module.PythonSandbox()
        old_proc = self._live_process()
        new_proc = self._live_process()
        sandbox._proc = new_proc
        sandbox._current_csv = "new.csv"

        sandbox._kill(expected_proc=old_proc)

        self.assertIs(sandbox._proc, new_proc)
        self.assertEqual(sandbox._current_csv, "new.csv")
        new_proc.kill.assert_not_called()


class SandboxTransportTests(unittest.TestCase):
    def test_expired_deadline_does_not_write_or_start_timer(self):
        sandbox = sandbox_module.PythonSandbox(timeout=30)
        proc = Mock()
        proc.poll.return_value = None
        sandbox._proc = proc

        with patch.object(
            sandbox_module.time,
            "monotonic",
            return_value=10.0,
        ), patch.object(sandbox_module.threading, "Timer") as timer:
            with self.assertRaises(sandbox_module.SandboxTimeoutError):
                sandbox._exec_raw("print(1)", deadline=9.0)

        proc.stdin.write.assert_not_called()
        timer.assert_not_called()

    def test_payload_serialization_cannot_extend_deadline(self):
        sandbox = sandbox_module.PythonSandbox(timeout=30)
        proc = Mock()
        proc.poll.return_value = None
        sandbox._proc = proc

        with patch.object(
            sandbox_module.time,
            "monotonic",
            side_effect=[9.0, 11.0],
        ), patch.object(
            sandbox_module.json,
            "dumps",
            return_value="{}",
        ), patch.object(sandbox_module.threading, "Timer") as timer:
            with self.assertRaises(sandbox_module.SandboxTimeoutError):
                sandbox._exec_raw("print(1)", deadline=10.0)

        proc.stdin.write.assert_not_called()
        timer.assert_not_called()

    def test_response_after_deadline_is_timeout_even_if_timer_is_delayed(self):
        sandbox = sandbox_module.PythonSandbox(timeout=30)
        proc = Mock()
        proc.poll.return_value = None
        proc.stdout = iter(
            [
                (
                    '{"id":"abcdefghijkl","stdout":"","stderr":"",'
                    '"exception":false}\n'
                )
            ]
        )
        sandbox._proc = proc
        fake_uuid = Mock()
        fake_uuid.hex = "abcdefghijkl9999"

        with patch.object(
            sandbox_module.time,
            "monotonic",
            side_effect=[0.0, 0.0, 2.0],
        ), patch.object(
            sandbox_module.uuid,
            "uuid4",
            return_value=fake_uuid,
        ), patch.object(
            sandbox_module.threading,
            "Timer",
        ) as timer, patch.object(sandbox, "_kill") as kill:
            with self.assertRaises(sandbox_module.SandboxTimeoutError):
                sandbox._exec_raw("print(1)", deadline=1.0)

        timer.assert_called_once()
        timer.return_value.cancel.assert_called()
        timer.return_value.join.assert_called()
        kill.assert_called_once_with(expected_proc=proc)

    def test_pipe_write_failure_is_cleaned_up_and_typed(self):
        sandbox = sandbox_module.PythonSandbox(timeout=1)
        proc = Mock()
        proc.poll.return_value = None
        proc.stdin.write.side_effect = OSError("broken pipe")
        sandbox._proc = proc

        with patch.object(sandbox, "_kill") as kill:
            with self.assertRaises(sandbox_module.SandboxTransportError):
                sandbox._exec_raw("print(1)")

        kill.assert_called_once_with(expected_proc=proc)

    def test_malformed_response_is_cleaned_up_and_typed(self):
        sandbox = sandbox_module.PythonSandbox(timeout=1)
        proc = Mock()
        proc.poll.return_value = None
        proc.stdout = iter(["not-json\n"])
        sandbox._proc = proc

        with patch.object(sandbox, "_kill") as kill:
            with self.assertRaises(sandbox_module.SandboxProtocolError):
                sandbox._exec_raw("print(1)")

        kill.assert_called_once_with(expected_proc=proc)

    def test_transport_error_is_not_reported_as_timeout(self):
        sandbox = sandbox_module.PythonSandbox(timeout=7)
        proc = Mock()
        proc.poll.return_value = None
        sandbox._proc = proc
        sandbox._current_csv = "table.csv"

        with patch.object(
            sandbox,
            "_exec_raw",
            side_effect=sandbox_module.SandboxTransportError("pipe failed"),
        ):
            result = sandbox.run("print(1)", "table.csv")

        self.assertEqual(result, "ERROR: pipe failed")


if __name__ == "__main__":
    unittest.main()
