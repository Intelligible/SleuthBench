"""Persistent Python subprocess runner for the run_python eval tool.

Runs LLM-authored Python code in a long-lived child process. State persists
across consecutive run() calls within one (qa_pair, model) evaluation session,
enabling multi-step analysis. The evaluation pipeline creates a fresh runner
for every independent session and attempts cleanup afterward, so state is never
intentionally shared between sessions. A csv_path change also restarts the child
process and loads a fresh DataFrame.

Communication uses one JSON
request/response per line over a REPL loop. Each run uses one effective
deadline—the earlier of the configured timeout and an optional caller
deadline—across CSV setup, its postcondition, and user code. Python output is
capped by character count and may receive a truncation marker. The runner uses
the repository's `.venv/bin/python3` when present and otherwise
`sys.executable`.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

SANDBOX_TIMEOUT = 30
SANDBOX_MAX_OUTPUT = 4_000

_VENV_PYTHON = Path(__file__).parent.parent / ".venv" / "bin" / "python3"


class SandboxError(RuntimeError):
    """Base class for sandbox process and protocol failures."""


class SandboxTimeoutError(SandboxError):
    """The sandbox did not return a result before its deadline."""


class SandboxProcessError(SandboxError):
    """The sandbox process is unavailable or exited unexpectedly."""


class SandboxTransportError(SandboxError):
    """Communication with the sandbox process failed."""


class SandboxProtocolError(SandboxError):
    """The sandbox returned a malformed protocol response."""


# The child process runs this REPL loop. A private, non-inheritable duplicate of
# stdout carries JSON protocol responses. Regular fd 1/2 are redirected to the
# null device, and the private descriptor is non-inheritable, so native writes
# and ordinary exec-spawned descendants cannot corrupt or block the protocol
# channel.
_REPL_SCRIPT = r"""
import io, json, os, sys, traceback

class _CappedStringIO(io.StringIO):
    def __init__(self, limit):
        super().__init__()
        self._limit = limit
        self._captured = 0

    def write(self, value):
        if not isinstance(value, str):
            raise TypeError("string argument expected")
        value_len = len(value)
        remaining = self._limit - self._captured
        if remaining > 0:
            chunk = value[:remaining]
            super().write(chunk)
            self._captured += len(chunk)
        return value_len

_protocol_fd = os.dup(sys.stdout.fileno())
os.set_inheritable(_protocol_fd, False)
_protocol = os.fdopen(
    _protocol_fd,
    "w",
    encoding="utf-8",
    errors="replace",
    newline="\n",
    buffering=1,
)

_null_fd = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(_null_fd, sys.stdout.fileno())
    os.dup2(_null_fd, sys.stderr.fileno())
finally:
    os.close(_null_fd)

_request_stream = sys.stdin
_json_loads = json.loads
_json_dumps = json.dumps

def _repl():
    while True:
        request_line = _request_stream.readline()
        if not request_line:
            break
        try:
            request = _json_loads(request_line)
            request_id = request["id"]
            code = request["code"]
            capture_limit = max(1, int(request["capture_limit"]))
        except Exception:
            continue

        stdout_capture = _CappedStringIO(capture_limit)
        stderr_capture = _CappedStringIO(capture_limit)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        raised = False
        try:
            exec(compile(code, "<model_code>", "exec"), _g)
        except BaseException:
            raised = True
            traceback.print_exc(file=stderr_capture)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        response = _json_dumps(
            {
                "id": request_id,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "exception": raised,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        _protocol.write(response + "\n")
        _protocol.flush()

_g = {}
_repl()
"""


class PythonSandbox:
    """Execute Python code in a persistent subprocess with df pre-loaded."""

    def __init__(self, timeout: int = SANDBOX_TIMEOUT, max_output: int = SANDBOX_MAX_OUTPUT) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self._python = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
        self._proc: subprocess.Popen | None = None
        self._current_csv: str | None = None
        self._proc_lock = threading.RLock()

    def ping(self) -> bool:
        """Return True if the configured Python interpreter can be launched."""
        try:
            result = subprocess.run(
                [self._python, "-c", "print('ok')"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0 and result.stdout.strip() == "ok"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _start(self, csv_path: str, *, deadline: float | None = None) -> None:
        """Start (or restart) the persistent subprocess with df loaded."""
        if deadline is None:
            deadline = time.monotonic() + max(0.0, float(self.timeout))
        with self._proc_lock:
            self._kill()
            if self._proc is not None:
                raise RuntimeError("Could not terminate previous sandbox process")
            self._proc = subprocess.Popen(
                # Force the child interpreter itself to use UTF-8 too. The parent
                # pipe encoding below controls stdin/stdout decoding on this side,
                # but without -X utf8 the Windows child can still use cp1252 and
                # crash when model code prints characters such as → or ≈.
                [self._python, "-X", "utf8", "-c", _REPL_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Python-level stderr is captured by the REPL. Native fd 2 is
                # discarded so C extensions and spawned processes cannot fill an
                # unread pipe and deadlock the sandbox.
                stderr=subprocess.DEVNULL,
                text=True,
                # Force UTF-8 on the pipes. Without this the pipes default to the
                # platform locale encoding (cp1252 on Windows), so model-authored
                # code containing characters like → or ≈ raises UnicodeEncodeError
                # ('charmap' codec) on stdin.write and the whole eval entry is lost.
                encoding="utf-8",
                errors="replace",
            )
            self._current_csv = csv_path
        # Pre-load df and common imports; force non-interactive matplotlib backend.
        # matplotlib is an optional dep (notebook extra) — guard the import so a
        # missing install doesn't abort the setup block and leave df undefined.
        setup = (
            "try:\n    import matplotlib\n    matplotlib.use('Agg')\nexcept Exception:\n    pass\n"
            "import pandas as pd\nimport numpy as np\nimport scipy\n"
            f"df = pd.read_csv({repr(csv_path)})\n"
        )
        try:
            _, setup_err, _ = self._exec_raw(setup, deadline=deadline)
            _, postcondition_err, postcondition_failed = self._exec_raw(
                "assert 'df' in dir() and df is not None",
                deadline=deadline,
            )
        except Exception:
            # Do not retain a process whose setup sequence did not complete.
            self._kill()
            raise
        if postcondition_failed:
            # Stderr from setup may contain a benign warning, so it is not the
            # success criterion. The explicit df postcondition is the gate. If
            # that fails, don't retain a half-initialized process and prefer the
            # setup traceback because it identifies the import/read root cause.
            self._kill()
            detail = setup_err.strip() or postcondition_err.strip()
            raise RuntimeError(f"sandbox setup failed (df not loaded):\n{detail}")

    def _kill(self, expected_proc: subprocess.Popen | None = None) -> None:
        """Attempt cleanup without discarding a process that may still be live.

        ``expected_proc`` prevents a stale timeout callback from killing a
        replacement process. Windows first attempts process-tree cleanup with
        taskkill; POSIX direct kill reaches only the immediate child. If exit
        cannot be confirmed, ``self._proc`` is deliberately retained so a later
        cleanup attempt can still reach it.
        """
        with self._proc_lock:
            proc = self._proc
            if proc is None:
                return
            if expected_proc is not None and proc is not expected_proc:
                return

            self._current_csv = None
            terminated = proc.poll() is not None

            if not terminated and sys.platform == "win32":
                # Prefer taskkill so ordinary spawned descendants are terminated
                # with the REPL parent. If it fails or times out, direct-kill the
                # parent below rather than silently losing the live process.
                try:
                    result = subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        check=False,
                    )
                    if result.returncode == 0:
                        try:
                            proc.wait(timeout=5)
                            terminated = True
                        except Exception:
                            terminated = proc.poll() is not None
                except Exception:
                    pass

            if not terminated:
                # On POSIX this kills only the immediate child. On Windows it is
                # the fallback when taskkill cannot tear down the process tree.
                # Containerization should eventually provide uniform tree cleanup.
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                    terminated = True
                except Exception:
                    terminated = proc.poll() is not None

            if terminated and self._proc is proc:
                for stream_name in ("stdin", "stdout", "stderr"):
                    stream = getattr(proc, stream_name, None)
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                self._proc = None

    def close(self) -> None:
        """Attempt to terminate the child and discard state after confirmed exit."""
        self._kill()

    def _exec_raw(
        self,
        code: str,
        *,
        deadline: float | None = None,
    ) -> tuple[str, str, bool]:
        """Return ``(stdout, stderr, raised_exception)`` before the deadline."""
        if deadline is None:
            deadline = time.monotonic() + max(0.0, float(self.timeout))
        with self._proc_lock:
            proc = self._proc
            if proc is None:
                raise SandboxProcessError("Sandbox process not running")
            if proc.poll() is not None:
                self._kill(expected_proc=proc)
                raise SandboxProcessError("Sandbox process not running")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxTimeoutError(f"Sandbox timed out after {self.timeout}s")

        request_id = uuid.uuid4().hex[:12]
        payload = json.dumps(
            {
                "id": request_id,
                "code": code,
                "capture_limit": max(1, self.max_output + 1),
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ) + "\n"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxTimeoutError(f"Sandbox timed out after {self.timeout}s")
        timed_out = threading.Event()

        def _kill_on_timeout() -> None:
            timed_out.set()
            self._kill(expected_proc=proc)

        timer = threading.Timer(remaining, _kill_on_timeout)
        timer.start()

        def _stop_timer() -> None:
            # cancel() alone cannot stop a callback that has already started.
            # Joining prevents it from killing this persistent process after a
            # successful response has been accepted for the next request.
            timer.cancel()
            timer.join()

        completed_at: float | None = None
        failure: Exception | None = None
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
            for line in proc.stdout:
                try:
                    response = json.loads(line)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise SandboxProtocolError("Malformed response from sandbox process") from exc
                if not isinstance(response, dict) or response.get("id") != request_id:
                    raise SandboxProtocolError("Mismatched response from sandbox process")
                out = response.get("stdout")
                err = response.get("stderr")
                raised = response.get("exception")
                if not isinstance(out, str) or not isinstance(err, str) or not isinstance(raised, bool):
                    raise SandboxProtocolError("Malformed response from sandbox process")
                completed_at = time.monotonic()
                break
            else:
                raise SandboxProcessError("Sandbox process died unexpectedly")
        except Exception as exc:
            failure = exc
            completed_at = time.monotonic()
        finally:
            _stop_timer()

        expired = (
            timed_out.is_set()
            or completed_at is None
            or completed_at >= deadline
        )
        if expired or failure is not None:
            self._kill(expected_proc=proc)
        if expired:
            timeout_error = SandboxTimeoutError(
                f"Sandbox timed out after {self.timeout}s"
            )
            if failure is not None:
                raise timeout_error from failure
            raise timeout_error
        if failure is not None:
            if isinstance(failure, SandboxError):
                raise failure
            raise SandboxTransportError(
                "Error communicating with sandbox process"
            ) from failure

        if out.endswith("\n"):
            out = out[:-1]
            if out.endswith("\r"):
                out = out[:-1]
        if err.endswith("\n"):
            err = err[:-1]
            if err.endswith("\r"):
                err = err[:-1]
        return out, err, raised

    def run(
        self,
        code: str,
        csv_path: Path | str,
        *,
        deadline: float | None = None,
    ) -> str:
        """Run with df loaded, restarting the child if it exited or CSV changed."""
        if not isinstance(code, str):
            return "ERROR: Python code must be a string"

        started_at = time.monotonic()
        local_deadline = started_at + max(0.0, float(self.timeout))
        effective_deadline = (
            min(local_deadline, deadline)
            if deadline is not None
            else local_deadline
        )
        effective_timeout = max(0.0, effective_deadline - started_at)
        timeout_result = f"ERROR: timed out after {effective_timeout:g}s"
        if effective_timeout <= 0:
            return timeout_result
        csv_str = str(csv_path)

        # Restart if csv changed or process is dead
        if self._current_csv != csv_str or self._proc is None or self._proc.poll() is not None:
            try:
                self._start(csv_str, deadline=effective_deadline)
            except SandboxTimeoutError:
                return timeout_result
            except RuntimeError as exc:
                return f"ERROR: {exc}"

        # Auto-print trailing expression, REPL-style
        exec_code = code
        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.unparse(tree.body[-1].value)
                rest = tree.body[:-1]
                if rest:
                    rest_code = ast.unparse(ast.Module(body=rest, type_ignores=[]))
                    exec_code = rest_code + f"\n_result = ({last_expr})\nif _result is not None: print(_result)\n"
                else:
                    exec_code = f"_result = ({last_expr})\nif _result is not None: print(_result)\n"
        except SyntaxError:
            pass  # fall through with original code

        try:
            out, err, raised = self._exec_raw(
                exec_code,
                deadline=effective_deadline,
            )
        except SandboxTimeoutError:
            return timeout_result
        except SandboxError as exc:
            return f"ERROR: {exc}"

        err = err.strip()
        if raised:
            detail = err or "Sandbox code raised an exception"
            if len(detail) > self.max_output:
                detail = detail[: self.max_output] + "\n[truncated]"
            return f"ERROR:\n{detail}"

        if err:
            out = f"{out}\n\n[stderr]\n{err}" if out else f"[stderr]\n{err}"

        if len(out) > self.max_output:
            out = out[: self.max_output] + "\n[output truncated]"
        return out or "(no output)"
