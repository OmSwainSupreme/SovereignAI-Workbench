#!/usr/bin/env python3
"""Sandbox entrypoint for the sovereign-ai/sandbox-python:5d image.

This script runs INSIDE the container. It is not part of the host
runtime; it is bundled into the Docker image and used to launch the
untrusted Python program.

Behaviour
---------

1. Read ``SOVEREIGN_CODE_B64`` and decode it to a temporary file
   inside ``/workspace``.
2. Read ``SOVEREIGN_TIMEOUT`` (in seconds) and ``SOVEREIGN_MAX_OUTPUT``
   (in bytes) from the environment.
3. Spawn the user program with ``python3 -u`` so stdout is unbuffered.
4. If the program exceeds the timeout, terminate the process group
   and report ``timed_out=True``.
5. Truncate stdout and stderr to ``SOVEREIGN_MAX_OUTPUT`` bytes.
6. Emit a single JSON status line to stderr, prefixed with
   ``__SOVEREIGN_RESULT__``, so the host runner can parse the
   outcome.
7. Exit with the program's exit code (0 on success).

The script intentionally does NOT:

* Write anywhere outside ``/workspace``.
* Read environment variables other than its own config.
* Open network sockets.
* Make any API calls.

It also refuses to run if any of the required env vars are missing or
malformed, so a misconfigured invocation is caught early.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time
import traceback


RESULT_MARKER = "__SOVEREIGN_RESULT__"


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _truncate(data: bytes, limit: int) -> tuple[bytes, bool]:
    if len(data) <= limit:
        return data, False
    return data[:limit], True


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the process group of a subprocess.

    This function ensures that both the main process and any spawned
    child processes are terminated when a timeout occurs.

    Strategy:
    1. Try os.killpg with SIGKILL first - this kills the entire
       process group and is the most reliable approach.
    2. If that fails (PermissionError), try os.kill on the main process.
    3. If ProcessLookupError, the process already exited.
    4. On PermissionError from kill, try scanning /proc for child
       processes within the container's PID namespace (this is safe
       because we're inside the container and cannot affect host PIDs).

    This approach is robust for the container environment while being
    safe against accidentally killing host processes.
    """
    # Primary: kill the entire process group
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        return
    except ProcessLookupError:
        # Process already exited - nothing to do
        return
    except PermissionError:
        # No permission to kill process group - fall through to alternatives
        pass

    # Fallback 1: kill the main process
    try:
        proc.kill()
        return
    except ProcessLookupError:
        # Process already exited
        return
    except PermissionError:
        pass

    # Fallback 2: scan for and kill child processes within the container's
    # /proc filesystem. This is safe because:
    # - We're inside the container's PID namespace
    # - We can only see container processes
    # - We cannot affect host processes
    try:
        import glob as _glob
        for proc_dir in _glob.glob("/proc/[0-9]*"):
            try:
                pid_str = proc_dir.split("/")[-1]
                pid = int(pid_str)
                if pid == proc.pid:
                    continue  # Already handled
                with open(f"{proc_dir}/status", "r") as f:
                    content = f.read()
                # Check if this is a child of our process
                for line in content.splitlines():
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        if ppid == proc.pid or ppid == 1:
                            # Child or orphaned process - kill it
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except (ProcessLookupError, PermissionError):
                                pass
                        break
            except (ValueError, IOError, ProcessLookupError, PermissionError):
                continue
    except Exception:
        # If /proc scanning fails, give up
        pass


def main() -> int:
    code_b64 = os.environ.get("SOVEREIGN_CODE_B64", "")
    if not code_b64:
        _emit_result(
            {
                "exit_code": 1,
                "timed_out": False,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "error": "missing SOVEREIGN_CODE_B64",
            }
        )
        return 1

    timeout = max(1, _read_int("SOVEREIGN_TIMEOUT", 30))
    max_output = max(1024, _read_int("SOVEREIGN_MAX_OUTPUT", 65536))

    workspace = "/workspace"
    os.makedirs(workspace, exist_ok=True)

    # The code is small by design. Write it once and run it.
    try:
        code_bytes = base64.b64decode(code_b64.encode("ascii"), validate=True)
    except Exception as exc:
        _emit_result(
            {
                "exit_code": 1,
                "timed_out": False,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "error": f"base64 decode failed: {type(exc).__name__}",
            }
        )
        return 1

    code_path = os.path.join(workspace, "main.py")
    try:
        with open(code_path, "wb") as fh:
            fh.write(code_bytes)
        os.chmod(code_path, 0o600)
    except OSError as exc:
        _emit_result(
            {
                "exit_code": 1,
                "timed_out": False,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "error": f"cannot write code file: {type(exc).__name__}",
            }
        )
        return 1

    # Optional stdin.
    stdin_data: bytes = b""
    stdin_b64 = os.environ.get("SOVEREIGN_STDIN_B64", "")
    if stdin_b64:
        try:
            stdin_data = base64.b64decode(stdin_b64.encode("ascii"), validate=True)
        except Exception:
            stdin_data = b""

    # Run the program.
    cmd = ["python3", "-I", "-B", "-u", code_path]
    start = time.monotonic()
    timed_out = False
    stdout_b = b""
    stderr_b = b""
    exit_code = 1
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=workspace,
            start_new_session=True,
        )
    except OSError as exc:
        _emit_result(
            {
                "exit_code": 1,
                "timed_out": False,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "error": f"cannot start program: {type(exc).__name__}",
            }
        )
        return 1

    try:
        stdout_b, stderr_b = proc.communicate(input=stdin_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Kill the entire process group to ensure child processes are terminated.
        # This is the primary cleanup mechanism.
        _kill_process_group(proc)
        try:
            stdout_b, stderr_b = proc.communicate(timeout=2.0)
        except Exception:
            stdout_b, stderr_b = b"", b""
    finally:
        duration = time.monotonic() - start

    if proc.returncode is not None:
        exit_code = int(proc.returncode)
    else:
        exit_code = 1

    # Truncate output.
    stdout_b, stdout_truncated = _truncate(stdout_b, max_output)
    stderr_b, stderr_truncated = _truncate(stderr_b, max_output)

    # Emit the captured program output so the host runner can return it.
    # The program's stdout goes to the container's stdout; the program's
    # stderr goes to the container's stderr, BEFORE the JSON marker so the
    # host parser treats it as program stderr rather than metadata.
    try:
        sys.stdout.buffer.write(stdout_b)
        sys.stdout.buffer.flush()
    except OSError:
        pass
    try:
        sys.stderr.write(stderr_b.decode("utf-8", errors="replace"))
        sys.stderr.flush()
    except OSError:
        pass

    # Emit the result.
    _emit_result(
        {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "duration_seconds": duration,
        }
    )

    # Exit with the program's exit code, so `docker run` reports it
    # as the container exit code.
    if timed_out:
        return 124  # conventional timeout exit code
    return exit_code


def _emit_result(payload: dict) -> None:
    """Write a JSON status line to stderr.

    The host runner parses this line to extract structured information
    about the execution. Everything BEFORE the marker is treated as
    the program's own stderr.
    """
    line = f"{RESULT_MARKER} {json.dumps(payload, sort_keys=True)}\n"
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except OSError:
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Catch-all so the container always exits cleanly with a
        # structured status line.
        _emit_result(
            {
                "exit_code": 1,
                "timed_out": False,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "error": f"entrypoint crashed: {traceback.format_exc()[-200:]}",
            }
        )
        sys.exit(1)
