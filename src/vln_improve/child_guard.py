"""Launch one evaluation command whose lifetime is bounded by its parent.

This standalone stdlib entry point deliberately avoids Popen(preexec_fn=...),
which is unsafe once the trainer has CUDA/BLAS worker threads. On Linux the
kernel terminates this process after its direct parent dies; exec preserves
that protection for the actual evaluator. Other platforms use parent polling.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time


PR_SET_PDEATHSIG = 1


class ParentExited(RuntimeError):
    """The requesting parent exited before evaluation could start safely."""


def _check_parent(expected_parent: int) -> None:
    if type(expected_parent) is not int or expected_parent <= 1:
        raise ValueError("expected parent PID must be greater than one")
    if os.getppid() != expected_parent:
        raise ParentExited("training parent exited; refusing to launch orphan evaluation")


def arm_linux_parent_death(expected_parent: int) -> None:
    """Fail closed if Linux cannot bind this process to its requesting parent."""
    _check_parent(expected_parent)
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    # An abruptly lost training process cannot perform graceful cleanup. This
    # signal applies only to its evaluator, which has no training state to save.
    if prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # If the parent died before prctl ran, Linux would not send the signal.
    _check_parent(expected_parent)


def _portable_supervisor(command: list[str], expected_parent: int) -> int:
    """Bound direct child lifetime by parent PID when prctl is unavailable."""
    stopping = []
    previous = {}

    def handler(signum, frame):
        stopping.append(signum)

    process = None
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, handler)
        _check_parent(expected_parent)
        if stopping:
            return 128 + stopping[-1]
        # Inherit the guard's private process group. The pipeline's normal
        # killpg cleanup consequently reaches both the guard and its child.
        process = subprocess.Popen(command)
        while process.poll() is None:
            if stopping or os.getppid() != expected_parent:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.1)
        return process.returncode if process.returncode >= 0 else 128 - process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        for signum, prior in previous.items():
            signal.signal(signum, prior)


def run(command: list[str], expected_parent: int) -> int:
    if not command:
        raise ValueError("an evaluation command is required")
    if sys.platform.startswith("linux"):
        arm_linux_parent_death(expected_parent)
        # No shell, no extra process, and no unguarded GPU-owning child.
        os.execvp(command[0], command)
        raise AssertionError("execvp unexpectedly returned")
    return _portable_supervisor(command, expected_parent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    try:
        return run(command, options.parent_pid)
    except (ParentExited, OSError, ValueError) as error:
        print(f"child_guard: {error}", file=sys.stderr, flush=True)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
