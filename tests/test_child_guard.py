import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from vln_improve import child_guard
from vln_improve.pipeline import evaluate_head


GUARD = Path(child_guard.__file__).resolve()


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("test child did not reach its expected lifecycle state")


def running(pid):
    # Query only a PID created by this test; a zombie owns no GPU/resources.
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


def fake_prctl(monkeypatch, *, parents, result=0):
    parent_ids = iter(parents)
    calls = []

    def prctl(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(child_guard.os, "getppid", lambda: next(parent_ids))
    monkeypatch.setattr(child_guard.ctypes, "CDLL", lambda *a, **k: SimpleNamespace(prctl=prctl))
    return calls


def test_linux_guard_arms_kernel_signal_and_rechecks_parent(monkeypatch):
    calls = fake_prctl(monkeypatch, parents=[123, 123])
    child_guard.arm_linux_parent_death(123)
    assert calls == [(child_guard.PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)]


def test_parent_exit_before_arming_does_not_start_unprotected_child(monkeypatch):
    calls = fake_prctl(monkeypatch, parents=[1])
    with pytest.raises(child_guard.ParentExited):
        child_guard.arm_linux_parent_death(123)
    assert calls == []


def test_parent_exit_during_arming_is_detected(monkeypatch):
    fake_prctl(monkeypatch, parents=[123, 1])
    with pytest.raises(child_guard.ParentExited):
        child_guard.arm_linux_parent_death(123)


def test_failed_prctl_fails_closed(monkeypatch):
    fake_prctl(monkeypatch, parents=[123], result=-1)
    monkeypatch.setattr(child_guard.ctypes, "get_errno", lambda: errno.EPERM)
    with pytest.raises(OSError) as caught:
        child_guard.arm_linux_parent_death(123)
    assert caught.value.errno == errno.EPERM


def test_linux_exec_occurs_only_after_parent_guard(monkeypatch):
    events = []
    monkeypatch.setattr(child_guard.sys, "platform", "linux")
    monkeypatch.setattr(child_guard, "arm_linux_parent_death", lambda pid: events.append(("arm", pid)))

    def execvp(binary, command):
        events.append(("exec", binary, command))
        raise SystemExit(0)

    monkeypatch.setattr(child_guard.os, "execvp", execvp)
    with pytest.raises(SystemExit):
        child_guard.run(["python", "eval.py"], 123)
    assert events == [("arm", 123), ("exec", "python", ["python", "eval.py"])]


def test_guard_preserves_success_exit_and_arguments(tmp_path):
    output = tmp_path / "arguments.json"
    command = [sys.executable, str(GUARD), "--parent-pid", str(os.getpid()), "--",
               sys.executable, "-c", "import json,sys;open(sys.argv[1],'w').write(json.dumps(sys.argv[2:]))",
               str(output), "path with spaces", "$(literal)"]
    result = subprocess.run(command, timeout=8, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == ["path with spaces", "$(literal)"]


def test_guard_preserves_child_failure_exit_code():
    result = subprocess.run([sys.executable, str(GUARD), "--parent-pid", str(os.getpid()), "--",
                             sys.executable, "-c", "raise SystemExit(17)"], timeout=8)
    assert result.returncode == 17


def test_parent_sigkill_stops_owned_evaluator_and_spares_unrelated_process(tmp_path):
    """Real process test: Linux uses prctl; macOS exercises polling fallback."""
    ready = tmp_path / "evaluator.pid"
    guarded = tmp_path / "guard.pid"
    child_code = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(90)"
    parent_code = (
        "import os,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,sys.argv[1],'--parent-pid',str(os.getpid()),'--',"
        "sys.executable,'-c',sys.argv[4],sys.argv[2]],start_new_session=True);"
        "open(sys.argv[3],'w').write(str(p.pid));time.sleep(90)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code, str(GUARD), str(ready),
                               str(guarded), child_code], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(90)"], start_new_session=True)
    guard_pid = None
    try:
        wait_until(lambda: ready.exists() and ready.read_text().strip() and guarded.exists())
        evaluator_pid = int(ready.read_text())
        guard_pid = int(guarded.read_text())
        assert running(evaluator_pid)
        parent.kill()
        parent.wait(timeout=5)
        wait_until(lambda: not running(evaluator_pid))
        wait_until(lambda: not running(guard_pid))
        assert unrelated.poll() is None
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if guard_pid is not None and running(guard_pid):
            os.killpg(guard_pid, signal.SIGKILL)
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_pipeline_evaluation_runs_through_guard(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_duet.py").write_text(
        "import json,sys\n"
        "output=sys.argv[sys.argv.index('--output')+1]\n"
        "open(output,'w').write(json.dumps({'evaluation_finished':True}))\n"
    )
    report = evaluate_head(tmp_path / "head.pt", tmp_path / "report.json",
                           {"project_root": str(tmp_path), "config": "fixture.json", "timeout_seconds": 5},
                           lambda: False)
    assert report == {"evaluation_finished": True}


def test_pipeline_graceful_stop_cleans_guard_and_evaluator(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    ready = tmp_path / "evaluation.pid"
    (scripts / "run_duet.py").write_text(
        "import os,time\n"
        f"open({str(ready)!r},'w').write(str(os.getpid()))\n"
        "time.sleep(90)\n"
    )
    with pytest.raises(InterruptedError):
        evaluate_head(tmp_path / "head.pt", tmp_path / "report.json",
                      {"project_root": str(tmp_path), "config": "fixture.json", "timeout_seconds": 5},
                      lambda: ready.exists() and bool(ready.read_text().strip()))
    assert not running(int(ready.read_text()))
