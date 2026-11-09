"""CPU-only policy and isolated process-group integration checks."""
import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tools.gpu_borrow.monitor import GPU, MonitorError, Process
from tools.gpu_borrow.supervisor import BorrowRunner, GPULocks, select_gpus


UUID0 = "GPU-00000000-0000-0000-0000-000000000000"
UUID1 = "GPU-11111111-1111-1111-1111-111111111111"
TEACHER = os.getuid() + 10000


def gpu(index=0, processes=(), used=14, utilization=0):
    return GPU(index, (UUID0, UUID1)[index], used, utilization, tuple(processes))


class FakeMonitor:
    def __init__(self):
        self.value = {0: gpu(), 1: gpu(1)}

    def snapshot(self, uuids=None):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.monitor = FakeMonitor()
        self.now = 0.0
        self.runners = []
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        for runner in self.runners:
            runner.close()
        self.quiet.__exit__(None, None, None)
        self.temporary.cleanup()

    def make_runner(self, indices=(0,), command=None, resume=None, **kwargs):
        command = command or [sys.executable, "-c", "import time; time.sleep(30)"]
        runner = BorrowRunner(
            monitor=self.monitor, gpus=tuple(gpu(i) for i in indices),
            priority_uid=TEACHER, command=command,
            resume_command=resume or command, cwd=self.directory,
            state_dir=self.directory, run_dir=self.directory / str(len(self.runners)),
            clock=lambda: self.now, **kwargs,
            lock_root=self.directory,
        )
        self.runners.append(runner)
        return runner

    def start(self, runner):
        runner.tick()
        self.now += 600
        runner.tick()
        self.assertIsNotNone(runner.child)

    def wait_until_exited(self, runner):
        deadline = time.monotonic() + 5
        while runner.peek_exit() is None:
            if time.monotonic() > deadline:
                self.fail("test child failed to exit")
            time.sleep(0.01)

    def test_full_ten_minutes_before_first_start(self):
        runner = self.make_runner()
        runner.start_child = Mock()
        runner.tick()
        self.now = 599.9
        runner.tick()
        runner.start_child.assert_not_called()
        self.now = 600
        runner.tick()
        runner.start_child.assert_called_once()

    def test_teacher_on_other_card_does_not_preempt(self):
        runner = self.make_runner()
        self.start(runner)
        pid = runner.child.pid
        self.monitor.value[1] = gpu(1, [Process(123, TEACHER)])
        runner.tick()
        self.assertEqual(pid, runner.child.pid)

    def test_teacher_context_with_zero_load_preempts_only_selected_job(self):
        first, second = self.make_runner(), self.make_runner((1,))
        self.start(first)
        self.start(second)
        unaffected = second.child.pid
        self.monitor.value[0] = gpu(0, [Process(123, TEACHER)], used=14)
        first.tick()
        second.tick()
        self.assertIsNone(first.child)
        self.assertEqual(first.phase, "WAITING")
        self.assertEqual(second.child.pid, unaffected)
        self.assertFalse(first.finished)

    def test_each_card_has_its_own_continuous_idle_timer(self):
        runner = self.make_runner((0, 1))
        runner.start_child = Mock()
        runner.tick()
        self.now = 300
        self.monitor.value[0] = gpu(0, [Process(123, TEACHER)])
        runner.tick()
        self.now = 301
        self.monitor.value[0] = gpu()
        runner.tick()
        self.now = 900
        runner.tick()
        runner.start_child.assert_not_called()
        self.now = 901
        runner.tick()
        runner.start_child.assert_called_once()

    def test_monitor_failure_preempts_and_restarts_ten_minute_timer(self):
        runner = self.make_runner()
        self.start(runner)
        self.monitor.value = MonitorError("device query failed")
        runner.tick()
        self.assertIsNone(runner.child)
        self.assertEqual(runner.idle_since, {})
        self.monitor.value = {0: gpu()}
        self.now += 1
        runner.tick()
        self.now += 599
        runner.tick()
        self.assertIsNone(runner.child)
        self.now += 1
        runner.tick()
        self.assertIsNotNone(runner.child)

    def test_graphics_baseline_allowed_but_teacher_graphics_is_busy(self):
        runner = self.make_runner()
        runner.start_child = Mock()
        self.monitor.value[0] = gpu(0, [Process(123, os.getuid(), "G")])
        runner.tick()
        self.now = 600
        runner.tick()
        runner.start_child.assert_called_once()
        runner.locks.release()
        runner.start_child.reset_mock()
        self.monitor.value[0] = gpu(0, [Process(123, TEACHER, "G")])
        runner.tick()
        self.now = 1200
        runner.tick()
        runner.start_child.assert_not_called()

    def test_high_unattributed_memory_resets_idle(self):
        runner = self.make_runner()
        runner.start_child = Mock()
        runner.tick()
        self.now = 599
        self.monitor.value[0] = gpu(used=1024)
        runner.tick()
        self.monitor.value[0] = gpu()
        self.now = 600
        runner.tick()
        runner.start_child.assert_not_called()
        self.assertEqual(runner.idle_since[UUID0], 600)

    def test_failed_command_not_retried_when_teacher_arrives(self):
        runner = self.make_runner(command=[sys.executable, "-c", "raise SystemExit(7)"])
        self.start(runner)
        self.wait_until_exited(runner)
        self.monitor.value[0] = gpu(0, [Process(123, TEACHER)])
        runner.tick()
        self.assertTrue(runner.finished)
        self.assertEqual(runner.phase, "FAILED")
        self.assertEqual(runner.exit_code, 7)
        self.assertEqual(runner.starts, 1)

    def test_successful_command_is_not_resumed(self):
        runner = self.make_runner(command=[sys.executable, "-c", "pass"])
        self.start(runner)
        self.wait_until_exited(runner)
        runner.tick()
        self.assertTrue(runner.finished)
        self.assertEqual(runner.exit_code, 0)
        self.assertEqual(runner.phase, "COMPLETED")

    def test_exact_command_arguments_survive_systemd_environment_expansion(self):
        target = self.directory / "arguments.json"
        values = ["$HOME", "${PATH}", "two words", "literal $(date)", "--", ""]
        command = [sys.executable, "-c", (
            "import json,pathlib,sys; "
            f"pathlib.Path({str(target)!r}).write_text(json.dumps(sys.argv[1:]))"
        ), *values]
        runner = self.make_runner(command=command)
        self.start(runner)
        self.wait_until_exited(runner)
        runner.tick()
        self.assertEqual(__import__('json').loads(target.read_text()), values)

    def test_resume_command_and_uuid_visible_devices(self):
        target = self.directory / "environment.txt"
        resume = [sys.executable, "-c", "import os,pathlib; pathlib.Path(" +
                  repr(str(target)) + ").write_text(os.environ['CUDA_VISIBLE_DEVICES'])"]
        runner = self.make_runner(resume=resume)
        self.start(runner)
        self.monitor.value[0] = gpu(0, [Process(123, TEACHER)])
        runner.tick()
        self.monitor.value[0] = gpu()
        self.now += 1
        runner.tick()
        self.now += 599
        runner.tick()
        self.assertIsNone(runner.child)
        self.now += 1
        runner.tick()
        self.wait_until_exited(runner)
        runner.tick()
        self.assertEqual(target.read_text(), UUID0)
        self.assertEqual(runner.starts, 2)
        self.assertEqual(runner.phase, "COMPLETED")

    def test_multicard_job_child_group_is_killed_but_unrelated_group_lives(self):
        child_file = self.directory / "descendant.pid"
        command = [sys.executable, "-c", (
            "import subprocess,sys,time,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], start_new_session=True); "
            f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(30)"
        )]
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
                                     start_new_session=True)
        try:
            runner = self.make_runner((0, 1), command=command)
            self.start(runner)
            deadline = time.monotonic() + 5
            while not child_file.exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            descendant = int(child_file.read_text())
            self.monitor.value[0] = gpu(0, [Process(descendant, os.getuid())], used=1024)
            runner.tick()
            self.assertIsNotNone(runner.child, "own detached worker must not look external")
            self.monitor.value[1] = gpu(1, [Process(123, TEACHER)])
            runner.tick()
            self.assertIsNone(runner.child)
            self.assertIsNone(unrelated.poll())
            deadline = time.monotonic() + 5
            while Path(f"/proc/{descendant}/stat").exists():
                stat = Path(f"/proc/{descendant}/stat").read_text()
                if stat.split(")", 1)[1].split()[0] == "Z":
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
        finally:
            os.killpg(unrelated.pid, signal.SIGKILL)
            unrelated.wait()

    def test_grace_period_includes_detached_worker_after_launcher_exit(self):
        ready = self.directory / "ready"
        saved = self.directory / "saved"
        worker = (
            "import pathlib,signal,time; "
            "signal.signal(signal.SIGTERM, lambda *args: "
            f"(time.sleep(0.08),pathlib.Path({str(saved)!r}).write_text('saved'),exit(0))); "
            f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)"
        )
        launcher = [sys.executable, "-c", (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session=True); "
            "time.sleep(30)"
        )]
        runner = self.make_runner(command=launcher, grace_seconds=1)
        self.start(runner)
        deadline = time.monotonic() + 5
        while not ready.exists():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.monitor.value[0] = gpu(0, [Process(123, TEACHER)])
        runner.tick()
        self.assertEqual(saved.read_text(), "saved")
        self.assertIsNone(runner.child)

    def test_gpu_locks_block_other_wrapper_and_release(self):
        one = GPULocks(self.directory, (UUID0, UUID1))
        two = GPULocks(self.directory, (UUID1,))
        self.assertTrue(one.acquire())
        self.assertFalse(two.acquire())
        one.release()
        self.assertTrue(two.acquire())
        two.release()

    def test_independent_gpus_can_run_concurrently_after_ten_minutes(self):
        first, second = self.make_runner((0,)), self.make_runner((1,))
        first.tick()
        second.tick()
        self.now = 600
        first.tick()
        second.tick()
        self.assertIsNotNone(first.child)
        self.assertIsNotNone(second.child)

    def test_launch_does_not_query_host_memory_or_psi(self):
        runner = self.make_runner()
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if str(path) in {"/proc/meminfo", "/proc/pressure/memory"}:
                raise AssertionError("GPU borrowing has no host-memory admission policy")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text):
            self.start(runner)
        self.assertIsNotNone(runner.child)

    def test_device_selection_accepts_full_uuid_preserves_order(self):
        selection = select_gpus(self.monitor.value, f"{UUID1},0")
        self.assertEqual([g.index for g in selection], [1, 0])
        with self.assertRaises(ValueError):
            select_gpus(self.monitor.value, "0,0")
        with self.assertRaises(ValueError):
            select_gpus(self.monitor.value, "GPU-111")


if __name__ == "__main__":
    unittest.main()
