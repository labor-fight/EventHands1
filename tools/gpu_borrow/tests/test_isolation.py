"""Exercise the system boundary without root or allocating GPU memory."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from tools.gpu_borrow.isolation import SystemIsolation


class SystemIsolationTests(unittest.TestCase):
    def test_no_silent_fallback_when_system_backend_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("tools.gpu_borrow.isolation.SYSTEM_HELPER", Path(temporary) / "missing"):
                with self.assertRaisesRegex(ValueError, "不回退"):
                    SystemIsolation()

    def test_request_is_private_and_preserves_argv_environment_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            isolation = object.__new__(SystemIsolation)
            isolation.root = Path(temporary)
            argv = ["python", "two words.py", "$HOME", "literal $(date)"]
            path = isolation.publish("abc-1", argv=argv, cwd=Path.cwd(),
                                     env={"CUDA_VISIBLE_DEVICES": "GPU-test", "PATH": "/custom"})
            result = json.loads(path.read_text())
            self.assertEqual(result["argv"], argv)
            self.assertEqual(result["env"]["PATH"], "/custom")
            self.assertEqual(result["supervisor_pid"], os.getpid())
            self.assertGreater(result["supervisor_start_time"], 0)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                isolation.publish("abc-1", argv=argv, cwd=Path.cwd(), env={})

    def test_control_calls_use_narrow_helper_and_noninteractive_sudo(self):
        isolation = object.__new__(SystemIsolation)
        with patch("tools.gpu_borrow.isolation.subprocess.run") as run:
            isolation.show("abc-1", "ControlGroup")
            call = run.call_args.args[0]
            self.assertEqual(call[:2], ["sudo", "-n"])
            self.assertEqual(call[-3:], ["show", "abc-1", "ControlGroup"])
            isolation.kill("abc-1", signal.SIGKILL)
            self.assertEqual(run.call_args.args[0][-3:], ["kill", "abc-1", "KILL"])


if __name__ == "__main__":
    unittest.main()
