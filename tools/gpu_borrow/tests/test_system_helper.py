"""Privilege boundaries and installer validation; no real system units or sudo."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.gpu_borrow import install_system as installer
from tools.gpu_borrow import system_helper as helper


CONFIG = {"user": "lyq", "uid": 1004, "gid": 1004, "home": "/home/lyq",
          "slice": "gpuborrow.slice"}
ACCOUNT = SimpleNamespace(pw_uid=1004, pw_gid=1004, pw_dir="/home/lyq")
JOB = "012345abcdef-1"


class HelperTests(unittest.TestCase):
    def test_no_options_paths_or_unit_names_as_job_id(self):
        for value in ("../other", "--help", "a.service", "a/b", "a b", "", "a" * 65):
            with self.subTest(value=value), self.assertRaises(helper.HelperError):
                helper.validate_job_id(value)
        self.assertEqual(helper.unit_name(CONFIG, JOB), f"gpu-borrow-1004-{JOB}.service")

    def test_fixed_start_is_unprivileged_without_added_memory_policy(self):
        command = helper.start_command(CONFIG, JOB)
        for value in ("--system", "--uid=1004", "--gid=1004", "--slice=gpuborrow.slice",
                      "--property=KillMode=control-group"):
            self.assertIn(value, command)
        self.assertEqual(command[-5:], ["/usr/bin/python3", "-I", helper.INSTALLED_HELPER, "exec", JOB])
        self.assertFalse(any("PAMName" in item for item in command))
        self.assertFalse(any("MemoryHigh" in item or "MemoryMax" in item
                             or "MemorySwapMax" in item for item in command))

    def test_privileged_actions_reject_wrong_identity(self):
        with patch.object(helper.os, "getuid", return_value=1004), \
                patch.object(helper.os, "geteuid", return_value=1004):
            with self.assertRaises(helper.HelperError):
                helper.authorize_root(CONFIG)
        with patch.object(helper.os, "getuid", return_value=0), \
                patch.object(helper.os, "geteuid", return_value=0), \
                patch.dict(os.environ, {"SUDO_UID": "1005"}):
            with self.assertRaises(helper.HelperError):
                helper.authorize_root(CONFIG)

    def test_sudo_caller_is_uid_not_sudo_user_name(self):
        with patch.object(helper.os, "getuid", return_value=0), \
                patch.object(helper.os, "geteuid", return_value=0), \
                patch.dict(os.environ, {"SUDO_UID": "1004", "SUDO_USER": "someone-else"}):
            helper.authorize_root(CONFIG)

    def test_root_cannot_execute_user_request(self):
        with patch.object(helper.os, "getuid", return_value=0), \
                patch.object(helper.os, "geteuid", return_value=0), \
                patch.object(helper, "read_request") as read, \
                patch.object(helper.os, "execvpe") as execute:
            with self.assertRaises(helper.HelperError):
                helper.execute_request(CONFIG, JOB)
        read.assert_not_called()
        execute.assert_not_called()

    def test_literal_argv_and_environment_only_after_drop(self):
        request = {"argv": ["python", "train.py", "--label=$HOME", "${NOT_EXPANDED}"],
                   "env": {"PATH": "/an/env/bin", "CUDA_VISIBLE_DEVICES": "GPU-example"},
                   "cwd": "/a/project", "supervisor_pid": 123, "supervisor_start_time": 456}
        with contextlib.ExitStack() as stack:
            for name in ("getuid", "geteuid", "getgid", "getegid"):
                stack.enter_context(patch.object(helper.os, name, return_value=1004))
            stack.enter_context(patch.object(helper, "read_request", return_value=request))
            change = stack.enter_context(patch.object(helper.os, "chdir"))
            execute = stack.enter_context(patch.object(helper.os, "execvpe"))
            helper.execute_request(CONFIG, JOB)
        change.assert_called_once_with("/a/project")
        execute.assert_called_once_with("python", request["argv"], request["env"])

    def test_no_arbitrary_signal_property_or_root_command(self):
        for args in (["kill", JOB, "STOP"], ["show", JOB, "ExecStart"],
                     ["start", JOB, "--uid=0"], ["stop", JOB]):
            with self.subTest(args=args), self.assertRaises(helper.HelperError), \
                    patch.object(helper, "load_config") as load:
                helper.main(args)
                load.assert_not_called()

    def test_show_and_kill_have_clean_environment(self):
        with patch.object(helper, "load_config", return_value=CONFIG), \
                patch.object(helper, "authorize_root"), \
                patch.object(helper.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
                patch.dict(os.environ, {"DBUS_SYSTEM_BUS_ADDRESS": "unix:path=/attacker"}):
            self.assertEqual(helper.main(["show", JOB, "ControlGroup"]), 0)
        command = run.call_args.args[0]
        self.assertEqual(command, [helper.SYSTEMCTL, "--system", "show", "--property=ControlGroup",
                                   "--value", f"gpu-borrow-1004-{JOB}.service"])
        self.assertNotIn("DBUS_SYSTEM_BUS_ADDRESS", run.call_args.kwargs["env"])

    def test_bad_supervisor_identity_prevents_any_launch(self):
        request = {"supervisor_pid": 123, "supervisor_start_time": 456}
        with patch.object(helper, "read_request", return_value=request), \
                patch.object(helper, "supervisor_alive", return_value=False), \
                patch.object(helper.subprocess, "Popen") as launch:
            with self.assertRaises(helper.HelperError):
                helper.run_start(CONFIG, JOB)
        launch.assert_not_called()

    def test_supervisor_loss_kills_only_fixed_unit(self):
        request = {"supervisor_pid": 123, "supervisor_start_time": 456,
                   "argv": ["arbitrary-user-code"], "env": {"DBUS_SYSTEM_BUS_ADDRESS": "fake"}}
        client = Mock()
        client.poll.side_effect = [None, None, 137]
        with patch.object(helper, "read_request", return_value=request), \
                patch.object(helper, "supervisor_alive", side_effect=[True, False]), \
                patch.object(helper.subprocess, "Popen", return_value=client) as launch, \
                patch.object(helper.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as stop, \
                patch.object(helper.time, "sleep"):
            self.assertEqual(helper.run_start(CONFIG, JOB), 137)
        self.assertNotIn("arbitrary-user-code", launch.call_args.args[0])
        self.assertNotIn("DBUS_SYSTEM_BUS_ADDRESS", launch.call_args.kwargs["env"])
        self.assertEqual(stop.call_count, 2)  # Retry even if first kill preceded unit registration.
        for call in stop.call_args_list:
            self.assertEqual(call.args[0][-1], f"gpu-borrow-1004-{JOB}.service")
            self.assertIn("--signal=SIGKILL", call.args[0])

    def test_request_symlink_and_fifo_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(CONFIG, home=temporary, uid=os.getuid())
            target = helper.request_path(cfg, JOB)
            target.parent.mkdir(parents=True)
            other = Path(temporary) / "other.json"
            other.write_text('{}')
            target.symlink_to(other)
            with self.assertRaises(helper.HelperError):
                helper.read_request(cfg, JOB)
            target.unlink()
            os.mkfifo(target)
            with self.assertRaises(helper.HelperError):
                helper.read_request(cfg, JOB)

    def test_request_size_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(CONFIG, home=temporary, uid=os.getuid())
            target = helper.request_path(cfg, JOB)
            target.parent.mkdir(parents=True)
            target.write_text("x" * 256)
            target.chmod(0o600)
            with patch.object(helper, "MAX_REQUEST_BYTES", 32), self.assertRaises(helper.HelperError):
                helper.read_request(cfg, JOB)

    def test_trusted_config_owner_permissions_and_no_legacy_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(CONFIG))
            trusted = SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=0, st_size=1000)
            with patch.object(helper, "CONFIG_PATH", path), \
                    patch.object(helper.pwd, "getpwnam", return_value=ACCOUNT), \
                    patch.object(helper.os, "fstat", return_value=trusted):
                self.assertEqual(helper.load_config(), CONFIG)
                path.write_text(json.dumps(dict(CONFIG, **installer.LEGACY_LIMITS)))
                with self.assertRaises(helper.HelperError):
                    helper.load_config()
            for metadata in (SimpleNamespace(st_mode=stat.S_IFREG | 0o664, st_uid=0, st_size=1000),
                             SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=1004, st_size=1000)):
                with patch.object(helper, "CONFIG_PATH", path), \
                        patch.object(helper.os, "fstat", return_value=metadata), \
                        self.assertRaises(helper.HelperError):
                    helper.load_config()

    def test_supervisor_identity_includes_start_time_and_uid(self):
        fields = ["S"] + ["0"] * 18 + ["12345"]
        proc_stat = "42 (python (worker)) " + " ".join(fields)
        with patch.object(Path, "read_text", side_effect=["Uid:\t1004\t1004\t1004\t1004\n", proc_stat]):
            self.assertTrue(helper.supervisor_alive(CONFIG, 42, 12345))
        with patch.object(Path, "read_text", side_effect=["Uid:\t1004\t1004\t1004\t1004\n", proc_stat]):
            self.assertFalse(helper.supervisor_alive(CONFIG, 42, 12346))
        with patch.object(Path, "read_text", return_value="Uid:\t1005\t1005\t1005\t1005\n"):
            self.assertFalse(helper.supervisor_alive(CONFIG, 42, 12345))


class InstallerTests(unittest.TestCase):
    def test_existing_different_file_or_wrong_mode_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing"
            path.write_bytes(b"old")
            metadata = SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=0)
            with patch.object(installer, "_check_directory"), \
                    patch.object(Path, "lstat", return_value=metadata):
                with self.assertRaisesRegex(installer.InstallError, "differs"):
                    installer.check_existing([(path, b"new", 0o644)])
                with self.assertRaisesRegex(installer.InstallError, "mode"):
                    installer.check_existing([(path, b"old", 0o755)])
            self.assertEqual(path.read_bytes(), b"old")

    def test_plan_names_only_expected_system_files(self):
        with patch.object(installer.pwd, "getpwnam", return_value=ACCOUNT):
            cfg, files = installer.build_files("lyq")
        self.assertEqual(cfg, CONFIG)
        self.assertEqual([mode for _, _, mode in files], [0o755, 0o644, 0o644, 0o440])
        for setting in (b"MemoryHigh=infinity", b"MemoryMax=infinity", b"MemorySwapMax=infinity"):
            self.assertIn(setting, files[2][1])
        self.assertEqual(set(cfg), {"user", "uid", "gid", "home", "slice"})
        self.assertEqual(files[-1][1].count(b"NOPASSWD:"), 1)
        self.assertIn(b"/usr/local/libexec/gpu-borrow-system *", files[-1][1])

    def test_check_and_plan_do_not_install(self):
        for option in ("--check", "--plan"):
            with self.subTest(option=option), \
                    patch.object(installer.pwd, "getpwnam", return_value=ACCOUNT), \
                    patch.object(installer, "validate_files") as validate, \
                    patch.object(installer, "install_files") as install, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(installer.main(["--user", "lyq", option]), 0)
                validate.assert_called_once()
                install.assert_not_called()

    def test_validation_failure_prevents_installation(self):
        with patch.object(installer.pwd, "getpwnam", return_value=ACCOUNT), \
                patch.object(installer, "validate_files", side_effect=installer.InstallError("invalid sudoers")), \
                patch.object(installer, "install_files") as install:
            with self.assertRaises(installer.InstallError):
                installer.main(["--user", "lyq"])
        install.assert_not_called()

    def test_both_official_validators_run_without_system_mutation(self):
        with patch.object(installer.pwd, "getpwnam", return_value=ACCOUNT):
            _cfg, files = installer.build_files("lyq")
        with patch.object(installer, "_validator", side_effect=lambda name: "/validator/" + name), \
                patch.object(installer.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            installer.validate_files(files)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:2], ["/validator/systemd-analyze", "verify"])
        self.assertEqual(commands[1][:2], ["/validator/visudo", "-cf"])

    def test_update_only_recognizes_the_previous_identity_and_known_files(self):
        old = json.dumps(dict(CONFIG, **installer.LEGACY_LIMITS)).encode()
        self.assertTrue(installer._known_old_file(installer.CONFIG, old, CONFIG))
        different_identity = json.dumps(dict(CONFIG, user="someone-else", **installer.LEGACY_LIMITS)).encode()
        self.assertFalse(installer._known_old_file(installer.CONFIG, different_identity, CONFIG))
        self.assertFalse(installer._known_old_file(installer.HELPER, b"arbitrary root code", CONFIG))
        self.assertFalse(installer._known_old_file(installer.SUDOERS, b"extra sudo permission", CONFIG))
        self.assertFalse(installer._known_old_file(Path("/etc/other"), old, CONFIG))

    def test_runtime_rollback_touches_only_tool_slice_and_actual_child_services(self):
        own = "gpu-borrow-1004-012345abcdef-1.service"
        outside = "gpu-borrow-1004-fedcba654321-1.service"
        calls = []

        def fake_systemctl(*args):
            calls.append(args)
            if args[0] == "list-units":
                return SimpleNamespace(returncode=0, stdout=f"{own} loaded active running ours\n{outside} loaded active running other\n", stderr="")
            if args[0] == "show":
                return SimpleNamespace(returncode=0, stdout="gpuborrow.slice\n" if args[-1] == own else "other.slice\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(installer, "_systemctl", side_effect=fake_systemctl):
            installer.clear_old_limits(CONFIG)
        mutations = [call for call in calls if call[0] == "set-property"]
        self.assertEqual([call[2] for call in mutations], ["gpuborrow.slice", own])
        for call in mutations:
            self.assertEqual(call[3:], tuple(installer.UNLIMITED))
        self.assertFalse(any(call[0] in {"stop", "restart", "start", "kill"} for call in calls))
        self.assertFalse(any("oomd" in " ".join(call) for call in calls))

    def test_update_flag_is_reviewable_without_privileged_actions(self):
        with patch.object(installer.pwd, "getpwnam", return_value=ACCOUNT), \
                patch.object(installer, "validate_files"), \
                patch.object(installer, "install_files") as install, \
                patch.object(installer, "clear_old_limits") as clear, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(installer.main(["--user", "lyq", "--update", "--plan"]), 0)
        install.assert_not_called()
        clear.assert_not_called()
        self.assertIn('"update": true', output.getvalue())


if __name__ == "__main__":
    unittest.main()
