#!/usr/bin/python3 -I
"""Root-installed, standalone boundary for isolated GPU-borrow system services.

Privileged start uses only supervisor identity metadata from an inert request.
Only ``exec``, after systemd has dropped both real and effective identities,
uses the user's argv/env/cwd.
Do not import this file from the writable project in the installed helper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import re
import signal
import stat
import subprocess
import sys
import time


CONFIG_PATH = Path("/etc/gpu-borrow-system.json")
INSTALLED_HELPER = "/usr/local/libexec/gpu-borrow-system"
SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
PYTHON = "/usr/bin/python3"
JOB_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z", re.ASCII)
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "SYSTEMD_COLORS": "0"}
CONFIG_KEYS = {"user", "uid", "gid", "home", "slice"}
MAX_REQUEST_BYTES = 1024 * 1024


class HelperError(RuntimeError):
    pass


def validate_job_id(job_id: str) -> str:
    if not JOB_ID.fullmatch(job_id):
        raise HelperError("invalid job ID")
    return job_id


def load_config() -> dict:
    """The sole authority for identity and paths is a root-owned file."""
    try:
        fd = os.open(CONFIG_PATH, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0
                    or metadata.st_mode & 0o022 or metadata.st_size > 16384):
                raise HelperError("config must be a root-owned, non-writable regular file")
            cfg = json.load(stream)
    except (OSError, ValueError) as exc:
        raise HelperError(f"cannot read trusted configuration: {exc}") from exc
    if not isinstance(cfg, dict) or set(cfg) != CONFIG_KEYS:
        raise HelperError("invalid configuration fields")
    if not isinstance(cfg["user"], str) or not re.fullmatch(r"[a-z_][a-z0-9_-]*", cfg["user"]):
        raise HelperError("invalid configured user")
    numeric = CONFIG_KEYS - {"user", "home", "slice"}
    if any(type(cfg[key]) is not int or cfg[key] < 0 for key in numeric):
        raise HelperError("invalid numeric configuration")
    try:
        account = pwd.getpwnam(cfg["user"])
    except KeyError as exc:
        raise HelperError("configured user no longer exists") from exc
    if (cfg["uid"] <= 0 or account.pw_uid != cfg["uid"] or account.pw_gid != cfg["gid"]
            or account.pw_dir != cfg["home"] or not Path(cfg["home"]).is_absolute()
            or cfg["home"] == "/"):
        raise HelperError("configured identity differs from the account database")
    if cfg["slice"] != "gpuborrow.slice":
        raise HelperError("invalid isolation slice")
    return cfg


def authorize_root(cfg: dict) -> None:
    if os.getuid() != 0 or os.geteuid() != 0:
        raise HelperError("this action requires the installed sudo helper")
    caller = os.environ.get("SUDO_UID")
    if caller is not None and (not caller.isascii() or not caller.isdecimal()
                               or int(caller) != cfg["uid"]):
        raise HelperError("sudo caller is not the configured borrower")


def unit_name(cfg: dict, job_id: str) -> str:
    return f"gpu-borrow-{cfg['uid']}-{validate_job_id(job_id)}.service"


def request_path(cfg: dict, job_id: str) -> Path:
    return (Path(cfg["home"]) / ".local/state/gpu-borrow/system-requests"
            / (validate_job_id(job_id) + ".json"))


def start_command(cfg: dict, job_id: str) -> list[str]:
    return [SYSTEMD_RUN, "--system", f"--unit={unit_name(cfg, job_id)}",
            f"--uid={cfg['uid']}", f"--gid={cfg['gid']}", f"--slice={cfg['slice']}",
            "--collect", "--wait", "--pipe", "--property=KillMode=control-group",
            "--property=OOMPolicy=stop", "--property=TimeoutStopSec=5s",
            "--", PYTHON, "-I", INSTALLED_HELPER,
            "exec", job_id]


def read_request(cfg: dict, job_id: str) -> dict:
    """Read bounded inert data; root callers use ONLY supervisor metadata."""
    path = request_path(cfg, job_id)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != cfg["uid"]
                    or metadata.st_mode & 0o022 or metadata.st_size > MAX_REQUEST_BYTES):
                raise HelperError("request must be a bounded, borrower-owned regular file")
            text = stream.read(MAX_REQUEST_BYTES + 1)
            if len(text) > MAX_REQUEST_BYTES:
                raise HelperError("request grew beyond the size limit")
            request = json.loads(text)
    except (OSError, ValueError) as exc:
        raise HelperError(f"cannot read job request: {exc}") from exc
    fields = {"argv", "env", "cwd"}
    metadata_fields = {"supervisor_pid", "supervisor_start_time"}
    if not isinstance(request, dict) or set(request) not in (fields, fields | metadata_fields):
        raise HelperError("request requires exactly argv, env and cwd")
    return request


def supervisor_alive(cfg: dict, pid: int, started: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        owners = next(line.split()[1:] for line in status.splitlines() if line.startswith("Uid:"))
        if len(owners) != 4 or any(int(value) != cfg["uid"] for value in owners[:2]):
            return False
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return fields[0] not in {"Z", "X"} and int(fields[19]) == started
    except (OSError, ValueError, StopIteration, IndexError):
        return False


def run_start(cfg: dict, job_id: str) -> int:
    request = read_request(cfg, job_id)
    pid, started = request.get("supervisor_pid"), request.get("supervisor_start_time")
    if (type(pid) is not int or not 0 < pid < 2 ** 31
            or type(started) is not int or not 0 < started < 2 ** 64):
        raise HelperError("start requires a supervisor PID and start time")
    if not supervisor_alive(cfg, pid, started):
        raise HelperError("borrower supervisor is no longer alive with the recorded identity")
    # The privileged process never uses request argv/env/cwd. They are parsed
    # afresh only by the unprivileged exec process inside the isolated service.
    del request
    cancelled = [False]
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[signum] = signal.signal(signum, lambda _sig, _frame: cancelled.__setitem__(0, True))
    client = None
    try:
        client = subprocess.Popen(start_command(cfg, job_id), env=SAFE_ENV)
        while True:
            result = client.poll()
            if result is not None:
                return result
            if cancelled[0] or not supervisor_alive(cfg, pid, started):
                # Repeat until systemd-run exits: the first kill can race the
                # registration of a new unit. Never kill just the local client
                # and accidentally leave a later-created service running.
                subprocess.run([SYSTEMCTL, "--system", "kill", "--kill-who=all",
                                "--signal=SIGKILL", unit_name(cfg, job_id)],
                               env=SAFE_ENV, timeout=5, check=False)
                cancelled[0] = True
            time.sleep(0.5)
    except BaseException:
        if client is not None:
            # Best effort on exceptional manager failure. Ordinary cancellation
            # and supervisor death use the repeated, synchronous path above.
            subprocess.run([SYSTEMCTL, "--system", "kill", "--kill-who=all",
                            "--signal=SIGKILL", unit_name(cfg, job_id)],
                           env=SAFE_ENV, timeout=5, check=False)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def execute_request(cfg: dict, job_id: str) -> None:
    # In particular, sudo helper exec must NEVER run user argv as root.
    if (os.getuid() != cfg["uid"] or os.geteuid() != cfg["uid"]
            or os.getgid() != cfg["gid"] or os.getegid() != cfg["gid"]):
        raise HelperError("exec is permitted only after systemd has dropped privileges")
    request = read_request(cfg, job_id)
    argv, env, cwd = request["argv"], request["env"], request["cwd"]
    if (not isinstance(argv, list) or not argv or not argv[0]
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)):
        raise HelperError("argv must be a nonempty list of strings without NUL")
    if (not isinstance(env, dict) or any(
            not isinstance(key, str) or not key or "=" in key or "\0" in key
            or not isinstance(value, str) or "\0" in value for key, value in env.items())):
        raise HelperError("env must contain valid string environment variables")
    if not isinstance(cwd, str) or "\0" in cwd or not Path(cwd).is_absolute():
        raise HelperError("cwd must be absolute")
    os.chdir(cwd)
    os.execvpe(argv[0], argv, env)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or args[0] not in {"start", "exec", "kill", "show"}:
        raise HelperError("usage: start|exec JOBID; kill JOBID TERM|KILL; show JOBID ControlGroup|LoadState")
    verb, job_id = args[:2]
    validate_job_id(job_id)
    if ((verb in {"start", "exec"} and len(args) != 2)
            or (verb in {"kill", "show"} and len(args) != 3)):
        raise HelperError("unexpected arguments")
    if verb == "kill" and args[2] not in {"TERM", "KILL"}:
        raise HelperError("only TERM and KILL are permitted")
    if verb == "show" and args[2] not in {"ControlGroup", "LoadState"}:
        raise HelperError("unsupported show property")
    cfg = load_config()
    if verb == "exec":
        execute_request(cfg, job_id)
        return 0  # execvpe does not return; useful for mocked tests.
    authorize_root(cfg)
    if verb == "start":
        return run_start(cfg, job_id)
    elif verb == "kill":
        command = [SYSTEMCTL, "--system", "kill", "--kill-who=all",
                   f"--signal=SIG{args[2]}", unit_name(cfg, job_id)]
    else:
        command = [SYSTEMCTL, "--system", "show", f"--property={args[2]}",
                   "--value", unit_name(cfg, job_id)]
    # Do not inherit user DBUS_*, SYSTEMD_*, LD_* or Python environment here.
    return subprocess.run(command, env=SAFE_ENV, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HelperError, OSError, subprocess.SubprocessError) as exc:
        print(f"gpu-borrow-system: {exc}", file=sys.stderr)
        raise SystemExit(1)
