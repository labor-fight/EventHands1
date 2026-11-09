"""Keep large jobs outside the desktop's user-manager cgroup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess


SYSTEM_HELPER = Path("/usr/local/libexec/gpu-borrow-system")
SYSTEM_CONFIG = Path("/etc/gpu-borrow-system.json")


class SystemIsolation:
    def __init__(self):
        for path in (SYSTEM_HELPER, SYSTEM_CONFIG):
            if not path.is_file() or path.stat().st_uid != 0 or path.stat().st_mode & 0o022:
                raise ValueError(
                    "系统隔离后端尚未安装：请先按 README 用 sudo 安装；"
                    "大内存借卡任务保持暂停，不回退到桌面用户 scope"
                )
        self.config = json.loads(SYSTEM_CONFIG.read_text())
        if self.config["uid"] != os.getuid():
            raise ValueError("系统隔离后端不是为当前用户配置的")
        self.root = Path(self.config["home"]) / ".local/state/gpu-borrow"

    def command(self, verb: str, job_id: str, *args: str) -> list[str]:
        return ["sudo", "-n", str(SYSTEM_HELPER), verb, job_id, *args]

    def unit(self, job_id: str) -> str:
        return f"gpu-borrow-{os.getuid()}-{job_id}.service"

    def publish(self, job_id: str, *, argv: list[str], cwd: Path, env: dict) -> Path:
        directory = self.root / "system-requests"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
            raise ValueError("system-requests 目录必须属于自己且权限为 0700")
        path = directory / f"{job_id}.json"
        stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        with path.open("x") as handle:
            json.dump({"argv": argv, "cwd": str(cwd), "env": env,
                       "supervisor_pid": os.getpid(), "supervisor_start_time": int(stat[19])}, handle)
        path.chmod(0o600)
        return path

    def show(self, job_id: str, prop: str):
        return subprocess.run(self.command("show", job_id, prop),
                              capture_output=True, text=True, timeout=5)

    def kill(self, job_id: str, sig: signal.Signals):
        name = "TERM" if sig == signal.SIGTERM else "KILL"
        return subprocess.run(self.command("kill", job_id, name),
                              capture_output=True, text=True, timeout=5)
