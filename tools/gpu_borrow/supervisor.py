"""Run one resumable command, reclaiming only its own process group."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable
import uuid

from .monitor import GPU, MonitorError


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def select_gpus(snapshot: dict[int, GPU], requested: str) -> tuple[GPU, ...]:
    selected = []
    for item in requested.split(","):
        item = item.strip()
        matches = [g for g in snapshot.values()
                   if item == str(g.index) or item == g.uuid]
        if len(matches) != 1:
            raise ValueError(f"GPU {item!r} 不存在；请使用 nvidia-smi 编号或完整 UUID")
        if matches[0].uuid in {g.uuid for g in selected}:
            raise ValueError("GPU 列表不能重复")
        selected.append(matches[0])
    return tuple(selected)


class GPULocks:
    """Per-user, per-UUID leases; the child must not inherit these fds."""

    def __init__(self, directory: Path, uuids: tuple[str, ...]):
        self.directory = directory / "locks"
        self.uuids = tuple(sorted(uuids))
        self.files = []

    def acquire(self) -> bool:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            for uuid in self.uuids:
                handle = (self.directory / f"{uuid}.lock").open("a+")
                self.files.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            self.release()
            return False
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        for handle in self.files:
            handle.close()
        self.files.clear()


class BorrowRunner:
    def __init__(
        self, *, monitor, gpus: tuple[GPU, ...], priority_uid: int,
        command: list[str], resume_command: list[str], cwd: Path,
        state_dir: Path, run_dir: Path, idle_seconds: float = 600,
        grace_seconds: float = 0, idle_memory_mib: int = 256,
        idle_utilization: int = 5, clock: Callable[[], float] = time.monotonic,
        lock_root: Path | None = None,
        system_isolation=None,
    ):
        self.monitor = monitor
        self.gpus = gpus
        self.uuids = tuple(g.uuid for g in gpus)
        self.priority_uid = priority_uid
        self.command = command
        self.resume_command = resume_command
        self.cwd = cwd
        self.run_dir = run_dir
        self.idle_seconds = idle_seconds
        self.grace_seconds = grace_seconds
        self.idle_memory_mib = idle_memory_mib
        self.idle_utilization = idle_utilization
        self.clock = clock
        self.uid = os.getuid()
        self.system_isolation = system_isolation
        self.job_id: str | None = None
        # Log/state locations may vary, but leases must be shared by every
        # wrapper of this user. lock_root is dependency injection for tests.
        self.locks = GPULocks(lock_root or Path.home() / ".local/state/gpu-borrow",
                              self.uuids)
        self.idle_since: dict[str, float] = {}
        self.child: subprocess.Popen | None = None
        self.scope: str | None = None
        self.scope_path: str | None = None
        self.instance_id = uuid.uuid4().hex[:12]
        self.starts = 0
        self.finished = False
        self.exit_code = 0
        self.phase = "WAITING"
        self.reason = "等待每张卡连续空闲"
        self.last_exit_code = None
        self.last_error = None
        self.cards = []
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.events = (self.run_dir / "events.jsonl").open("a", buffering=1)
        self._last_saved = 0.0

    def event(self, action: str, **fields) -> None:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "event": action, **fields}
        self.events.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def save(self, force: bool = False) -> None:
        if not force and self.clock() - self._last_saved < 1:
            return
        atomic_json(self.run_dir / "state.json", {
            "wrapper_pid": os.getpid(), "uid": self.uid,
            "phase": self.phase, "reason": self.reason,
            "job_pid": self.child.pid if self.child else None,
            "scope": self.scope,
            "isolation": "system" if self.system_isolation else "user",
            "starts": self.starts, "last_exit_code": self.last_exit_code,
            "command": self.command, "resume_command": self.resume_command,
            "cwd": str(self.cwd), "gpus": self.cards,
            "idle_seconds": self.idle_seconds,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        self._last_saved = self.clock()

    def owns(self, process) -> bool:
        if self.child is None or process.uid != self.uid:
            return False
        try:
            # torchrun workers may setsid(); cgroup membership survives it.
            groups = Path(f"/proc/{process.pid}/cgroup").read_text().splitlines()
            return any(row.startswith("0::") and
                       self.scope in row[3:].split("/") for row in groups)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise MonitorError(f"无法核验进程 {process.pid} 的 cgroup") from exc

    def signal_scope(self, signum: signal.Signals) -> None:
        if self.system_isolation:
            result = self.system_isolation.kill(self.job_id, signum)
        else:
            result = subprocess.run(
                ["systemctl", "--user", "kill", "--kill-who=all",
                 f"--signal={signum.name}", self.scope],
                capture_output=True, text=True, timeout=5,
            )
        if result.returncode:
            # Collected empty scopes have already disappeared. A bus error or
            # failure to kill a populated unit must never be silently ignored.
            query = self.show_scope("LoadState")
            if query.stdout.strip() not in {"not-found", ""} or query.returncode:
                raise RuntimeError(f"无法结束自己的 scope {self.scope}: {result.stderr.strip()}")

    def show_scope(self, prop: str):
        if self.system_isolation:
            return self.system_isolation.show(self.job_id, prop)
        return subprocess.run(
            ["systemctl", "--user", "show", f"--property={prop}",
             "--value", self.scope], capture_output=True, text=True, timeout=5,
        )

    def scope_populated(self) -> bool:
        if self.scope_path is None:
            query = self.show_scope("ControlGroup")
            if query.returncode:
                raise RuntimeError(f"无法查询自己的 scope {self.scope}")
            if not query.stdout.strip():
                return False
            self.scope_path = query.stdout.strip()
        events = Path("/sys/fs/cgroup") / self.scope_path.lstrip("/") / "cgroup.events"
        try:
            values = dict(row.split() for row in events.read_text().splitlines())
        except FileNotFoundError:
            return False
        if values.get("populated") not in {"0", "1"}:
            raise RuntimeError(f"无法确认自己的 scope 是否已清空：{self.scope}")
        return values["populated"] == "1"

    def blockers(self, gpu: GPU) -> list:
        # The workstation has a small Xorg graphics-only context on each GPU.
        # Keep it visible in probe, but allow that small system baseline.
        return [p for p in gpu.processes if not self.owns(p)
                and (p.uid == self.priority_uid or p.kind != "G")]

    def peek_exit(self):
        # WNOWAIT retains our child's PID even after exit. Never signal a
        # recycled PID/PGID, and clean up surviving DDP ranks before reaping.
        return os.waitid(os.P_PID, self.child.pid,
                         os.WEXITED | os.WNOHANG | os.WNOWAIT)

    def stop_child(self, reason: str, *, graceful: bool = False) -> None:
        if self.child is None:
            return
        child = self.child
        self.event("stopping", pid=child.pid, reason=reason)
        # Each attempt has a distinct scope containing all ranks, including
        # those which created their own process groups or sessions.
        try:
            if graceful and self.grace_seconds > 0:
                self.signal_scope(signal.SIGTERM)
                deadline = time.monotonic() + self.grace_seconds
                while time.monotonic() < deadline:
                    if not self.scope_populated():
                        break
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            self.signal_scope(signal.SIGKILL)
            deadline = time.monotonic() + 5
            while self.scope_populated():
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"自己的 scope 尚未清空：{self.scope}")
                time.sleep(0.02)
            try:
                self.last_exit_code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # The local systemd-run client can outlive its emptied scope.
                # Its unreaped PID still belongs to this Popen, never lmx.
                os.killpg(child.pid, signal.SIGKILL)
                self.last_exit_code = child.wait()
        except BaseException:
            self.phase, self.reason = "CLEANUP_FAILED", "未能确认借用作业已全部退出"
            self.event("cleanup_failed", scope=self.scope)
            self.save(force=True)
            raise
        self.child = None
        self.scope = None
        self.scope_path = None
        self.locks.release()
        self.idle_since.clear()
        self.event("released", reason=reason, exit_code=self.last_exit_code)

    def start_child(self) -> None:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(g.uuid for g in self.gpus)
        env["GPU_BORROW_ATTEMPT"] = str(self.starts + 1)
        argv = self.command if self.starts == 0 else self.resume_command
        self.job_id = f"{self.instance_id}-{self.starts + 1}"
        self.scope = f"gpu-borrow-{self.job_id}.scope"
        scope_properties = []
        groups = Path("/proc/self/cgroup").read_text().splitlines()
        current_unit = next((row.rsplit("/", 1)[-1] for row in groups
                             if row.startswith("0::")), "")
        if current_unit.endswith(".service") and not current_unit.startswith("user@"):
            # When the wrapper itself is a user service, its scope must stop
            # even if that service crashes or is forcefully terminated.
            scope_properties = [f"--property=BindsTo={current_unit}",
                                f"--property=After={current_unit}"]
        if self.system_isolation:
            self.scope = self.system_isolation.unit(self.job_id)
            self.system_isolation.publish(self.job_id, argv=argv, cwd=self.cwd, env=env)
            launcher = self.system_isolation.command("start", self.job_id)
        else:
            argv_file = self.run_dir / f"attempt-{self.starts + 1}.json"
            atomic_json(argv_file, {"argv": argv})
            launcher = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                 f"--unit={self.scope}", *scope_properties, "--",
                 sys.executable, str(Path(__file__).with_name("exec_job.py")),
                 str(argv_file)]
        with (self.run_dir / "command.log").open("ab", buffering=0) as log:
            self.child = subprocess.Popen(
                launcher,
                cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                close_fds=True,
            )
        # Do not acknowledge a running attempt until its scope exists. An
        # immediate preemption must not miss a scope still being registered.
        deadline = time.monotonic() + 5
        while True:
            if self.peek_exit() is not None:
                break
            query = self.show_scope("ControlGroup")
            if query.returncode == 0 and query.stdout.strip():
                self.scope_path = query.stdout.strip()
                if self.system_isolation and not self.scope_path.startswith("/gpuborrow.slice/"):
                    raise RuntimeError("系统借卡作业未进入独立 gpuborrow.slice，停止")
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"等待自己的 scope 注册超时：{self.scope}")
            time.sleep(0.02)
        self.starts += 1
        self.phase = "RUNNING"
        self.reason = "借用中"
        self.event("started", pid=self.child.pid, attempt=self.starts,
                   gpu_uuids=[g.uuid for g in self.gpus])
        self.save(force=True)

    def tick(self) -> None:
        if self.child is not None and self.peek_exit() is not None:
            # Observe genuine completion/failure before a new teacher context
            # or monitor failure can misclassify it as a resumable preemption.
            self.stop_child("命令已退出，清理残留子进程")
            self.exit_code = (self.last_exit_code if self.last_exit_code >= 0
                              else 128 - self.last_exit_code)
            self.phase = "COMPLETED" if self.exit_code == 0 else "FAILED"
            self.reason = "命令正常结束" if self.exit_code == 0 else "命令失败，不自动重试"
            self.finished = True
            self.event(self.phase.lower(), exit_code=self.exit_code)
            self.save(force=True)
            return
        now = self.clock()
        try:
            snapshot = {g.uuid: g for g in self.monitor.snapshot(self.uuids).values()}
            selected = [snapshot[g.uuid] for g in self.gpus]
            blockers = {g.uuid: self.blockers(g) for g in selected}
        except (MonitorError, KeyError) as exc:
            message = str(exc)
            if message != self.last_error:
                self.event("monitor_error", error=message)
            self.last_error = message
            self.stop_child("监测状态未知", graceful=True)
            self.idle_since.clear()
            self.cards = []
            self.phase, self.reason = "WAITING", f"监测失败：{message}"
            self.save(force=True)
            return
        self.last_error = None
        self.cards = []
        for gpu in selected:
            busy = bool(blockers[gpu.uuid])
            idle = (not busy and gpu.used_mib <= self.idle_memory_mib
                    and gpu.utilization <= self.idle_utilization)
            if self.child is None and idle:
                self.idle_since.setdefault(gpu.uuid, now)
            else:
                self.idle_since.pop(gpu.uuid, None)
            elapsed = max(0, now - self.idle_since.get(gpu.uuid, now))
            self.cards.append({
                "index": gpu.index, "uuid": gpu.uuid,
                "idle_for_seconds": round(elapsed, 2),
                "used_mib": gpu.used_mib, "utilization": gpu.utilization,
                "blocker_pids": [p.pid for p in blockers[gpu.uuid]],
            })
        if self.child is not None:
            conflicts = [g.index for g in selected if blockers[g.uuid]]
            if conflicts:
                self.stop_child(f"GPU {conflicts} 出现外部任务", graceful=True)
                self.phase, self.reason = "WAITING", "已让卡，重新等待连续空闲"
            self.save()
            return
        ready = all(g.uuid in self.idle_since and
                    now - self.idle_since[g.uuid] >= self.idle_seconds
                    for g in selected)
        if not ready:
            self.save()
            return
        if not self.locks.acquire():
            self.reason = "其他 gpu-borrow 任务持有这张卡，等待"
            self.save()
            return
        try:
            # Recheck after acquiring all leases, before starting a command.
            fresh = {g.uuid: g for g in self.monitor.snapshot(self.uuids).values()}
            if any(self.blockers(fresh[g.uuid]) or
                   fresh[g.uuid].used_mib > self.idle_memory_mib or
                   fresh[g.uuid].utilization > self.idle_utilization
                   for g in self.gpus):
                self.idle_since.clear()
                self.locks.release()
                return
            self.start_child()
        except (MonitorError, KeyError) as exc:
            self.idle_since.clear()
            self.phase, self.reason = "WAITING", f"启动前复查失败：{exc}"
            self.event("monitor_error", error=str(exc))
            self.locks.release()
        except BaseException:
            self.locks.release()
            raise

    def close(self) -> None:
        try:
            self.stop_child("包装器退出", graceful=True)
            if not self.finished:
                self.phase, self.reason = "STOPPED", "包装器已退出；再次启动需重新计时"
            self.save(force=True)
        finally:
            self.locks.release()
            self.events.close()
