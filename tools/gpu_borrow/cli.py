#!/usr/bin/env python3
"""CLI usable both as a module and through a symlink in ~/.local/bin."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import pwd
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.gpu_borrow.monitor import NvidiaMonitor, MonitorError
from tools.gpu_borrow.supervisor import BorrowRunner, select_gpus
from tools.gpu_borrow.isolation import SystemIsolation


def nonnegative(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("必须为有限非负数")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="lmx 优先、按卡让出、空闲 10 分钟后续用")
    sub = parser.add_subparsers(dest="action", required=True)
    default_state = Path.home() / ".local/state/gpu-borrow"
    run = sub.add_parser("run", help="包装任意支持断点恢复的命令")
    run.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    run.add_argument("--priority-user", default="lmx")
    run.add_argument("--idle-seconds", type=nonnegative, default=600)
    run.add_argument("--poll-seconds", type=nonnegative, default=0.5)
    run.add_argument("--grace-seconds", type=nonnegative, default=0)
    run.add_argument("--idle-memory-mib", type=int, default=256)
    run.add_argument("--idle-utilization", type=int, choices=range(101), default=5)
    run.add_argument("--resume-command", help="抢占后使用的命令；按 shell 引号拆分，但不启动 shell")
    run.add_argument("--state-dir", type=Path, default=default_state)
    run.add_argument("--backend", choices=("auto", "nvml", "smi"), default="auto")
    run.add_argument("--isolation", choices=("system", "user"), default="system",
                     help="默认系统级内存隔离；user 只供受控的轻量兼容测试")
    run.add_argument("command", nargs=argparse.REMAINDER)
    probe = sub.add_parser("probe", help="只读查看 GPU 与进程 UID")
    probe.add_argument("--gpus", default="0,1,2,3")
    probe.add_argument("--backend", choices=("auto", "nvml", "smi"), default="auto")
    status = sub.add_parser("status", help="查看本工具的任务记录")
    status.add_argument("--state-dir", type=Path, default=default_state)
    sub.add_parser("check-system", help="用极小 CPU 作业核对系统隔离，不使用 GPU")
    args = parser.parse_args(argv)
    if args.action == "check-system":
        isolation = SystemIsolation()
        job_id = "check-" + uuid.uuid4().hex[:12]
        code = (
            "import json,os,pathlib; "
            "row=next(x for x in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')); "
            "cg=row[3:]; assert cg.startswith('/gpuborrow.slice/'),cg; "
            "base=pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/'); "
            f"assert os.getuid()=={os.getuid()}; "
            "assert (base/'memory.max').read_text().strip()=='max'; "
            "assert (base/'memory.high').read_text().strip()=='max'; "
            "assert (base/'memory.swap.max').read_text().strip()=='max'; "
            "shared=pathlib.Path('/sys/fs/cgroup/gpuborrow.slice'); "
            "assert (shared/'memory.max').read_text().strip()=='max'; "
            "assert (shared/'memory.high').read_text().strip()=='max'; "
            "assert (shared/'memory.swap.max').read_text().strip()=='max'; "
            "print(json.dumps({'verified':True,'uid':os.getuid(),'cgroup':cg,'gpu_used':False}))"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        isolation.publish(job_id, argv=["/usr/bin/python3", "-I", "-c", code],
                          cwd=Path.cwd(), env=env)
        result = subprocess.run(isolation.command("start", job_id), timeout=20)
        return result.returncode
    if args.action == "status":
        records = []
        for path in sorted((args.state_dir / "runs").glob("*/state.json")):
            row = json.loads(path.read_text())
            row["run_dir"] = str(path.parent)
            records.append(row)
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0
    if args.action == "run":
        if os.getuid() == 0 or os.geteuid() != os.getuid():
            parser.error("请用自己的普通用户运行，不能以 root 执行训练命令")
        if not args.gpus:
            parser.error("请用 --gpus 或 CUDA_VISIBLE_DEVICES 明确要借用的卡")
        if args.poll_seconds <= 0 or args.idle_memory_mib < 0:
            parser.error("poll-seconds 必须 > 0，idle-memory-mib 必须 >= 0")
        if args.command[:1] == ["--"]:
            args.command = args.command[1:]
        if not args.command:
            parser.error("-- 后必须提供原始启动命令")
        priority_uid = pwd.getpwnam(args.priority_user).pw_uid
        if priority_uid == os.getuid():
            parser.error("priority-user 不能是自己")
        system_isolation = SystemIsolation() if args.isolation == "system" else None
        if args.isolation == "user":
            systemd = subprocess.run(
                ["systemctl", "--user", "show", "--property=Version", "--value"],
                capture_output=True, text=True, timeout=5,
            )
            if systemd.returncode:
                parser.error("需要可用的用户 systemd；请先启用用户服务管理器")
    monitor = NvidiaMonitor(backend=args.backend)
    try:
        gpus = select_gpus(monitor.snapshot(), args.gpus)
        if args.action == "probe":
            print(json.dumps({"backend": monitor.backend,
                              "gpus": [asdict(g) for g in gpus]}, indent=2))
            return 0
        resume = shlex.split(args.resume_command) if args.resume_command else args.command
        if not resume:
            parser.error("resume-command 不能为空")
        args.state_dir = args.state_dir.expanduser().resolve()
        args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if args.state_dir.stat().st_uid != os.getuid() or args.state_dir.stat().st_mode & 0o077:
            parser.error("state-dir 必须属于自己，且权限为 0700")
        name = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        runner = BorrowRunner(
            monitor=monitor, gpus=gpus, priority_uid=priority_uid,
            command=args.command, resume_command=resume, cwd=Path.cwd(),
            state_dir=args.state_dir, run_dir=args.state_dir / "runs" / name,
            idle_seconds=args.idle_seconds, grace_seconds=args.grace_seconds,
            idle_memory_mib=args.idle_memory_mib, idle_utilization=args.idle_utilization,
            system_isolation=system_isolation,
        )
        stopping = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, lambda _sig, _frame: stopping.set())
        runner.event("waiting", state_dir=str(runner.run_dir), backend=monitor.backend,
                     idle_seconds=args.idle_seconds)
        try:
            while not stopping.is_set() and not runner.finished:
                started = time.monotonic()
                runner.tick()
                if not runner.finished:
                    stopping.wait(max(0, args.poll_seconds - (time.monotonic() - started)))
            return runner.exit_code if runner.finished else 130
        finally:
            runner.close()
    finally:
        monitor.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, MonitorError, OSError) as exc:
        print(f"gpu-borrow: {exc}", file=sys.stderr)
        raise SystemExit(1)
