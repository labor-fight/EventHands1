#!/usr/bin/python3 -I
"""Validate or install the fixed system backend; no jobs or user services start."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys
import tempfile


HELPER = Path("/usr/local/libexec/gpu-borrow-system")
CONFIG = Path("/etc/gpu-borrow-system.json")
SLICE = Path("/etc/systemd/system/gpuborrow.slice")
SUDOERS = Path("/etc/sudoers.d/gpu-borrow-system")
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "SYSTEMD_COLORS": "0"}
IDENTITY_KEYS = {"user", "uid", "gid", "home", "slice"}
# Exact old artifacts produced by this installer. --update is deliberately
# not a generic overwrite switch for arbitrary root-owned code/configuration.
LEGACY_HELPER_SHA256 = {"7ec984ce5489095f02d6f81360a17b4ca338ddae1831e0022fc14dc5f11f8174"}
LEGACY_SLICE_SHA256 = {"bb3f6294228f01b8a24f76e1615a0863214c1943017b21962a614220207b1f3e"}
LEGACY_LIMITS = {"memory_high_bytes": 160 * 1024 ** 3,
                 "memory_max_bytes": 192 * 1024 ** 3,
                 "memory_swap_max_bytes": 0, "job_budget_bytes": 192 * 1024 ** 3,
                 "reserve_bytes": 64 * 1024 ** 3}
UNLIMITED = ["MemoryHigh=infinity", "MemoryMax=infinity", "MemorySwapMax=infinity"]


class InstallError(RuntimeError):
    pass


def build_files(user: str) -> tuple[dict, list[tuple[Path, bytes, int]]]:
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
        raise InstallError("invalid --user")
    try:
        account = pwd.getpwnam(user)
    except KeyError as exc:
        raise InstallError("--user must name an existing account") from exc
    if account.pw_uid <= 0 or not Path(account.pw_dir).is_absolute() or account.pw_dir == "/":
        raise InstallError("--user must be a non-root account with an absolute home")
    cfg = {"user": user, "uid": account.pw_uid, "gid": account.pw_gid,
           "home": account.pw_dir, "slice": "gpuborrow.slice"}
    helper = Path(__file__).with_name("system_helper.py").read_bytes()
    if not helper.startswith(b"#!/usr/bin/python3 -I\n"):
        raise InstallError("helper must use the isolated system Python shebang")
    compile(helper, str(HELPER), "exec")
    slice_text = ("[Unit]\nDescription=GPU borrow jobs isolated from the desktop user tree\n\n"
                  "[Slice]\nMemoryAccounting=yes\nMemoryHigh=infinity\n"
                  "MemoryMax=infinity\nMemorySwapMax=infinity\n")
    sudo_text = ("# Fixed helper validates caller, action, job ID and systemd properties.\n"
                 f"{user} ALL=(root) NOPASSWD: {HELPER} *\n")
    return cfg, [(HELPER, helper, 0o755),
                 (CONFIG, (json.dumps(cfg, indent=2) + "\n").encode(), 0o644),
                 (SLICE, slice_text.encode(), 0o644),
                 (SUDOERS, sudo_text.encode(), 0o440)]


def _validator(name: str) -> str:
    for path in (f"/usr/bin/{name}", f"/usr/sbin/{name}", f"/sbin/{name}"):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise InstallError(f"required validator is not installed: {name}")


def validate_files(files: list[tuple[Path, bytes, int]]) -> None:
    """Both ordinary-user --check and real installation validate exact bytes."""
    analyze, visudo = _validator("systemd-analyze"), _validator("visudo")
    with tempfile.TemporaryDirectory(prefix="gpu-borrow-install-") as temporary:
        base = Path(temporary)
        for target, content, _mode in files:
            (base / target.name).write_bytes(content)
        for command in ([analyze, "verify", str(base / SLICE.name)],
                        [visudo, "-cf", str(base / SUDOERS.name)]):
            result = subprocess.run(command, capture_output=True, text=True,
                                    env=SAFE_ENV, timeout=30, check=False)
            if result.returncode:
                raise InstallError(f"validation failed: {result.stderr.strip() or result.stdout.strip()}")


def _check_directory(directory: Path) -> None:
    for part in (directory, *directory.parents):
        if not part.exists() and not part.is_symlink():
            continue
        metadata = part.lstat()
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0
                or metadata.st_mode & 0o022):
            raise InstallError(f"unsafe installation directory: {part}")


def _known_old_file(target: Path, content: bytes, cfg: dict) -> bool:
    if target == CONFIG:
        try:
            old = json.loads(content)
        except (ValueError, UnicodeError):
            return False
        return old == dict(cfg, **LEGACY_LIMITS)
    if target == HELPER:
        return hashlib.sha256(content).hexdigest() in LEGACY_HELPER_SHA256
    if target == SLICE:
        return hashlib.sha256(content).hexdigest() in LEGACY_SLICE_SHA256
    return False


def check_existing(files: list[tuple[Path, bytes, int]], *, update: bool = False) -> None:
    cfg = json.loads(next(content for target, content, _mode in files if target == CONFIG)) if update else None
    for target, content, mode in files:
        _check_directory(target.parent)
        if target.is_symlink():
            raise InstallError(f"refusing symlink destination: {target}")
        if target.exists():
            metadata = target.lstat()
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0
                    or metadata.st_mode & 0o022):
                raise InstallError(f"unsafe existing destination: {target}")
            current = target.read_bytes()
            if current != content and not (update and _known_old_file(target, current, cfg)):
                raise InstallError(f"existing file differs or is unknown; review it before replacing: {target}")
            if stat.S_IMODE(metadata.st_mode) != mode:
                raise InstallError(f"existing file has unexpected mode; review before replacing: {target}")


def _write_file(target: Path, content: bytes, mode: int) -> None:
    descriptor, staging = tempfile.mkstemp(prefix=".gpu-borrow-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
            os.fchown(stream.fileno(), 0, 0)
        os.replace(staging, target)
    finally:
        Path(staging).unlink(missing_ok=True)


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["/usr/bin/systemctl", "--system", *args], env=SAFE_ENV,
                          capture_output=True, text=True, timeout=30, check=False)


def clear_old_limits(cfg: dict) -> None:
    """Only undo this tool's old limits; never stop jobs or change oomd."""
    result = _systemctl("set-property", "--runtime", cfg["slice"], *UNLIMITED)
    if result.returncode:
        raise InstallError("could not remove the old slice memory limits: " + result.stderr.strip())
    prefix = f"gpu-borrow-{cfg['uid']}-"
    listing = _systemctl("list-units", "--type=service", "--all", "--no-legend", "--plain",
                         "--full", "--no-pager", prefix + "*.service")
    if listing.returncode:
        raise InstallError("could not inspect this tool's existing services: " + listing.stderr.strip())
    pattern = re.compile(re.escape(prefix) + r"[a-z0-9][a-z0-9-]{0,63}\.service\Z", re.ASCII)
    for line in listing.stdout.splitlines():
        if not line.strip():
            continue
        unit = line.split()[0]
        if not pattern.fullmatch(unit):
            raise InstallError("unexpected unit name in the bounded update query")
        parent = _systemctl("show", "--property=Slice", "--value", unit)
        if parent.returncode:
            absent = _systemctl("show", "--property=LoadState", "--value", unit)
            if absent.returncode == 0 and absent.stdout.strip() == "not-found":
                continue  # A transient job completed during the query.
            raise InstallError(f"could not verify the slice of {unit}")
        if parent.stdout.strip() != cfg["slice"]:
            continue
        result = _systemctl("set-property", "--runtime", unit, *UNLIMITED)
        if result.returncode:
            raise InstallError(f"could not remove the old memory limits of {unit}: {result.stderr.strip()}")


def install_files(files: list[tuple[Path, bytes, int]], *, update: bool = False) -> None:
    if os.getuid() != 0 or os.geteuid() != 0:
        raise InstallError("installation needs sudo; use --check or --plan without root")
    check_existing(files, update=update)
    previous: dict[Path, tuple[bytes, int] | None] = {}
    try:
        # Sudoers comes last; no passwordless entry exists before helper+config.
        for target, content, mode in files:
            target.parent.mkdir(parents=True, mode=0o755, exist_ok=True)
            _check_directory(target.parent)
            if target.exists() and target.read_bytes() == content:
                continue
            previous[target] = ((target.read_bytes(), stat.S_IMODE(target.stat().st_mode))
                                if target.exists() else None)
            _write_file(target, content, mode)
        result = _systemctl("daemon-reload")
        if result.returncode:
            raise InstallError("systemctl daemon-reload failed")
    except BaseException:
        for target, old in reversed(previous.items()):
            if old is None:
                target.unlink(missing_ok=True)
            else:
                _write_file(target, old[0], old[1])
        raise
    # A runtime update failure must not silently restore the withdrawn policy.
    # The new files remain installed and the command reports the incomplete step.
    if update:
        cfg = json.loads(next(content for target, content, _mode in files if target == CONFIG))
        clear_old_limits(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="explicit borrower account, normally lyq")
    parser.add_argument("--update", action="store_true", help="replace only recognized prior tool files and remove their old limits")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate staged files without installing")
    mode.add_argument("--plan", action="store_true", help="validate and print proposed configuration")
    args = parser.parse_args(argv)
    cfg, files = build_files(args.user)
    validate_files(files)
    if args.plan:
        print(json.dumps({"config": cfg, "update": args.update,
                          "targets": [str(path) for path, _, _ in files]}, indent=2))
        for path, content, _mode in files[1:]:
            print(f"\n{path}\n{content.decode()}", end="")
        return 0
    if args.check:
        print("Validated system slice and sudoers; no system files changed.")
        return 0
    install_files(files, update=args.update)
    print("Installed isolated system backend without a host-memory policy. No job, user service or oomd was stopped/restarted.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InstallError, OSError, subprocess.SubprocessError) as exc:
        print(f"gpu-borrow install: {exc}", file=sys.stderr)
        raise SystemExit(1)
