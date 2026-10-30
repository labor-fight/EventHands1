"""Opt-in, separately identified audit-hook repair for frozen Stage5 jobs.

The historical source freezes and checkpoints retain their original bytes.
An amendment records this launcher's and the replacement guard's identities;
every Python child in the same study explicitly re-enters this launcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


MODULE = "tools.s43_revisit.s43_field_hook_repair"
QUEUE_MODULE = "tools.s43_revisit.s43_field_full_queue"
ROOT = Path(__file__).resolve().parents[2]
GUARD_SOURCE = ROOT / "tools/s43_rel/access_guard.py"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_amendment(path, expected_sha256=None):
    path = Path(path).resolve()
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("audit-hook amendment identity changed")
    amendment = json.loads(path.read_text(encoding="utf-8"))
    sources = amendment.get("repair_sources", {})
    required = {str(Path(__file__).resolve()), str(GUARD_SOURCE.resolve())}
    if not required.issubset(sources):
        raise RuntimeError("amendment must identify both repair executable sources")
    for name, expected in sources.items():
        source = Path(name)
        if not source.is_absolute() or sha256(source) != expected:
            raise RuntimeError("audit-hook repair source changed: " + name)
    return amendment, digest


def install_guard():
    """Replace imported aliases without changing any frozen source on disk."""
    replacement = importlib.import_module("tools.s43_rel.access_guard")
    legacy = importlib.import_module("tools.s43_rel.mean_shape_replay")
    original = legacy.InferenceAccessGuard
    if original is not replacement.InferenceAccessGuard:
        for name, module in tuple(sys.modules.items()):
            if module is None or not name.startswith("tools."):
                continue
            for key, value in tuple(vars(module).items()):
                if value is original:
                    setattr(module, key, replacement.InferenceAccessGuard)
    replacement._ensure_audit_hook()
    return replacement


def _study_module(name):
    return name.startswith(("tools.s43_revisit.", "tools.s43_rel."))


def repair_command(command, amendment, *, resume_chain=False, amendment_sha256=None):
    """Wrap only explicit Python -m study jobs; leave other commands intact."""
    if isinstance(command, (str, bytes)):
        return command
    command = list(command)
    if not command or "-m" not in command:
        return command
    interpreter = os.fsdecode(command[0])
    if not Path(interpreter).name.startswith("python"):
        return command
    index = command.index("-m")
    if index + 1 >= len(command):
        return command
    target = command[index + 1]
    if target == MODULE or not _study_module(target):
        return command
    arguments = command[index + 2:]
    if resume_chain and target == QUEUE_MODULE:
        remaining = []
        cursor = 0
        while cursor < len(arguments):
            value = arguments[cursor]
            if value == "--pilot-controller-pid":
                if cursor + 1 >= len(arguments):
                    raise ValueError("missing pilot controller argument")
                cursor += 2
                continue
            if value.startswith("--pilot-controller-pid="):
                cursor += 1
                continue
            remaining.append(value)
            cursor += 1
        arguments = remaining
        if "--resume-existing" not in arguments:
            arguments.append("--resume-existing")
    prefix = command[:index + 1] + [MODULE, "--amendment", str(Path(amendment).resolve())]
    if amendment_sha256 is not None:
        prefix += ["--amendment-sha256", amendment_sha256]
    if resume_chain:
        prefix.append("--resume-chain")
    return prefix + ["--module", target] + arguments


class RepairSubprocess:
    """A per-module subprocess facade, without process-global monkeypatches."""

    def __init__(self, amendment, digest, *, resume_chain=False, on_spawn=None, backend=subprocess):
        self.amendment = Path(amendment).resolve()
        self.digest = digest
        self.resume_chain = resume_chain
        self.on_spawn = on_spawn
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _command(self, command, kwargs):
        verify_amendment(self.amendment, self.digest)
        wrapped = repair_command(command, self.amendment, resume_chain=self.resume_chain,
                                 amendment_sha256=self.digest)
        if self.on_spawn is not None:
            self.on_spawn(command, wrapped, kwargs)
        return wrapped

    def Popen(self, command, *args, **kwargs):
        return self.backend.Popen(self._command(command, kwargs), *args, **kwargs)

    def run(self, command, *args, **kwargs):
        return self.backend.run(self._command(command, kwargs), *args, **kwargs)


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _argument(arguments, key):
    for index, value in enumerate(arguments):
        if value == key and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(key + "="):
            return value.split("=", 1)[1]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256")
    parser.add_argument("--resume-chain", action="store_true")
    parser.add_argument("--module", required=True)
    args, arguments = parser.parse_known_args(argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if args.module == MODULE or not _study_module(args.module):
        raise ValueError("explicit study module required")
    amendment, digest = verify_amendment(args.amendment, args.amendment_sha256)
    receipt_path = args.amendment.resolve().parent / "launches" / f"{os.getpid()}_{time.time_ns()}.json"
    receipt = dict(schema="s43_audit_hook_repair_launch_v1", status="RUNNING", pid=os.getpid(),
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   amendment_path=str(args.amendment.resolve()), amendment_sha256=digest,
                   original_sourcefreeze=amendment.get("original_sourcefreeze"),
                   repair_sources=amendment["repair_sources"], module=args.module,
                   target_argv=[args.module, *arguments], launcher_argv=list(sys.argv),
                   resume_chain=args.resume_chain, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   guard_class="tools.s43_rel.access_guard.InferenceAccessGuard")
    checkpoint = _argument(arguments, "--resume")
    if checkpoint is not None:
        receipt["resume_checkpoint"] = {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)}
    output = _argument(arguments, "--out")
    if output is not None:
        receipt["output_directory"] = str(Path(output).resolve())
    _dump(receipt_path, receipt)
    lock = threading.Lock()

    def on_spawn(original, effective, kwargs):
        row = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   original_command=original, effective_command=effective,
                   physical_gpu=kwargs.get("env", os.environ).get("CUDA_VISIBLE_DEVICES"))
        with lock:
            with receipt_path.with_suffix(".children.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    previous_argv = sys.argv
    target = None
    old_subprocess = None
    guard = None
    try:
        guard = install_guard()
        receipt["guard_diagnostics_at_start"] = guard.diagnostics()
        target = importlib.import_module(args.module)
        if hasattr(target, "subprocess"):
            old_subprocess = target.subprocess
            target.subprocess = RepairSubprocess(args.amendment, digest, resume_chain=args.resume_chain,
                                                 on_spawn=on_spawn, backend=old_subprocess)
        sys.argv = [args.module, *arguments]
        _dump(receipt_path, receipt)
        try:
            result = target.main()
        finally:
            verify_amendment(args.amendment, digest)
        receipt["status"] = "COMPLETE"
        return result
    except BaseException as exc:
        receipt.update(status="FAILED", error=repr(exc))
        raise
    finally:
        sys.argv = previous_argv
        if old_subprocess is not None:
            target.subprocess = old_subprocess
        if guard is not None:
            receipt["guard_diagnostics_at_exit"] = guard.diagnostics()
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(receipt_path, receipt)


if __name__ == "__main__":
    main()
