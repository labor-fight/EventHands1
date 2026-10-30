"""Separate launcher provenance and explicit repair propagation; no real jobs."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.s43_revisit import s43_field_hook_repair as R


def amendment_fixture(tmp_path, monkeypatch):
    guard = tmp_path / "guard.py"
    guard.write_text("# fixture replacement guard\n")
    monkeypatch.setattr(R, "GUARD_SOURCE", guard)
    path = tmp_path / "amendment.json"
    value = dict(repair_sources={str(Path(R.__file__).resolve()): R.sha256(R.__file__),
                                str(guard.resolve()): R.sha256(guard)},
                 original_sourcefreeze={"path": "/fixture/original.json", "sha256": "original"})
    path.write_text(json.dumps(value))
    return path, guard


def test_queue_resume_rewrite_preserves_source_and_propagates_repair(tmp_path):
    amendment = tmp_path / "amendment.json"
    old = [sys.executable, "-u", "-m", R.QUEUE_MODULE, "--pilot-controller-pid", "42",
           "--source-freeze", "oldfreeze.json"]
    wrapped = R.repair_command(old, amendment, resume_chain=True, amendment_sha256="sealed")
    assert wrapped[:4] == [sys.executable, "-u", "-m", R.MODULE]
    assert wrapped[wrapped.index("--module") + 1] == R.QUEUE_MODULE
    assert "--pilot-controller-pid" not in wrapped
    assert wrapped.count("--resume-existing") == 1
    assert "--resume-chain" in wrapped
    assert wrapped[wrapped.index("--amendment-sha256") + 1] == "sealed"
    assert wrapped[wrapped.index("--source-freeze") + 1] == "oldfreeze.json"
    assert old[4:6] == ["--pilot-controller-pid", "42"]
    assert R.repair_command(wrapped, amendment, resume_chain=True) == wrapped


def test_only_study_python_modules_are_wrapped_and_fresh_queue_stays_fresh(tmp_path):
    path = tmp_path / "amendment.json"
    commands = [["echo", "-m", R.QUEUE_MODULE], [sys.executable, "-m", "json.tool"],
                ["nvidia-smi"], "python -m tools.s43_revisit.s43_field_full_queue"]
    for command in commands:
        assert R.repair_command(command, path, resume_chain=True) == command
    old = [sys.executable, "-m", R.QUEUE_MODULE, "--pilot-controller-pid=42"]
    fresh = R.repair_command(old, path)
    assert "--pilot-controller-pid=42" in fresh and "--resume-existing" not in fresh
    resumed = R.repair_command(old, path, resume_chain=True)
    assert "--pilot-controller-pid=42" not in resumed and "--resume-existing" in resumed
    trainer = [sys.executable, "-m", "tools.s43_revisit.s43_field_full_training", "train", "--resume", "last.ckpt"]
    assert R.repair_command(trainer, path, resume_chain=True)[-3:] == ["train", "--resume", "last.ckpt"]


def test_source_or_amendment_changes_fail_closed(tmp_path, monkeypatch):
    path, guard = amendment_fixture(tmp_path, monkeypatch)
    _, digest = R.verify_amendment(path)
    guard.write_text("# changed\n")
    with pytest.raises(RuntimeError, match="repair source changed"):
        R.verify_amendment(path, digest)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="amendment identity changed"):
        R.verify_amendment(path, digest)


def test_installer_updates_existing_aliases_without_changing_files(monkeypatch):
    legacy = ModuleType("tools.s43_rel.mean_shape_replay")
    original = type("OldGuard", (), {})
    legacy.InferenceAccessGuard = original
    replacement = ModuleType("tools.s43_rel.access_guard")
    new_guard = type("NewGuard", (), {})
    replacement.InferenceAccessGuard = new_guard
    calls = []
    replacement._ensure_audit_hook = lambda: calls.append("ensure")
    alias = ModuleType("tools.s43_revisit.fixture_alias")
    alias.Guard = original
    for module in (legacy, replacement, alias):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    assert R.install_guard() is replacement
    assert legacy.InferenceAccessGuard is new_guard and alias.Guard is new_guard
    assert R.install_guard() is replacement
    assert calls == ["ensure", "ensure"]


def test_proxy_propagates_pinned_amendment_to_popen_and_run(tmp_path, monkeypatch):
    path, _ = amendment_fixture(tmp_path, monkeypatch)
    _, digest = R.verify_amendment(path)
    calls = []
    backend = SimpleNamespace(Popen=lambda cmd, *a, **kw: calls.append(("popen", cmd, kw)),
                              run=lambda cmd, *a, **kw: calls.append(("run", cmd, kw)), STDOUT=subprocess.STDOUT)
    proxy = R.RepairSubprocess(path, digest, resume_chain=True, backend=backend)
    original = [sys.executable, "-m", "tools.s43_revisit.s43_field_full_training", "train", "--resume", "last.ckpt"]
    proxy.Popen(original, env={"CUDA_VISIBLE_DEVICES": "4"})
    proxy.run(original, check=True)
    assert proxy.STDOUT == subprocess.STDOUT
    for kind, command, kwargs in calls:
        assert command[2] == R.MODULE
        assert command[command.index("--amendment-sha256") + 1] == digest
        assert command[-2:] == ["--resume", "last.ckpt"]
    assert calls[0][2]["env"] == {"CUDA_VISIBLE_DEVICES": "4"}
    assert calls[1][2]["check"] is True


@pytest.mark.parametrize("change_source", [False, True])
def test_main_receipt_preserves_resume_identity_and_restores_process_state(tmp_path, monkeypatch, change_source):
    path, source = amendment_fixture(tmp_path, monkeypatch)
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint, never loaded")
    target = ModuleType("tools.s43_revisit.fixture_target")
    backend = SimpleNamespace(Popen=lambda command, **kwargs: "fixture process")
    target.subprocess = backend
    argv = sys.argv
    guards = []
    monkeypatch.setattr(R, "install_guard", lambda: SimpleNamespace(diagnostics=lambda: {"registered": True}))
    def execute():
        assert target.subprocess is not backend
        assert sys.argv == [target.__name__, "train", "--resume", str(checkpoint), "--out", str(tmp_path / "out")]
        guards.append(target.subprocess.Popen([sys.executable, "-m", R.QUEUE_MODULE, "--pilot-controller-pid", "42"]))
        if change_source:
            source.write_text("# mutated during target\n")
    target.main = execute
    monkeypatch.setitem(sys.modules, target.__name__, target)
    command = ["--amendment", str(path), "--resume-chain", "--module", target.__name__,
               "train", "--resume", str(checkpoint), "--out", str(tmp_path / "out")]
    if change_source:
        with pytest.raises(RuntimeError, match="repair source changed"):
            R.main(command)
    else:
        R.main(command)
    assert target.subprocess is backend and sys.argv is argv
    receipt_file = next((tmp_path / "launches").glob("*.json"))
    receipt = json.loads(receipt_file.read_text())
    assert receipt["status"] == ("FAILED" if change_source else "COMPLETE")
    assert receipt["resume_checkpoint"]["sha256"] == R.sha256(checkpoint)
    assert receipt["original_sourcefreeze"]["sha256"] == "original"
    assert receipt["guard_diagnostics_at_start"] == {"registered": True}
    children = [json.loads(line) for line in receipt_file.with_suffix(".children.jsonl").read_text().splitlines()]
    assert children[0]["effective_command"][2] == R.MODULE
    assert "--resume-existing" in children[0]["effective_command"]
