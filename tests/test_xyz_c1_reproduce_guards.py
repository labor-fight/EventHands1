"""CPU-only replay guards; synthetic release metadata never launches training."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "tools/xyz_c1_repro/reproduce.py"
TRAIN = REPO / "tools/xyz_c1_repro/train.py"
CONFIG = REPO / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"
spec = importlib.util.spec_from_file_location("c1_reproduce_guards", ENTRY)
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class ReplayGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.canonical_root = self.root / "outputs/xyz_c1_repro/20261018"
        self.canonical_root.mkdir(parents=True)
        self.original = yaml.safe_load(CONFIG.read_text())

    def fixture_reference(self):
        """Build isolated synthetic E/P/A, never a real complete6000 release."""
        wrapper = self.root / "tools/xyz_c1_repro/train.py"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_bytes(TRAIN.read_bytes())
        entry = wrapper.with_name("reproduce.py")
        entry.write_bytes(ENTRY.read_bytes())
        evidence = self.canonical_root / "preflight_evidence.json"
        reproduce.save(evidence, {"status": "SYNTHETIC_TEST_ONLY"})
        protocol = {"status": "FROZEN_READY", "runs": [], "file_sha256": {}, "immutable_data_stat": {},
                    "preflight_evidence_path": str(evidence.relative_to(self.root)),
                    "preflight_acceptance_sha256": reproduce.sha(evidence),
                    "math_policy": {"synthetic": True}, "model_seed": 3407}
        protocol_path = self.canonical_root / "protocol.json"
        reproduce.save(protocol_path, protocol)
        acceptance = {"status": "PASS", "preflight_evidence_sha256": reproduce.sha(evidence),
                      "bindings": {"wrapper_sha256": reproduce.sha(wrapper),
                                   "protocol_sha256": reproduce.sha(protocol_path)}}
        acceptance_path = self.canonical_root / "launch_acceptance.json"
        reproduce.save(acceptance_path, acceptance)
        status_path = self.canonical_root / "status.json"
        reproduce.save(status_path, {"status": "SYNTHETIC_TEST_ONLY"})
        full_path = self.canonical_root / "full_reproduction.json"
        reproduce.save(full_path, {"status": "SYNTHETIC_TEST_ONLY"})
        return {"protocol": protocol, "protocol_path": protocol_path, "acceptance_path": acceptance_path,
                "status_path": status_path, "full_path": full_path, "canonical": {"name": "fixture_canonical"}}, entry

    def test_not_ready_refuses_without_launch_or_output(self):
        output = self.root / "new"
        with mock.patch.object(reproduce, "fresh_output", return_value=output), \
                mock.patch.object(reproduce, "load_reference", side_effect=lambda: reproduce.require_ready({"status": "TRAINING", "baseline_ready": False})), \
                mock.patch.object(reproduce, "execute_replay") as run, \
                mock.patch.object(reproduce, "collect_runtime") as runtime:
            with self.assertRaisesRegex(RuntimeError, "not released"):
                reproduce.main(["--output-dir", str(output)])
            run.assert_not_called()
            runtime.assert_not_called()
        self.assertFalse(output.exists())

    def test_partial_ready_states_refuse(self):
        for value in ({}, {"status": "COMPLETE_REPRODUCIBLE_BASELINE_AND_ROWS", "baseline_ready": False},
                      {"status": "TRAINED", "baseline_ready": True}, {"status": "PASS", "baseline_ready": True}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                reproduce.require_ready(value)
        reproduce.require_ready({"status": "COMPLETE_REPRODUCIBLE_BASELINE_AND_ROWS", "baseline_ready": True})

    def test_gpu_pair_requires_distinct_ordered_two(self):
        self.assertEqual(reproduce.pair("0,1"), [0, 1])
        self.assertEqual(reproduce.pair("4,5"), [4, 5])
        for value in ("0", "0,0", "0,1,2", "0, 1", "-1,2", "cuda:0,cuda:1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reproduce.pair(value)

    def test_gpu_pair_must_match_preflighted_order(self):
        reference = {"protocol": {"resource_policy": {"physical_pairs": [[0, 1], [2, 3], [4, 5]]}}}
        for value in ("1,0", "6,7", "0,2"):
            with self.subTest(value=value), mock.patch.object(reproduce, "fresh_output"), \
                    mock.patch.object(reproduce, "load_reference", return_value=reference), \
                    mock.patch.object(reproduce, "collect_runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "preflighted"):
                    reproduce.main(["--output-dir", "unused", "--gpus", value])
                runtime.assert_not_called()

    def test_fresh_directory_only_inside_replays(self):
        output = self.canonical_root / "replays/my_replay_01"
        self.assertEqual(reproduce.fresh_output(output, self.canonical_root), output)
        self.assertFalse(output.exists())
        for value in (self.root / "outside", output / "nested", output.with_name("bad-name"), output.with_name("a" * 57)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reproduce.fresh_output(value, self.canonical_root)

    def test_existing_and_dangling_symlink_preserved(self):
        output = self.canonical_root / "replays/kept"
        output.mkdir(parents=True)
        sentinel = output / "evidence"
        sentinel.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            reproduce.fresh_output(output, self.canonical_root)
        self.assertEqual(sentinel.read_text(), "preserve me")
        dangling = output.with_name("dangling")
        dangling.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            reproduce.fresh_output(dangling, self.canonical_root)
        self.assertTrue(dangling.is_symlink())

    def test_symlink_replays_parent_refuses(self):
        other = self.root / "elsewhere"
        other.mkdir()
        (self.canonical_root / "replays").symlink_to(other, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            reproduce.fresh_output(self.canonical_root / "replays/new", self.canonical_root)

    def test_config_changes_only_three_identity_fields(self):
        original = copy.deepcopy(self.original)
        with mock.patch.object(reproduce, "ROOT", self.root):
            value = reproduce.replay_config(original, "new_run_s3407", self.canonical_root / "replays/fresh/run")
        self.assertEqual(original, self.original)
        changed = {(section, key) for section in original if isinstance(original[section], dict)
                   for key in original[section] if value[section][key] != original[section][key]}
        self.assertEqual(changed, set(reproduce.IDENTITY_CONFIG_FIELDS))
        self.assertEqual(reproduce.normalized_recipe(value), reproduce.normalized_recipe(original))

    def test_bad_seed_budget_batch_lr_encoder_source_refuse(self):
        for section, key, value in ((None, "SEED", 3408), ("TRAIN", "MAX_STEPS", 5999),
                                    ("TRAIN", "BATCH_SIZE_PER_GPU", 256), ("TRAIN", "DEVICES", 1),
                                    ("TRAIN", "LR", .001), ("MODEL", "ENCODER", "other"),
                                    ("MODEL", "XAG_ASSOC", "a1"), ("DATA", "XAG_SOURCE_ROOT", "cache")):
            cfg = copy.deepcopy(self.original)
            (cfg if section is None else cfg[section])[key] = value
            with self.subTest(section=section, key=key), mock.patch.object(reproduce, "ROOT", self.root):
                with self.assertRaisesRegex(RuntimeError, "recipe"):
                    reproduce.replay_config(cfg, "bad", self.canonical_root / "replays/bad/run")

    def test_overrides_and_abbreviations_rejected(self):
        for option in (("--seed", "3408"), ("--lr", ".001"), ("--resume", "old.ckpt"), ("--output", "bad")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(reproduce, "load_reference") as load:
                with self.assertRaises(SystemExit):
                    reproduce.main(["--output-dir", "unused", *option])
                load.assert_not_called()

    def test_environment_uuid_driver_version_drift_refuses(self):
        expected = {"python_full": "3.9", "packages": {"torch": "2.1"},
                    "gpu_inventory": [{"uuid": "old", "driver_version": "580"}], "scope": "description"}
        reproduce.compare_runtime(expected, dict(expected, scope="new description"))
        for field, value in (("python_full", "3.10"), ("packages", {"torch": "2.2"}),
                             ("gpu_inventory", [{"uuid": "new", "driver_version": "580"}]),
                             ("gpu_inventory", [{"uuid": "old", "driver_version": "590"}])):
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "environment/hardware"):
                reproduce.compare_runtime(expected, dict(expected, **{field: value}))

    def test_busy_gpu_refuses(self):
        result = SimpleNamespace(stdout="GPU-selected, 123456789, training\nGPU-other, 2, training\n")
        with mock.patch.object(reproduce.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                reproduce.ensure_idle([0, 1], [{"index": "0", "uuid": "GPU-selected"}, {"index": "1", "uuid": "GPU-second"}])
            reproduce.ensure_idle([0, 1], [{"index": "0", "uuid": "GPU-free"}, {"index": "1", "uuid": "GPU-second"}])

    def test_exited_launcher_does_not_leave_owned_child(self):
        # A CPU child emulates an orphaned DDP rank. The launcher exits before
        # cleanup and the child ignores TERM, so cleanup must reach KILL.
        pidfile = self.root / "owned_child.pid"
        child = ("import os,signal,time;from pathlib import Path;"
                 "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                 f"Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)")
        launcher = ("import subprocess,sys,time;from pathlib import Path;"
                    f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);"
                    f"path=Path({str(pidfile)!r});"
                    "\nwhile not path.exists():time.sleep(.01)\n")
        process = subprocess.Popen([sys.executable, "-c", launcher], start_new_session=True)
        try:
            self.assertEqual(process.wait(timeout=10), 0)
            child_pid = int(pidfile.read_text())
            self.assertEqual(os.getpgid(child_pid), process.pid)
            reproduce.stop_owned([process], timeout=.2)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{child_pid}/stat")
                if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    break
                time.sleep(.02)
            else:
                self.fail("Owned child survived its exited launcher's cleanup")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def test_frozen_source_and_data_changes_refuse(self):
        source, data = self.root / "source", self.root / "data"
        source.write_bytes(b"source")
        data.write_bytes(b"data")
        stat = data.stat()
        protocol = {"file_sha256": {"source": reproduce.sha(source)},
                    "immutable_data_stat": {str(data): {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}}
        with mock.patch.object(reproduce, "ROOT", self.root):
            reproduce.verify_sources(protocol)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Frozen source"):
                reproduce.verify_sources(protocol)
            source.write_bytes(b"source")
            data.write_bytes(b"newdata")
            with self.assertRaisesRegex(RuntimeError, "Frozen data"):
                reproduce.verify_sources(protocol)

    def test_new_bundle_passes_sealed_formal_binding_and_detects_mutation(self):
        reference, entry = self.fixture_reference()
        output = self.canonical_root / "replays/schema_test"
        with mock.patch.object(reproduce, "ROOT", self.root), mock.patch.object(reproduce, "__file__", str(entry)):
            run, protocol, protocol_path, acceptance_path, _ = reproduce.prepare_bundle(reference, output, [0, 1], self.original)
            reproduce.verify_sources(protocol)
            self.assertEqual(protocol["runs"], [run])
            self.assertEqual(run["sampler"], "legacy_none")
            self.assertEqual(run["fresh_steps"], 6000)
            self.assertEqual(protocol["preflight_acceptance_sha256"], reference["protocol"]["preflight_acceptance_sha256"])
            self.assertEqual(protocol["math_policy"], reference["protocol"]["math_policy"])
            self.assertEqual(reproduce.normalized_recipe(yaml.safe_load((output / "config.yaml").read_text())),
                             reproduce.normalized_recipe(self.original))
            # Execute the unchanged, isolated sealed binding function without
            # importing its torch/DDP startup or creating a real acceptance.
            node = next(n for n in ast.parse(TRAIN.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "check_formal_binding")
            namespace = {"Path": Path, "json": json, "file_sha": reproduce.sha,
                         "__file__": str(self.root / "tools/xyz_c1_repro/train.py")}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
            options = SimpleNamespace(protocol=protocol_path, acceptance=acceptance_path, sampler="legacy_none")
            config_path = self.root / run["config"]
            namespace["check_formal_binding"](options, config_path)
            options.sampler = "dedicated"
            with self.assertRaisesRegex(ValueError, "sampler"):
                namespace["check_formal_binding"](options, config_path)
            options.sampler = "legacy_none"
            config_path.write_text(config_path.read_text() + "\n# mutated\n")
            with self.assertRaisesRegex(ValueError, "config"):
                namespace["check_formal_binding"](options, config_path)

    def test_invalid_recipe_makes_no_directory(self):
        reference, entry = self.fixture_reference()
        output = self.canonical_root / "replays/schema_bad"
        self.original["SEED"] = 3408
        with mock.patch.object(reproduce, "ROOT", self.root), mock.patch.object(reproduce, "__file__", str(entry)):
            with self.assertRaises(RuntimeError):
                reproduce.prepare_bundle(reference, output, [0, 1], self.original)
        self.assertFalse(output.exists())

    def test_acceptance_evidence_source_and_protocol_mutation_refuse(self):
        reference, _ = self.fixture_reference()
        protocol = reference["protocol"]
        acceptance = reproduce.read(reference["acceptance_path"])
        with mock.patch.object(reproduce, "ROOT", self.root):
            reproduce.validate_acceptance(protocol, reference["protocol_path"], acceptance)
            for field in ("protocol_sha256", "wrapper_sha256"):
                changed = copy.deepcopy(acceptance)
                changed["bindings"][field] = "0" * 64
                with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "E -> P -> A"):
                    reproduce.validate_acceptance(protocol, reference["protocol_path"], changed)
            (self.root / protocol["preflight_evidence_path"]).write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "E -> P -> A"):
                reproduce.validate_acceptance(protocol, reference["protocol_path"], acceptance)

    def test_full_trace_uses_complete_inventory_and_byte_equality(self):
        def arrays():
            return {seq + "/" + name: np.ones((n, 51), dtype=np.float32)
                    for seq, n in (("zgz_global", 1386), ("zgz_local", 1204)) for name in ("incoming", "prediction")}
        class Trace(dict):
            @property
            def files(self):
                return list(self)
        a, b = Trace(arrays()), Trace(arrays())
        reproduce.compare_traces(a, b)
        b["zgz_local/prediction"][-1, -1] = np.nextafter(np.float32(1), np.float32(2))
        with self.assertRaisesRegex(RuntimeError, "differs"):
            reproduce.compare_traces(a, b)
        b = Trace(arrays())
        del b["zgz_global/incoming"]
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            reproduce.compare_traces(a, b)
        b = Trace(arrays())
        b["zgz_global/prediction"] = b["zgz_global/prediction"][:-1]
        with self.assertRaisesRegex(RuntimeError, "differs"):
            reproduce.compare_traces(a, b)


if __name__ == "__main__":
    unittest.main()
