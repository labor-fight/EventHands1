"""Contract tests for XCL state modes and frozen-weight interventions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from tools.xcl.evaluate import array_digest, compare_traces, endpoint_grid, resolve_checkpoint, resolve_run, select_previous
from tools.xcl.interventions import apply
from tools.xcl.report import direction_support
from tools.xcl.evaluate_all import jobs


class StateContractTests(unittest.TestCase):
    def test_clamps_only_the_named_coordinates(self):
        raw = np.arange(51, dtype=np.float32)
        truth = raw + 100
        pose = select_previous("gt_pose", raw, truth)
        translation = select_previous("gt_transl", raw, truth)
        np.testing.assert_array_equal(pose[:3], raw[:3])
        np.testing.assert_array_equal(pose[3:], truth[3:])
        np.testing.assert_array_equal(translation[:3], truth[:3])
        np.testing.assert_array_equal(translation[3:], raw[3:])
        np.testing.assert_array_equal(raw, np.arange(51, dtype=np.float32))

    def test_common_and_gt_are_distinct(self):
        raw = np.zeros(51, np.float32); truth = raw + 1; common = raw + 2
        np.testing.assert_array_equal(select_previous("e2", raw, truth, common), common)
        np.testing.assert_array_equal(select_previous("e3", raw, truth), truth)
        with self.assertRaises(ValueError):
            select_previous("e2", raw, truth)

    def test_empty_event_packet_has_stable_fingerprint(self):
        empty = np.zeros((0, 5), np.float32)
        self.assertEqual(array_digest(empty), array_digest(empty.copy()))
        self.assertNotEqual(array_digest(empty), array_digest(np.zeros((0, 6), np.float32)))

    def test_clock_keeps_gap_resets(self):
        rows = endpoint_grid({"valid_runs_ms": np.array([[0, 100], [250, 350]])})
        self.assertEqual(rows, [(0, 49, 0, True), (0, 99, 49, False),
                                (1, 299, 250, True), (1, 349, 299, False)])

    def test_late_checkpoint_ambiguity_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x-step=5000.ckpt").write_bytes(b"one")
            (root / "x-step=5000-v1.ckpt").write_bytes(b"two")
            with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                resolve_checkpoint({"output": directory, "config": "unused"}, 5000)

    def test_single_registered_identity_required(self):
        with self.assertRaises(ValueError):
            resolve_run({"runs": [{"name": "x", "cell": "a"}, {"name": "y", "cell": "a"}]}, "a")

    def test_fresh_comparison_checks_dtype_and_all_values(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.npz", Path(directory) / "b.npz"
            np.savez(a, prediction=np.zeros((2590, 51), np.float32))
            np.savez(b, prediction=np.zeros((2590, 51), np.float64))
            with self.assertRaises(RuntimeError):
                compare_traces(a, b)
            np.savez(b, prediction=np.zeros((2590, 51), np.float32))
            self.assertEqual(compare_traces(a, b)["status"], "PASS")

    def test_descriptive_threshold_requires_all_three_and_late_signs(self):
        self.assertEqual(direction_support([-6, -1, -2], [-1, 2, -6], 5), "方向支持")
        self.assertEqual(direction_support([-6, 1, -2], [-1, -2, -6], 5), "噪声带内")
        self.assertEqual(direction_support([-4, -1, -2], [-1, -2, -4], 5), "噪声带内")
        self.assertEqual(direction_support([-6, -1, -2], [None, -2, -6], 5), "缺读数")

    def test_fixed_evaluation_queue_covers_all_registered_work(self):
        cells = ("a0f0", "a0f1", "a1f0", "a1f1", "a2f1", "a2f0", "a3f0", "c0", "a0fod", "a0foo")
        protocol = {"runs": [{"name": cell, "cell": cell} for cell in cells]}
        rows, traces = jobs(protocol, Path("/tmp/evaluations"))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(job["mode"] == "row" for job in rows))
        self.assertEqual(len(traces), 124)
        self.assertEqual(len({job["output"] for job in rows + traces}), 134)
        for cell in cells:
            own = [job for job in traces if job["cell"] == cell and job["intervention"] == "none"]
            self.assertEqual(len(own), 8)
        crossings = [job for job in traces if job["intervention"].startswith("route_")]
        self.assertEqual(len(crossings), 8)
        self.assertEqual({job["cell"] for job in crossings}, {"a0f0", "a0f1"})
        self.assertTrue(all(job["mode"] in ("e2", "e3") for job in crossings))


class InterventionTests(unittest.TestCase):
    def model(self):
        model = nn.Module(); model.xyz = nn.Module()
        model.xyz.rel = nn.Sequential(nn.Identity())
        model.xyz.slot_proj = nn.Conv1d(17 * 4, 17 * 2, 1, groups=17)
        model.xyz.xag_h3_transform = None
        model.xag_assoc_mode = "base"
        return model

    def test_relation_channels_zero_and_hook_is_removed(self):
        model = self.model(); features = torch.arange(13.).view(1, 13)
        with apply(model, "geometry_zero"):
            output = model.xyz.rel(features)
        self.assertTrue(torch.equal(output[:, :7], features[:, :7]))
        self.assertTrue(torch.equal(output[:, 7:11], torch.zeros(1, 4)))
        self.assertTrue(torch.equal(output[:, 11:], features[:, 11:]))
        self.assertTrue(torch.equal(model.xyz.rel(features), features))

    def test_hooks_restore_on_failure(self):
        model = self.model()
        with self.assertRaises(RuntimeError):
            with apply(model, "route_a3"):
                self.assertEqual(model.xag_assoc_mode, "full_oracle")
                raise RuntimeError("probe failure")
        self.assertEqual(model.xag_assoc_mode, "base")

    def test_residual_slot_only(self):
        model = self.model(); inputs = []
        hook = model.xyz.slot_proj.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
        try:
            with apply(model, "residual_zero"):
                model.xyz.slot_proj(torch.ones(1, 68, 1))
            model.xyz.slot_proj(torch.ones(1, 68, 1))
        finally:
            hook.remove()
        # Observation hook precedes intervention, so inspect output via second hook.
        with apply(model, "residual_zero"):
            seen = []
            hook = model.xyz.slot_proj.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
            model.xyz.slot_proj(torch.ones(1, 68, 1)); hook.remove()
        self.assertTrue(torch.equal(seen[0][:, :64], torch.ones(1, 64, 1)))
        self.assertTrue(torch.equal(seen[0][:, 64:], torch.zeros(1, 4, 1)))


if __name__ == "__main__":
    unittest.main()
