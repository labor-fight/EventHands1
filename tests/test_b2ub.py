"""B2UB CPU tests (docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104). Run with CUDA_VISIBLE_DEVICES="".

- B2-gt's configs differ from B2's (PFO2 fgs, seed 3407) only in the registered keys; its evaluation config names B2's
  deployed router and B2's protocol noise.
- The clamp feed-back replaces exactly the clamped blocks.
- tools/b2ub/engine.py `e1c` against PFO's engine (tools/pfo/engine.py) on a fake network: "none" reproduces `e1`
  bitwise, "TRF" reproduces `e3` at every endpoint but the first of each segment, a partial clamp feeds back the GT block.
- The job list is well formed (every dependency is an earlier job, the final readout waits for every reading).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.b2ub import arms as A  # noqa: E402
from tools.b2ub import engine as BE  # noqa: E402

REGISTERED = {"DATA.XAG_SOURCE_ROOT", "EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "PFO.ARM", "PFO.BASE_ARM",
              "PFO.EVAL_ROUTER", "PFO.EVAL_ROUTER_CHECKPOINT", "PFO.EVAL_ROUTER_KIND", "PFO.ROUTER", "PFO.ROUTER_CHECKPOINT",
              "PFO.ROUTER_KIND", "PFO.SECTION", "PFO.STUDY", "PFO.TRAIN_ROUTES"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


@pytest.mark.parametrize("phase", A.PHASES)
def test_b2gt_config_is_b2_plus_label_routes(phase):
    from tools.pfo2 import arms as P2
    a, b = flat(A.build_config("b2gt", 3407, phase)), flat(P2.build_config("fgs", 3407, phase))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert diff == REGISTERED
    assert a["PFO.ROUTER"] == "label" and a["PFO.EVAL_ROUTER"] == b["PFO.ROUTER"] == "r_s2d"
    assert a["PFO.EVAL_ROUTER_CHECKPOINT"] == b["PFO.ROUTER_CHECKPOINT"]
    assert a["DATA.XAG_SOURCE_ROOT"].endswith("outputs/xag/sources")


def test_b2gt_eval_config_names_b2s_router():
    import yaml
    from tools.pfo2.run import _eval_cfg
    mine = _eval_cfg(A.build_config("b2gt", 3407), "x")
    b2 = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
    assert {k: v for k, v in mine["S45SM"].items() if k != "ARM"} == {k: v for k, v in b2["S45SM"].items() if k != "ARM"}
    assert mine["TRACK"] == b2["TRACK"] and mine["MODEL"] == b2["MODEL"] and mine["LOSS"] == b2["LOSS"]


def test_feed_back_blocks():
    pred = np.arange(51, dtype=np.float32)
    truth = -np.arange(51, dtype=np.float32) - 1
    assert BE.clamp_slices("none") == ()
    out = BE.feed_back(pred, truth, BE.clamp_slices("R"))
    assert out.dtype == np.float32 and np.array_equal(out[3:6], truth[3:6])
    assert np.array_equal(out[:3], pred[:3]) and np.array_equal(out[6:], pred[6:])
    assert np.array_equal(BE.feed_back(pred, truth, BE.clamp_slices("TRF")), truth)
    out = BE.feed_back(pred, truth, BE.clamp_slices("TF"))
    assert np.array_equal(out[:3], truth[:3]) and np.array_equal(out[3:6], pred[3:6]) and np.array_equal(out[6:], truth[6:])
    with pytest.raises(RuntimeError):
        BE.clamp_slices("X")


# --------------------------------------------------------------------------- fake network for the engine comparison
class FakeNet:
    def __init__(self, ckpt, cfg_path):
        self.cfg = {"MANO": {"NPZ": "unused"}, "TRACK": {"PREV_NOISE_T": 0.005, "PREV_NOISE_R": 0.05, "PREV_NOISE_POSE": 0.05}}
        self.identity = {"net": "fake", "kind": "routed", "checkpoint": str(ckpt), "config": str(cfg_path)}

    def bind(self, ctx):
        pass

    def step(self, ctx, end, prev):
        prev = np.ascontiguousarray(prev, np.float32)
        out = np.float32(0.6) * prev + np.float32(0.4) * ctx.gt[int(end)] + np.float32(1e-3) * np.sin(
            np.float32(end) + np.arange(51, dtype=np.float32))
        return out.astype(np.float32)


def install_fakes(monkeypatch, tmp_path):
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from semkine import eval_track as ET
    ckpt, cfgp = tmp_path / "net.ckpt", tmp_path / "net.yaml"
    ckpt.write_bytes(b"fake")
    cfgp.write_text("x: 1\n")
    rng = np.random.default_rng(7)
    gt = rng.standard_normal((3000, 51)).astype(np.float32) * np.float32(0.1)
    runs = {"fake_a": np.array([[0, 520], [700, 1210]]), "fake_b": np.array([[100, 760]])}

    def sequences(cfg, names):
        return Path("/nonexistent"), [(s, s) for s in names]

    def sequence(root, directory, seq, device):
        return types.SimpleNamespace(seq=seq, gt=gt, aux={"valid_runs_ms": runs[seq], "betas": np.zeros(10, np.float32)})

    def endpoint_rows(aux):
        rows = []
        for rid, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)):
            prev = int(a)
            for end in np.arange(a + 49, b, 50):
                rows.append((rid, int(end), prev, 0))
                prev = int(end)
        return rows

    def score(mano, pred, target, betas, device):
        d = np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64))
        return {k: d.mean(1) for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "wrist_mm", "root_deg",
                                       "finger_deg")}
    monkeypatch.setattr(E, "sequences", sequences)
    monkeypatch.setattr(E, "endpoint_rows", endpoint_rows)
    monkeypatch.setattr(E, "score", score)
    monkeypatch.setattr(N, "sequence", sequence)
    monkeypatch.setattr(N, "load", lambda *a, **k: FakeNet(ckpt, cfgp))
    monkeypatch.setattr(ET, "sample_init_noise", lambda cfg, rng, s: rng.standard_normal(51).astype(np.float32) * 0.01)
    fake = types.ModuleType("mano_layer")

    class ManoLayer:
        def __init__(self, *a, **k):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self
    fake.ManoLayer = ManoLayer
    monkeypatch.setitem(sys.modules, "mano_layer", fake)
    spec = tmp_path / "spec.json"
    spec.write_text('{"kind": "routed", "checkpoint": "%s", "config": "%s"}' % (ckpt, cfgp))
    monkeypatch.setenv("PFO_SMOKE_DIR", str(tmp_path / "smoke"))
    monkeypatch.setenv("PFO_SMOKE_SEQS", "fake_a,fake_b")
    return E, spec


def test_e1c_matches_pfo_engine(monkeypatch, tmp_path):
    E, spec = install_fakes(monkeypatch, tmp_path)
    big = 10 ** 6
    e1 = np.load(E.run("fake", "e1", None, None, (), str(spec), limit=big) / "states.npz")
    e3 = np.load(E.run("fake", "e3", None, None, (), str(spec), limit=big) / "states.npz")
    none = np.load(BE.run("fake", str(spec), "none", limit=big) / "states.npz")
    trf = np.load(BE.run("fake", str(spec), "TRF", limit=big) / "states.npz")
    r = np.load(BE.run("fake", str(spec), "R", limit=big) / "states.npz")
    for s in ("fake_a", "fake_b"):
        for key in ("prediction", "incoming", "target", "gt_previous", "end_ms", "run_id", "elapsed_ms"):
            assert np.array_equal(none[f"{s}/{key}"], e1[f"{s}/{key}"]), (s, key)
        rid = trf[f"{s}/run_id"]
        first = np.r_[True, rid[1:] != rid[:-1]]
        assert first.sum() == len(np.unique(rid)) and (~first).sum() > 5
        assert np.array_equal(trf[f"{s}/end_ms"], e3[f"{s}/end_ms"])
        assert np.array_equal(trf[f"{s}/prediction"][~first], e3[f"{s}/prediction"][~first])
        assert not np.array_equal(trf[f"{s}/prediction"][first], e3[f"{s}/prediction"][first])
        # the R clamp feeds back the GT root rotation of the previous endpoint, the rest is the own prediction
        inc, pred, gtp = r[f"{s}/incoming"], r[f"{s}/prediction"], r[f"{s}/gt_previous"]
        rid = r[f"{s}/run_id"]
        nxt = np.r_[False, rid[1:] == rid[:-1]]
        idx = np.nonzero(nxt)[0]
        assert np.array_equal(inc[idx, 3:6], gtp[idx, 3:6])
        assert np.array_equal(inc[idx, :3], pred[idx - 1, :3]) and np.array_equal(inc[idx, 6:], pred[idx - 1, 6:])


PFO2_READ = ROOT / "outputs/pfo2/20261103/readings"


@pytest.mark.skipif(not (PFO2_READ / "pfo2_fgs_s3409/f1/report.json").exists(), reason="needs PFO2's B2 readings")
def test_analyze_runs_end_to_end_on_stand_in_readings(tmp_path, monkeypatch):
    """Every tag the readout reads, as a symlink to B2's own PFO2 reading of the same task (numbers are meaningless:
    the test only exercises the code path, the tables and the main-row writer)."""
    from tools.b2ub import analyze as AN
    from tools.b2ub.engine import CLAMPS
    out = tmp_path / "b2ub"

    def link(tag, task, seed, src_task=None):
        d = out / "readings" / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / task).symlink_to(PFO2_READ / f"pfo2_fgs_s{seed}" / (src_task or task))
    for seed in (3407, 3408, 3409):
        for route in ("label", "label_loc", "label_part", "label_onhand"):
            for task in ("e1", "e3", "pert") + (("f1",) if route == "label" else ()):
                link(f"b2_s{seed}__{route}", task, seed)
        for c in CLAMPS:
            link(f"b2_s{seed}__clamp-{c}", "e1c", seed, "e1")
            link(f"b2_s{seed}__label__clamp-{c}", "e1c", seed, "e1")
    for task in ("e1", "e3", "pert", "f1", "tl_e1"):
        link("b2gt_s3407__label", task, 3407)
    for st in (5000, 5500):
        link(f"b2gt_s3407@{st}__label", "e1", 3407)
    for task in ("e1", "e3", "pert"):
        link("b2gt_s3407", task, 3407)
    for c in CLAMPS:
        link(f"b2gt_s3407__label__clamp-{c}", "e1c", 3407, "e1")
    monkeypatch.setattr(AN, "OUT", out)
    monkeypatch.setattr(AN, "SEMKINE", tmp_path / "semkine")
    AN.all_()
    s = __import__("json").loads((out / "decisions/summary.json").read_text())
    assert s["u1"]["verdict"] in ("TEST_ROUTE_SUFFICES", "TEST_ROUTE_INSUFFICIENT")
    assert s["u3"]["verdict"] in ("BOUND_BELOW_S43", "BOUND_NOT_BELOW_S43")
    assert abs(s["u1_routes"]["deployed"]["five_mean"][4] - 11.57) < 0.01          # B2's three-seed RA
    for name in ("u1", "u2", "u3", "bound"):
        assert (out / "analysis" / f"{name}.md").read_text().startswith("## ")
    for name in ("b2label", "b2gt_label", "b2gt_router"):
        assert (tmp_path / "semkine" / f"b2ub_{name}_main_row.json").exists()


def test_jobs_well_formed(tmp_path, monkeypatch):
    from tools.b2ub import jobs as J
    from tools.b2ub import arms as AA
    monkeypatch.setattr(AA, "CONFIGS", tmp_path)
    monkeypatch.setattr(J, "write_config", lambda arm, seed, phase: tmp_path / f"{arm}_{seed}_{phase}.yaml")
    jobs = J.r1()
    keys = [j["key"] for j in jobs]
    final = jobs[-1]
    assert final["key"] == "final:analyze"
    readings = [j["key"] for j in jobs if j["key"].startswith(("u1:", "u2:", "u3:", "u3c:", "k:k4", "k:u3k4"))
                and j["kind"] in ("gpu", "cpu") and j["key"] not in ("u3:pre:accept", "u3:accept")]
    assert set(readings) <= set(final["deps"])
    assert sum(j["kind"] == "pair" for j in jobs) == 2
    assert all(j["mem_gib"] == 260 for j in jobs if j["kind"] == "pair")
    u2 = [k for k in keys if k.startswith(("u2:", "k:k3:"))]
    assert len(u2) == 2 * 8 * 3
    assert len([k for k in keys if k.startswith("u1:")]) == 3 * 4 * 3 + 3


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(pytest.main([__file__, "-q"]))
