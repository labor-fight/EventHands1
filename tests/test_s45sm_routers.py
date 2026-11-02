"""CPU tests: router training pieces, annealing, decoding variants and the router selections of the consolidated S45-SM
package (ported from the round 2 / 4 / 6 / 8 / 10 / 12 tests and the selection rules of rounds 4, 5, 6, 8, 6c)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45rt_router as RT  # noqa: E402
from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_dw as DW  # noqa: E402
from semkine import s45sm_hier as H  # noqa: E402
from semkine import s45sm_srp as SRP  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y, EventPacketBatch  # noqa: E402
from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402
from tools.s45sm import router_train as RTR  # noqa: E402
from tools.s45sm import selection as SEL  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"
ANNEALED = {"srpkda": "srpkd", "sr1a": "sr1", "srdwkda": "srdwkd"}
DECODED = {"srpkdh": "srpkd", "srpkdah": "srpkda", "srdwkdh": "srdwkd", "srdwkdah": "srdwkda", "t0ph": "t0p"}
ROUTERS12 = {"srpkd9": None, "srpkd9a": "anneal", "srpkd9h": "decode", "srpkd9ah": "decode"}


def batch(seed=0, n_per=(500, 0, 300, 800)):
    g = torch.Generator().manual_seed(seed)
    rows, ptr = [], [0]
    for b, n in enumerate(n_per):
        ev = torch.zeros(n, 5)
        ev[:, EV_BATCH] = b
        ev[:, EV_X] = torch.randint(60, 180, (n,), generator=g).float()
        ev[:, EV_Y] = torch.randint(40, 140, (n,), generator=g).float()
        ev[:, EV_T] = torch.sort(torch.rand(n, generator=g) * 0.05).values
        ev[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        rows.append(ev)
        ptr.append(ptr[-1] + n)
    ev = torch.cat(rows)
    N, B = ev.shape[0], len(n_per)
    z = torch.zeros(B)
    return EventPacketBatch(events=ev, ptr=torch.tensor(ptr), sequence_id=torch.zeros(B, dtype=torch.long),
                            t_start_us=torch.zeros(B, dtype=torch.long), t_end_us=torch.full((B,), 50000),
                            delta_t_s=z + 0.05, is_sequence_start=torch.zeros(B, dtype=torch.bool),
                            is_sequence_end=torch.zeros(B, dtype=torch.bool), target=torch.zeros(B, 51),
                            prev_state=torch.zeros(B, 51), betas=torch.zeros(B, 10), camera_K=torch.eye(3).repeat(B, 1, 1),
                            xag_source_index=torch.arange(N), xag_source_ms=torch.arange(N), xag_part=torch.arange(N) % 16,
                            xag_face=torch.arange(N) % 1538, xag_bary=torch.rand(N, 3, generator=g),
                            xag_valid=torch.ones(N, dtype=torch.bool))


# --------------------------------------------------------------------------- round 2: router-only augmentation
def test_augment_keeps_fields_aligned():
    b = batch()
    nb = RTR.augment(b, {"rot_deg": 60, "scale": [0.75, 1.33], "keep": [0.3, 1.0]}, torch.Generator().manual_seed(4))
    nb.validate()
    n = nb.events.shape[0]
    assert 0 < n < b.events.shape[0]
    for name in RTR.XAG_FIELDS:
        assert getattr(nb, name).shape[0] == n
    idx = nb.xag_source_index                                  # every kept event keeps its own label row
    assert torch.equal(nb.xag_face, b.xag_face[idx]) and torch.equal(nb.events[:, EV_T], b.events[idx, EV_T])
    assert torch.equal(nb.events[:, EV_BATCH], b.events[idx, EV_BATCH])
    assert torch.equal(nb.ptr[1:] - nb.ptr[:-1], torch.bincount(nb.events[:, EV_BATCH].long(), minlength=4))
    assert float(nb.events[:, EV_X].max()) <= 239 and float(nb.events[:, EV_Y].max()) <= 179


def test_identity_augment_is_exact():
    b = batch(1)
    nb = RTR.augment(b, {"rot_deg": 0, "scale": [1, 1], "keep": [1.0, 1.0001]}, torch.Generator().manual_seed(0))
    for k, v in b.__dict__.items():
        if torch.is_tensor(v):
            assert torch.equal(v, getattr(nb, k)), k


# --------------------------------------------------------------------------- round 4: distillation pieces
def teacher():
    net, _ = RT.load(R.router_ready("t0p"))
    return net


def test_registry_round4():
    spec = R.router_spec("srkd")
    assert spec["round"] == 4 and spec["kind"] == "sr" and spec["heldout"] == "ylf" and spec["steps"] == 6000
    assert spec["kd"]["teacher"] == "t0p" and R.router_spec("t0p")["kind"] == "t0p"


def test_patch_parts_and_teacher_distribution():
    t = teacher()
    pp = RTR.patch_parts(t)
    assert pp.shape == (t.n_patches,) and sorted(set(pp.tolist())) == list(range(SM.N_PARTS))
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 7, t.n_patches, generator=g)
    for T in (1.0, 2.0):
        tp = RTR.teacher_part_prob(pp, logits, T)
        manual = torch.stack([torch.softmax(logits / T, -1)[..., pp == k].sum(-1) for k in range(SM.N_PARTS)], -1)
        assert torch.allclose(tp, manual, atol=1e-6) and torch.allclose(tp.sum(-1), torch.ones(3, 7))


def test_kd_loss_part_terms():
    t = teacher()
    pp = RTR.patch_parts(t)
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 6, t.n_patches, generator=g)
    kd = {"T": 2.0, "lambda_part": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    tp = RTR.teacher_part_prob(pp, logits, 2.0)
    tout = {"patch": logits, "onhand": torch.tensor([[3.0] * 6, [3.0, 3.0, -3.0, 3.0, 9.0, 9.0]]),
            "coord": torch.randn(2, 6, 3, generator=g)}
    match = {"mask": mask, "part": tp.log() * 2.0, "onhand": tout["onhand"].clone(), "coord": tout["coord"].clone()}
    total, parts = RTR.kd_loss_part(match, tout, pp, kd)
    assert abs(float(parts["kd_part"])) < 1e-5 and float(parts["kd_coord"]) == 0.0
    off = dict(match, part=torch.zeros(2, 6, SM.N_PARTS), coord=match["coord"] + 1.0)
    total2, parts2 = RTR.kd_loss_part(off, tout, pp, kd)
    assert float(parts2["kd_part"]) > 0.01 and float(parts2["kd_coord"]) > 0.0 and float(total2) > float(total)


def test_kd_loss_patch_terms():
    g = torch.Generator().manual_seed(0)
    B, N = 2, 32
    mask = torch.rand(B, N, generator=g) > 0.2
    logits = torch.randn(B, N, 279, generator=g)
    t = {"patch": logits, "onhand": torch.randn(B, N, generator=g) * 3, "coord": torch.randn(B, N, 3, generator=g)}
    s = {"patch": logits.clone(), "onhand": t["onhand"].clone(), "coord": t["coord"].clone(), "mask": mask}
    kd = {"T": 2.0, "lambda_patch": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}
    _, parts = RTR.kd_loss_patch(s, t, kd)
    assert abs(float(parts["kd_patch"])) < 1e-5 and float(parts["kd_coord"]) == 0.0
    s["patch"] = torch.randn(B, N, 279, generator=g)
    total, parts = RTR.kd_loss_patch(s, t, kd)
    assert float(parts["kd_patch"]) > 0.01
    assert torch.isclose(total, parts["kd_patch"] + 0.5 * parts["kd_coord"] + 0.5 * parts["kd_onhand"])


# --------------------------------------------------------------------------- rounds 6 / 8: the slim kinds
def test_srpkd_registered_with_sr1_recipe():
    spec, ref = R.router_spec("srpkd"), R.router_spec("srkd")
    assert spec["round"] == 6 and spec["kind"] == SRP.KIND and spec["params"] == SM.n_params(SRP.build(NPZ))
    assert spec["params"] + P.S45_PARAMS <= P.BUDGET["params_total_max"]
    assert {k: spec[k] for k in ("heldout", "steps", "dz")} == {k: ref[k] for k in ("heldout", "steps", "dz")}
    assert spec["kd"] == {"teacher": "t0p", "T": 2.0, "lambda_patch": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}
    assert R.router_checkpoint("srpkd") == R.router_dir("srpkd") / "srpkd_final.ckpt"
    assert not [a for a, s in R.registry()["arms"].items() if s["round"] == 6]


def test_srdwkd_registration():
    reg = R.registry()
    r = reg["routers"]["srdwkd"]
    assert r["round"] == 8 and r["kind"] == "dw" and r["params"] == 137291 == SM.n_params(DW.build(NPZ)) and r["heldout"] == "ylf" and r["dz"]
    assert r["kd"] == reg["routers"]["srpkd"]["kd"] and r["steps"] == reg["routers"]["srpkd"]["steps"]
    assert not [a for a, s in reg["arms"].items() if s["round"] == 8]


def test_trainer_kinds_table():
    assert set(RTR.KINDS) == {"sr", "srp", "dw"}
    assert RTR.KINDS["sr"]["module"] is RTR.SRModule and RTR.KINDS["srp"]["module"] is RTR.PatchModule is RTR.KINDS["dw"]["module"]
    assert issubclass(RTR.KINDS["sr"]["anneal"], RTR.SRModule) and issubclass(RTR.KINDS["dw"]["anneal"], RTR.PatchModule)
    assert RTR.DECODE_KINDS == H.KINDS


# --------------------------------------------------------------------------- round 10 / 12: annealing and decoding
def test_registry_round10():
    routers = R.registry()["routers"]
    mine = {n for n, s in routers.items() if s["round"] == 10}
    assert mine == set(ANNEALED) | set(DECODED)
    assert not any(s["round"] == 10 for s in R.registry()["arms"].values())
    for name, base in ANNEALED.items():
        spec, b = RTR.anneal_spec(name)
        assert spec["anneal_of"] == base and spec["kind"] == b["kind"] and spec["params"] == b.get("params", 137028)
    for name, base in DECODED.items():
        spec, b = RTR.decode_spec(name)
        assert spec["decode_of"] == base and H.KINDS[spec["kind"]] == b["kind"] == RTR.DECODE_KINDS[spec["kind"]]
    with pytest.raises(RuntimeError):
        RTR.anneal_spec("srpkdh")
    with pytest.raises(RuntimeError):
        RTR.decode_spec("srpkda")


def test_anneal_schedule_and_modules():
    total = 2000
    f = [RTR.factor(s, total) for s in range(total + 1)]
    assert f[0] == pytest.approx(1 / RTR.ANNEAL_WARMUP) and f[RTR.ANNEAL_WARMUP - 1] == pytest.approx(1 - 99 / total)
    assert f[total] == 0.0 and all(a >= b for a, b in zip(f[RTR.ANNEAL_WARMUP - 1:], f[RTR.ANNEAL_WARMUP:]))
    for module in (RTR.SRAnneal(SM.build(NPZ)), RTR.PatchAnneal(SRP.build(NPZ), None, None, None)):
        module.anneal_steps = total
        cfg = module.configure_optimizers()
        opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
        assert cfg["lr_scheduler"]["interval"] == "step"
        lrs = []
        for _ in range(total):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert lrs[0] == pytest.approx(RTR.LR / RTR.ANNEAL_WARMUP) and max(lrs) == pytest.approx(RTR.LR * (1 - 99 / total))
        assert lrs[-1] == pytest.approx(RTR.LR * RTR.factor(total - 1, total))


def test_constant_schedule_of_a_fresh_router():
    module = RTR.SRModule(SM.build(NPZ))
    cfg = module.configure_optimizers()
    opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    lrs = []
    for _ in range(RTR.WARMUP + 5):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(RTR.LR / RTR.WARMUP) and lrs[RTR.WARMUP - 1] == pytest.approx(RTR.LR) and lrs[-1] == pytest.approx(RTR.LR)


@torch.no_grad()
def test_hier_kinds_load_the_base_weights(tmp_path):
    torch.manual_seed(5)
    ckpt = tmp_path / "srp.ckpt"
    SRP.save_checkpoint(SRP.build(NPZ), ckpt)
    try:
        SRP.register()
        H.register()
        base, hier = SM._load_router("srp", ckpt), SM._load_router("srph", ckpt)
        assert all(torch.equal(a, b) for a, b in zip(base.parameters(), hier.parameters()))
        assert isinstance(hier.patch, H.PartFirst) and not isinstance(base.patch, H.PartFirst)
    finally:
        H.unregister()
        SRP.unregister()


def test_registry_round12_routers():
    reg = R.registry()
    mine = {n for n, s in reg["routers"].items() if s["round"] == 12}
    assert mine == set(ROUTERS12)
    nine, eight = R.router_spec("srpkd9"), R.router_spec("srpkd")
    drop = ("heldout", "note", "round")
    assert {k: v for k, v in nine.items() if k not in drop} == {k: v for k, v in eight.items() if k not in drop}
    assert nine["heldout"] is None and eight["heldout"] == "ylf"
    spec, base = RTR.anneal_spec("srpkd9a")
    assert spec["anneal_of"] == "srpkd9" and base["heldout"] is None and spec["anneal_steps"] == 2000
    for name in ("srpkd9h", "srpkd9ah"):
        spec, base = RTR.decode_spec(name)
        assert spec["decode_of"] == {"srpkd9h": "srpkd9", "srpkd9ah": "srpkd9a"}[name] and base["kind"] == "srp"
    for name in ROUTERS12:
        assert int(R.router_spec(name)["params"]) == 137787


# --------------------------------------------------------------------------- the selection rules
def seqs(cross, occ):
    return {"ylf_local": {"part_cross_share": cross, "occupancy_overlap_patch": occ}}


def test_select_r4_rule():
    base = seqs(0.2937, 0.4671)
    assert SEL.rule_r4(base, seqs(0.2937 - 0.005, 0.4671))["selected"]
    assert not SEL.rule_r4(base, seqs(0.2937 - 0.004, 0.5))["selected"]
    r = SEL.rule_r4(base, seqs(0.25, 0.4670))
    assert r["cross_ok"] and not r["occupancy_ok"] and not r["selected"]
    assert list(r) == ["cross_ok", "occupancy_ok", "selected"]


def test_select_r6_rule_and_r8_uses_it():
    base = seqs(0.2937, 0.4671)
    assert SEL.rule_r6(base, seqs(0.2937 + 0.005, 0.4671 + 0.02))["selected"]
    r = SEL.rule_r6(base, seqs(0.2937 + 0.0051, 0.6))
    assert r["occupancy_ok"] and not r["cross_ok"] and not r["selected"] and list(r) == ["occupancy_ok", "cross_ok", "selected"]
    r = SEL.rule_r6(base, seqs(0.25, 0.4671 + 0.019))
    assert r["cross_ok"] and not r["occupancy_ok"] and not r["selected"]
    assert SEL.RULES["r8"]["rule"] is SEL.rule_r6 and SEL.RULES["r8"]["candidate"] == "srdwkd" and SEL.SEQ == "ylf_local"
    b = {"ylf_local": {"occupancy_overlap_patch": 0.50, "part_cross_share": 0.28}}
    good = {"ylf_local": {"occupancy_overlap_patch": 0.521, "part_cross_share": 0.284}}
    bad = {"ylf_local": {"occupancy_overlap_patch": 0.519, "part_cross_share": 0.27}}
    assert SEL.rule_r6(b, good)["selected"] and not SEL.rule_r6(b, bad)["selected"]


def test_selection_files_and_queries(tmp_path, monkeypatch):
    """Write the three router selections and a context selection into a scratch results directory."""
    monkeypatch.setenv("S45SM_RESULTS_DIR", str(tmp_path))
    (tmp_path / "readouts_d1").mkdir()
    from tools.s45sm.paths import sha
    values = {"sr1": (0.2937, 0.4671), "srkd": (0.2937, 0.4671), "srpkd": (0.30, 0.52), "srdwkd": (0.31, 0.51)}
    for router, (cross, occ) in values.items():
        row = {"part_cross_share": cross, "occupancy_overlap_patch": occ, "face_geodesic_median_mm": 12.0,
               "coverage_onhand_0p5": 0.98}
        (tmp_path / "readouts_d1" / f"{router}.json").write_text(json.dumps({
            "router": router, "router_checkpoint_sha256": sha(R.router_ready(router)),
            "sequences": {"ylf_local": row, "zgz_local": row}}))
    SEL.run("r4")
    assert SEL.selected() is False
    out = json.loads((tmp_path / "select_r4.json").read_text())
    assert list(out)[:6] == ["status", "computed_utc", "rule", "sequence", "margin", "cross_ok"] and out["router_for_round5"] == "sr1"
    assert "reference" not in out and set(out) >= {"sr1", "srkd", "descriptive_zgz_local"}
    with pytest.raises(RuntimeError):
        SEL.run("r4")                                                   # computed once
    SEL.run("r6")
    out = json.loads((tmp_path / "select_r6.json").read_text())
    assert out["reference"] == "sr1" and out["occupancy_ok"] and not out["cross_ok"] and SEL.selected_router(6) == "sr1"
    assert list(out)[4:7] == ["occupancy_gain", "cross_slack", "reference"]
    SEL.run("r8")
    assert SEL.selected_router(8) == "sr1"
    with pytest.raises(RuntimeError):
        SEL.selected_router(7)
    assert SEL.selected_context("sr1") is None


def test_context_selection_rule():
    def probe(values):
        return {"by_context": {str(c): {"ylf_local": {"occupancy_overlap_patch": o, "part_cross_share": x}}
                               for c, (o, x) in values.items()}}
    base = {50: (0.467, 0.294)}
    best, rows = SEL.rule_ctx(probe({**base, 100: (0.480, 0.290), 150: (0.490, 0.300), 200: (0.495, 0.297), 300: (0.495, 0.299)}))
    assert best == 200 and not rows["100"]["qualifies"] and rows["150"]["cross_ok"] is False
    assert rows["300"]["qualifies"]                                         # tie on occupancy: the shorter window
    best, _ = SEL.rule_ctx(probe({**base, 100: (0.47, 0.29), 150: (0.48, 0.29), 200: (0.486, 0.29), 300: (0.5, 0.31)}))
    assert best is None


# --------------------------------------------------------------------------- one CPU training step per module kind
def logged(module):
    seen = {}
    module.log = lambda name, value, **kwargs: seen.__setitem__(name, value)
    return seen


def kd_cfg(kind):
    return ({"T": 2.0, "lambda_part": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5} if kind == "sr" else
            {"T": 2.0, "lambda_patch": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5})


@pytest.mark.parametrize("kind", ["sr", "srp", "dw"])
@pytest.mark.parametrize("variant", ["plain", "aug", "aug_kd"])
def test_a_training_step_runs_on_cpu(kind, variant):
    torch.manual_seed(0)
    net = RTR.KINDS[kind]["build"](NPZ)
    aug = {"rot_deg": 30, "scale": [0.8, 1.2], "keep": [0.5, 1.0]} if variant != "plain" else None
    kd, teacher_net = (kd_cfg(kind), teacher().eval()) if variant == "aug_kd" else (None, None)
    for q in (teacher_net.parameters() if teacher_net is not None else ()):
        q.requires_grad_(False)
    module = RTR.KINDS[kind]["module"](net, aug, teacher_net, kd)
    seen = logged(module)
    loss = module.training_step(batch(1, (300, 0, 200)), 0)
    assert loss.ndim == 0 and bool(torch.isfinite(loss)) and "train_loss" in seen
    if variant == "aug_kd":
        assert any(k.startswith("train_kd_") for k in seen)
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in net.parameters())


def test_finish_callback_writes_checkpoint_and_completion(tmp_path):
    net = SM.build(NPZ)

    class Trainer:
        global_rank, global_step = 0, 50

    class Module:
        pass
    module = Module()
    module.net = net
    target = tmp_path / "sr_final.ckpt"
    RTR.Finish(target, "preflight", SM.save_checkpoint, {"heldout": "ylf"}).on_train_end(Trainer, module)
    done = json.loads(target.with_suffix(".json").read_text())
    assert done["status"] == "COMPLETE" and done["step"] == 50 and done["params"] == 137028 and done["checkpoint_sha256"] == P.sha(target)
    loaded, meta = SM.load(target)
    assert meta["heldout"] == "ylf" and "kd" not in meta and meta["warmup"] == RTR.WARMUP
    target2 = tmp_path / "anneal_final.ckpt"
    meta2 = {"anneal_of": "sr1", "base_checkpoint_sha256": "x", "heldout": "ylf"}
    RTR.Finish(target2, "formal", SM.save_checkpoint, meta2, anneal=True).on_train_end(Trainer, module)
    done2 = json.loads(target2.with_suffix(".json").read_text())
    assert done2["anneal_of"] == "sr1" and done2["base_checkpoint_sha256"] == "x" and SM.load(target2)[1]["anneal_warmup"] == RTR.ANNEAL_WARMUP
    target3 = tmp_path / "srp_final.ckpt"
    module.net = SRP.build(NPZ)
    RTR.Finish(target3, "formal", SRP.save_checkpoint, {"heldout": None, "kd": None}).on_train_end(Trainer, module)
    assert "kd" in SRP.load(target3)[1]                                 # the patch routers always record their kd (None without)
