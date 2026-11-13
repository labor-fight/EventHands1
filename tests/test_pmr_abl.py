"""ABL registry, trunk variant and gate rule (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111). CPU only, its own process:
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_pmr_abl.py"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def changed(a, b):
    return {k: sorted(kk for kk in set(a[k]) | set(b.get(k, {})) if a[k].get(kk) != b.get(k, {}).get(kk))
            for k in a if isinstance(a[k], dict) and a[k] != b.get(k)}


def test_registry_changes_only_the_ablated_block():
    from tools.pmr import arms_any6  # noqa: F401
    from tools.pmr import arms as A, arms_any2 as U, arms_rol2 as R2
    base = A.build_config("d1r", 3407)
    names = {"EVAL": ["OUTPUT_DIR"], "TRAIN": ["OUTPUT_DIR", "RUN_NAME"]}
    geo = changed(U.build_config("abgeo", 3407), base)
    assert geo == {**names, "MODEL": ["XAG_ASSOC"], "PFO": ["ARM", "ASSOC", "ROUTER", "ROUTER_CHECKPOINT", "ROUTER_KIND",
                                                             "SECTION"], "PMR": ["BASE_ARM", "DESIGN"]}
    nr = U.build_config("abnr", 3407)
    assert changed(nr, base) == {**names, "PFO": ["ARM", "SECTION"], "PMR": ["BASE_ARM", "DESIGN", "MODULE", "VARIANT"]}
    assert nr["PMR"]["MODULE"] == "semkine.pmr_trunk11" and nr["PFO"]["ROUTER"] == "r_s2d"
    for arm, module, variant in (("abmv3", "semkine.pmr_trunk", "mlp"), ("abg0v3", "semkine.pmr_trunk2", "rconst"),
                                 ("abh0v3", "semkine.pmr_trunk2", "rnohub"), ("abnrv3", "semkine.pmr_trunk11", "rnores")):
        c = U.build_config(arm, 3407)
        assert c["PFO"]["ROUTER"] == "rdr_v3" and c["PFO"]["ROUTER_KIND"] == "pfr" and c["LOSS"] == base["LOSS"]
        assert (c["PMR"]["MODULE"], c["PMR"]["VARIANT"]) == (module, variant) and c["AUG"] == base["AUG"]
    r1, r5 = R2.build_config("d1r_rol1", 3407), A.build_config("d1r", 3407)
    assert r1["ROL"]["K"] == 1 and not r1["ROL"]["GRAD"] and r1["MODEL"]["INIT_FROM"].endswith("pmr_d1r_s3407-step=6000.ckpt")
    assert {k: v for k, v in R2.build_config("d1r_rol5", 3407)["TRAIN"].items()} == r1["TRAIN"] | {
        "OUTPUT_DIR": R2.build_config("d1r_rol5", 3407)["TRAIN"]["OUTPUT_DIR"], "RUN_NAME": "pmr_d1r_rol5_s3407"}
    assert r5["PFO"]["ROUTER"] == "r_s2d"


def test_rnores_zeroes_only_the_residual():
    from semkine import pmr_trunk as PM, pmr_trunk11 as T11
    B, N, R = 2, 6, 4
    g = {"m": None}
    torch.manual_seed(0)
    assoc = {"q": torch.ones(B, N), "A": torch.full((B, N, R), 0.25), "vid": torch.randint(0, 778, (B, N, R)),
             "z": torch.full((B, N), 0.6)}
    mask = torch.ones(B, N, dtype=torch.bool)
    d = torch.nn.functional.normalize(torch.randn(B, N, 3) + torch.tensor([0, 0, 3.0]), dim=-1)
    verts = torch.randn(B, 778, 3) * 0.05 + torch.tensor([0, 0, 0.6])
    w = assoc["q"].unsqueeze(-1) * assoc["A"]
    m, P, res = PM.event_geometry(w, d, assoc["z"], assoc["vid"], verts)
    assert float(res.abs().sum()) > 0

    class Probe(T11.PMRNoResNet):
        def __init__(self):                       # geometry only: no modules needed
            torch.nn.Module.__init__(self)

    def parent_geometry(self, mask, d, verts, assoc, prev48, ablate, acc):
        mm, PP, rr = PM.event_geometry(w, d, assoc["z"], assoc["vid"], verts)
        return {"m": mm, "P": PP, "res": rr}
    native = PM.PMRNet.geometry
    PM.PMRNet.geometry = parent_geometry
    try:
        out = Probe().geometry(mask, d, verts, assoc, None, False, torch.float32)
    finally:
        PM.PMRNet.geometry = native
    assert torch.equal(out["P"], P) and torch.equal(out["m"], m) and float(out["res"].abs().sum()) == 0.0


def test_trunk_rule():
    from tools.pmr import abl
    ref = [11.69, 8.90, 9.59, 7.56, 10.20]
    assert abl.trunk_better([11.5, 8.6, 9.4, 7.3, 9.69], ref)[0]
    assert not abl.trunk_better([11.5, 8.6, 9.4, 7.3, 9.71], ref)[0]
    assert not abl.trunk_better([12.0, 8.6, 9.4, 7.3, 9.6], ref)[0]
    assert not abl.trunk_better([11.0, 9.21, 9.4, 7.3, 9.6], ref)[0]
