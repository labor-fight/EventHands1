"""ACU stage-1 CPU tests (docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104). Run with CUDA_VISIBLE_DEVICES="".

* registry: a1 / a2 are their bases' configs with names, directories and the ACU block only
* face groups equal S45-SM's face-part table mapped to hand groups
* CA on real zgz packets with the label identity at the GT previous state: most nodes associated, residuals of about a
  pixel; a node's association does not depend on the other nodes of the packet; the CA trunk has B2's parameters
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) != "", reason="CPU tests: CUDA_VISIBLE_DEVICES=''")


def test_registry_single_change():
    from tools.acu import arms as A
    for arm in A.ARMS:
        for phase in A.PHASES:
            cfg = A.build_config(arm, 3407, phase)
            base = A.base_config(A.ARMS[arm]["base"], 3407, phase)
            c, b = copy.deepcopy(cfg), copy.deepcopy(base)
            for d in (c, b):
                d["TRAIN"].pop("OUTPUT_DIR"), d["TRAIN"].pop("RUN_NAME"), d["EVAL"].pop("OUTPUT_DIR")
            c.pop("ACU")
            assert c == b, arm
            assert cfg["ACU"]["IFACE"] == "ca" and cfg["PFO"]["TRUNK"] == "gshare"


def test_face_groups_match_router_tables():
    import torch
    from semkine import acu_ca as CA
    from semkine import s45sm as SM
    with np.load(ROOT / "assets/mano_right.npz", allow_pickle=False) as z:
        faces, w = np.asarray(z["f"], np.int64), np.asarray(z["weights"])
    part = SM.face_part_table(w, faces)
    g = CA.face_groups(torch.tensor(faces), torch.tensor(w)).numpy()
    assert np.array_equal(g, np.asarray(CA.GROUP_OF_PART)[part])


@pytest.fixture(scope="module")
def label_net():
    import torch
    from semkine import acu_ca as CA
    from tools.acu.run import install_ca
    from tools.pfo import nets as N
    torch.set_num_threads(8)
    install_ca("gshare")
    sp = json.loads((ROOT / "outputs/pfo2/20261103/specs/pfo2_fgs_s3407.json").read_text())
    sp = {**sp, "checkpoint": Path(sp["checkpoint"]), "config": Path(sp["config"])}
    net = N.load("acu_test", route="label", device=torch.device("cpu"), spec=sp)
    net.model.xyz_keep_aux = True
    yield net, CA
    CA.uninstall()


def _packet(net, seq, i):
    import torch
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    root, seqs = E.sequences(net.cfg, (seq,))
    ctx = N.sequence(root, seqs[0][1], seq, torch.device("cpu"))
    net.bind(ctx)
    rows = E.endpoint_rows(ctx.aux)
    _, end, prev_ms, _ = rows[i]
    return ctx, int(end), np.asarray(ctx.gt[int(prev_ms)], np.float32).copy()


def test_ca_label_identity_at_gt(label_net):
    net, CA = label_net
    ctx, end, prev = _packet(net, "zgz_local", 800)
    pred = net.step(ctx, end, prev)
    assert np.isfinite(pred).all()
    a, m = net.model.xyz_last["assoc"], net.model.xyz_last["mask"]
    q = a["q"][m]
    r = a["ca_geo"][m][:, 0] * CA.RHO_PX
    assert float((q > 0).float().mean()) > 0.7
    assert float(r[q > 0].abs().median()) < 2.0
    nu = a["ca_geo"][m][q > 0][:, 1:]
    assert np.allclose(nu.norm(dim=-1).numpy(), 1.0, atol=1e-4)


def test_ca_association_is_per_node(label_net):
    """Dropping half of the other nodes changes nothing for the kept nodes (no cross-node dependency)."""
    import torch
    from semkine import xag as XAG
    net, CA = label_net
    ctx, end, prev = _packet(net, "zgz_local", 800)
    net.step(ctx, end, prev)
    last = net.model.xyz_last
    d, mask, verts = last["d"], last["mask"], last["verts"]
    src = {k: last[k] for k in ("xag_part", "xag_face", "xag_bary", "xag_valid")}
    xyz = net.model.xyz
    args = (d, mask, verts, net.model.mano.f, xyz.vert_faces, xyz.lbs_w, None, None, None)
    zero = {k: torch.zeros_like(last["assoc"][k]) for k in ("vid", "A", "q", "z", "valid", "hit")}
    full, _ = XAG.route_association("face_route", zero, *args, source_part=src["xag_part"], source_face=src["xag_face"],
                                    source_bary=src["xag_bary"], source_valid=src["xag_valid"])
    keep = torch.zeros_like(mask)
    keep[:, ::2] = True
    half, _ = XAG.route_association("face_route", zero, *args, source_part=src["xag_part"], source_face=src["xag_face"],
                                    source_bary=src["xag_bary"], source_valid=src["xag_valid"] & keep)
    sel = keep & mask & src["xag_valid"]
    for k in ("vid", "A", "q", "z", "ca_geo"):
        assert torch.equal(full[k][sel], half[k][sel]), k


def test_ca_trunk_has_b2_parameters(label_net):
    import torch
    net, _ = label_net
    ck = torch.load(json.loads((ROOT / "outputs/pfo2/20261103/specs/pfo2_fgs_s3407.json").read_text())["checkpoint"],
                    map_location="cpu")["state_dict"]
    mine = net.model.state_dict()
    assert set(k for k in ck if k.startswith("xyz.")) == set(k for k in mine if k.startswith("xyz."))
    assert all(ck[k].shape == mine[k].shape for k in ck if k.startswith("xyz."))
