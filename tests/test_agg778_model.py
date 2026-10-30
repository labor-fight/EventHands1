"""AGG778 aggregation variants (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): opt-in, default bit-identical,
matpt / vin start as C1's function, every C1 parameter keeps its initial value, fan-out records are well formed."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine.agg778 import AGG_MODES, FANOUT_M, AggXYZMeshNet  # noqa: E402
from semkine.xyz_mesh import XYZMeshNet  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_xcl_model import activate_head, inputs, scene  # noqa: E402


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def build(agg=None):
    args, _ = scene()
    torch.manual_seed(3407)
    if agg is None:
        return XYZMeshNet(args[6], args[4])
    return AggXYZMeshNet(args[6], args[4], agg=agg)


def run(m, band=None):
    m.eval()
    kw = {"band": band} if band is not None else {}
    with torch.no_grad():
        return m(*inputs(), capture=True, **kw)


@pytest.mark.parametrize("agg", ["matpt", "mass", "fanout", "vin", "noabs"])
def test_shared_parameters_keep_c1_initial_values(agg):
    base, var = build(), build(agg)
    sb, sv = base.state_dict(), var.state_dict()
    for k, v in sb.items():
        assert k in sv and torch.equal(v, sv[k]), k
    extra = set(sv) - set(sb)
    assert extra == {"matpt": {"rel_extra.weight"}, "vin": {"vin_a.weight", "vin_b.weight"}}.get(agg, set())


@pytest.mark.parametrize("agg", ["matpt", "vin"])
def test_zero_initialised_variants_start_as_c1(agg):
    base, var = activate_head(build()), activate_head(build(agg))
    db, _, ab = run(base)
    dv, _, av = run(var)
    assert torch.equal(db, dv)
    assert torch.equal(ab["H0"], av["H0"])


def test_mass_variant_changes_only_pooling_semantics():
    base, var = activate_head(build()), activate_head(build("mass"))
    _, _, ab = run(base)
    _, _, av = run(var)
    assert torch.equal(ab["bslot"], av["bslot"]), "residual slot unchanged"
    assert ((av["O"] > 0) == (ab["O"] > 0)).all(), "soft support has the same carrier"
    assert (av["O"] <= 1).all() and (av["O"] >= 0).all()


def test_fanout_records_use_band_and_nearest_vertices():
    var = activate_head(build("fanout"))
    tok, d, mask, verts, assoc, prev48 = inputs()
    vid, w = var.fanout(d, mask, verts, assoc["q"], torch.tensor([0.5]))
    assert vid.shape == (1, d.shape[1], min(FANOUT_M, verts.shape[1])) and w.shape == vid.shape
    live = (assoc["q"] > 0) & mask
    assert torch.allclose(w.sum(-1)[live], assoc["q"][live].float(), atol=1e-6)
    assert (w[~live] == 0).all()
    with pytest.raises(ValueError):
        run(var)                                                                # band is required
    run(var, band=torch.tensor([0.5]))


def test_unknown_or_incompatible_variant_fails_closed():
    args, _ = scene()
    with pytest.raises(ValueError):
        AggXYZMeshNet(args[6], args[4], agg="legacy")
    with pytest.raises(ValueError):
        AggXYZMeshNet(args[6], args[4], agg="nope")
    with pytest.raises(ValueError):
        AggXYZMeshNet(args[6], args[4], agg="mass", geom=False)
    assert AGG_MODES[0] == "legacy"


def _model(extra):
    from config import load_config
    from model import MNISTModel
    cfg = load_config(str(ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"))
    cfg["MODEL"].update(extra)
    torch.manual_seed(3407)
    return MNISTModel(cfg)


def test_model_key_default_is_bit_identical_and_variants_dispatch():
    a, b = _model({}), _model({"XYZ_AGG": "legacy"})
    assert type(a.xyz) is XYZMeshNet and type(b.xyz) is XYZMeshNet
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    c = _model({"XYZ_AGG": "matpt"})
    assert type(c.xyz) is AggXYZMeshNet and c.xyz.agg == "matpt"
    sc = c.state_dict()
    assert all(torch.equal(sa[k], sc[k]) for k in sa), "shared initial values identical to C1"
    with pytest.raises(ValueError):
        _model({"XYZ_AGG": "bogus"})


def test_noabs_zeroes_exactly_the_absolute_cues():
    """R2 `noabs`: records lose token x/W, y/H and zn; residual tokens lose x/W, y/H; nothing else changes."""
    from semkine.agg778 import NOABS_RECORD_COLS, NOABS_TOKEN_COLS
    base, var = activate_head(build()), activate_head(build("noabs"))
    seen = {}
    for name, m in (("base", base), ("var", var)):
        calls = []
        h = m.rel.register_forward_hook(lambda mod, i, o, c=calls: c.append(i[0].detach().clone()))
        run(m)
        h.remove()
        seen[name] = calls
    rb, rv = seen["base"][0], seen["var"][0]
    keep = [c for c in range(rb.shape[1]) if c not in NOABS_RECORD_COLS]
    assert torch.equal(rv[:, keep], rb[:, keep])
    assert (rv[:, list(NOABS_RECORD_COLS)] == 0).all()
    if len(seen["base"]) > 1 and seen["base"][1].numel():
        tb, tv = seen["base"][1], seen["var"][1]
        keep_t = [c for c in range(tb.shape[1]) if c not in NOABS_TOKEN_COLS]
        assert torch.equal(tv[:, keep_t], tb[:, keep_t])
        assert (tv[:, list(NOABS_TOKEN_COLS)] == 0).all()
