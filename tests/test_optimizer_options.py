"""Generic optimiser options of `BaseModel`: `TRAIN.WEIGHT_DECAY` (AdamW with decoupled decay on the Conv / Linear weights)
and `TRAIN.EMA_DECAY` (an exponential moving average of the weights that every saved checkpoint carries).

Both are off by default (plain Adam, raw weights in every checkpoint); with them on:

  (a) WEIGHT_DECAY: parameter groups (every trainable parameter exactly once; decay = Conv / Linear weights), the
      decoupled decay arithmetic of one AdamW step, Adam when 0, the same LR schedule.
  (b) EMA: the update arithmetic and its warm-up, exactness for constant weights, checkpoint layout (also from inside the
      validation loop, where the live tensors hold the EMA), a real Lightning `fit` on CPU: saved state_dict = the EMA,
      load_from_checkpoint returns the EMA weights (evaluation) while `fit(ckpt_path=...)` returns the raw weights with the
      EMA continuing -- a run split by a resume equals the uninterrupted run bit for bit --, `validate(ckpt_path=...)` keeps
      the EMA, validation uses the EMA weights and restores the raw ones.

(They were part of `tests/test_dt_model3.py`, which was removed with the DT2 arms that tested them against a pristine
copy of the model; `tests/test_core_fingerprint.py` guards the default path now.)

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_optimizer_options.py -q
"""
from __future__ import annotations

import copy
import functools
import shutil
import sys
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config                                               # noqa: E402
from model import MNISTModel                                                 # noqa: E402

BASE = REPO / "configs" / "dt" / "dt_dz_l3.yaml"
H, W = 180, 240
BASE_K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------------------------------- helpers
@functools.lru_cache(maxsize=None)

def _load(path):
    return load_config(Path(path))


def _cfg(path=BASE, model=None, loss=None, train=None):
    cfg = copy.deepcopy(_load(path))
    cfg["MODEL"].update(model or {})
    cfg["LOSS"].update(loss or {})
    cfg["TRAIN"].update(train or {})
    return cfg


def _build(cls, cfg, seed=1234):
    torch.manual_seed(seed)
    return cls(copy.deepcopy(cfg))



def _randomize(m, seed=7):
    """What a trained model has and a fresh one does not: prev_mlp's zero-initialised last layer, BatchNorm statistics that
    are not the identity ."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if hasattr(m, "prev_mlp"):
            for p in m.prev_mlp.parameters():
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.copy_(0.1 * torch.randn(mod.running_mean.shape, generator=g))
                mod.running_var.copy_(1.0 + 0.2 * torch.rand(mod.running_var.shape, generator=g))
    return m



def _lnes(B, seed=0, empty=()):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(B, H, W, 2, generator=g) < 0.12).float() * torch.rand(B, H, W, 2, generator=g)
    for i in empty:
        x[i] = 0.0
    return x


def _prev(B, seed=0, norm=2.3):
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(B, 51)
    p[:, :2] = 0.05 * torch.randn(B, 2, generator=g)
    p[:, 2] = 0.55 + 0.03 * torch.randn(B, generator=g)
    axis = torch.randn(B, 3, generator=g)
    p[:, 3:6] = axis / axis.norm(dim=-1, keepdim=True) * norm
    p[:, 6:] = 0.3 * torch.randn(B, 45, generator=g)
    return p


def _betas(B, seed=3):
    return 0.3 * torch.randn(B, 10, generator=torch.Generator().manual_seed(seed))



def _assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k], sb[k]), k


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


class _Synth(Dataset):
    """Deterministic legacy-tuple samples (lnes, prev, target, betas, K): the target is the prev state plus a small step."""

    def __init__(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = [_lnes(1, seed=100 + i)[0] for i in range(n)]
        self.prev = [_prev(1, seed=200 + i)[0] for i in range(n)]
        self.y = [p + 0.02 * torch.randn(51, generator=g) for p in self.prev]
        self.betas = [_betas(1, seed=300 + i)[0] for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.prev[i], self.y[i], self.betas[i], BASE_K.clone() * torch.tensor([1.0, 1.0, 1.0])


def _fit_cfg(**kw):
    """dt_dz_l3's architecture with a 8-step schedule (the LR schedule reads TRAIN.MAX_STEPS)."""
    train = {"LR": 4e-3, "WARMUP_STEPS": 2, "MAX_STEPS": 8}
    train.update(kw.pop("train", {}))
    return _cfg(train=train, **kw)


def _trainer(tmp, max_steps, every=2, callbacks=()):
    ck = ModelCheckpoint(dirpath=str(tmp), filename="c-{step}", monitor=None, save_last=True, save_top_k=-1,
                         every_n_train_steps=every)
    return pl.Trainer(accelerator="cpu", devices=1, max_steps=max_steps, precision=32, logger=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                      accumulate_grad_batches=2, val_check_interval=4, check_val_every_n_epoch=None,
                      callbacks=[ck, *callbacks], default_root_dir=str(tmp)), ck


def _loaders():
    return (DataLoader(_Synth(8), batch_size=2, shuffle=False), DataLoader(_Synth(2, seed=9), batch_size=2, shuffle=False))



# ================================================================================================ (a) WEIGHT_DECAY
def test_wd_param_groups_cover_every_trainable_parameter_once_and_decay_conv_linear_weights_only():
    m = _build(MNISTModel, _cfg(train={"WEIGHT_DECAY": 0.01}), 5)
    groups = m._param_groups(0.01)
    assert [g["weight_decay"] for g in groups] == [0.01, 0.0]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in m.parameters() if p.requires_grad)
    name_of = {id(p): n for n, p in m.named_parameters()}
    decay = {name_of[id(p)] for p in groups[0]["params"]}
    plain = {name_of[id(p)] for p in groups[1]["params"]}
    mods = dict(m.named_modules())
    want_decay = {n + ".weight" for n, mod in mods.items() if isinstance(mod, (nn.Conv2d, nn.Linear))}
    assert decay == want_decay and len(decay) > 10
    for n in plain:
        mod = mods[n.rsplit(".", 1)[0]]
        assert n.endswith(".bias") or isinstance(mod, nn.BatchNorm2d), n
    assert {"rn.fc.weight", "conv1.weight", "prev_mlp.0.weight"} <= decay
    assert {"rn.fc.bias", "conv1.bias", "rn.bn1.weight", "rn.bn1.bias", "rn.layer1.0.bn1.weight"} <= plain


def test_wd_adamw_with_zero_gradient_applies_only_the_decoupled_decay():
    m = _build(MNISTModel, _fit_cfg(train={"WEIGHT_DECAY": 0.05}), 6)
    opt = m.configure_optimizers()["optimizer"]
    assert type(opt) is torch.optim.AdamW
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    lr = opt.param_groups[0]["lr"]
    opt.step()
    groups = m._param_groups(0.05)
    decayed = {n for n, p in m.named_parameters() if any(p is q for q in groups[0]["params"])}
    for n, p in m.named_parameters():
        if n in decayed:
            assert torch.allclose(p, before[n] * (1.0 - lr * 0.05), rtol=1e-6, atol=0), n
            if float(before[n].abs().max()) > 0:
                assert not torch.equal(p, before[n]), n
        else:
            assert torch.equal(p, before[n]), n


def test_wd_zero_decay_is_adam_and_the_schedule_is_unchanged():
    assert type(_build(MNISTModel, _fit_cfg(train={"WEIGHT_DECAY": 0.0})).configure_optimizers()["optimizer"]) is torch.optim.Adam
    sched = _build(MNISTModel, _fit_cfg(train={"WEIGHT_DECAY": 0.01})).configure_optimizers()
    ref = _build(MNISTModel, _fit_cfg()).configure_optimizers()
    assert all(sched["lr_scheduler"]["scheduler"].lr_lambdas[0](s) == ref["lr_scheduler"]["scheduler"].lr_lambdas[0](s)
               for s in range(10))


# ====================================================================================================== (b) EMA
def _perturb(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn(p.shape, generator=g))
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.add_(0.01 * torch.randn(mod.running_mean.shape, generator=g))
                mod.running_var.mul_(1.0 + 0.01 * torch.rand(mod.running_var.shape, generator=g))


def test_ema_update_arithmetic_warmup_and_tracked_entries():
    m = _build(MNISTModel, _cfg(train={"EMA_DECAY": 0.999}))
    live0 = m._ema_live()
    names = set(live0)
    sd_keys = set(m.state_dict())
    assert names <= sd_keys
    assert not any("num_batches_tracked" in k for k in names)
    assert {k for k in sd_keys if k.endswith(("running_mean", "running_var"))} <= names
    assert {n for n, _ in m.named_parameters()} <= names
    ref = {k: v.detach().clone().double() for k, v in live0.items()}
    m._ema_ready(live0)
    for step in range(30):
        _perturb(m, 100 + step)
        m.ema_update()
        d = min(0.999, (1.0 + step) / (10.0 + step))
        live = m._ema_live()
        for k in ref:
            ref[k] = ref[k] + (1.0 - d) * (live[k].double() - ref[k])
    assert m._ema_steps == 30
    worst = max(float((m._ema[k].double() - ref[k]).abs().max()) for k in ref)
    assert worst < 1e-5, worst
    assert min(0.999, (1.0 + 0) / (10.0 + 0)) == 0.1 and min(0.999, (1.0 + 20000) / (10.0 + 20000)) == 0.999


def test_ema_is_exactly_the_weights_when_they_do_not_change():
    m = _build(MNISTModel, _cfg(train={"EMA_DECAY": 0.999}))
    live = m._ema_live()
    m._ema_ready(live)
    for _ in range(40):
        m.ema_update()
    assert all(torch.equal(m._ema[k], v) for k, v in live.items())


def test_ema_off_means_off():
    m = _build(MNISTModel, _cfg())
    m.ema_update()
    assert m._ema is None and m._ema_steps == 0
    ck = {"state_dict": m.state_dict()}
    m.on_save_checkpoint(ck)
    assert set(ck) == {"state_dict"}
    with pytest.raises(RuntimeError, match="EMA_DECAY"):
        m.ema_state_dict()


def test_ema_checkpoint_layout_and_the_raw_weights_survive_a_save_from_inside_validation():
    m = _randomize(_build(MNISTModel, _cfg(train={"EMA_DECAY": 0.9})))
    m._ema_ready(m._ema_live())
    for s in range(5):
        _perturb(m, 40 + s)
        m.ema_update()
    raw = {k: v.clone() for k, v in m.state_dict().items()}
    ema = m.ema_state_dict()
    assert any(not torch.equal(raw[k], ema[k]) for k in raw)
    ck = {"state_dict": m.state_dict()}
    m.on_save_checkpoint(ck)
    assert list(ck["state_dict"]) == list(raw) == list(ck["raw_state_dict"])
    assert all(torch.equal(ck["state_dict"][k], ema[k]) for k in raw)
    assert all(torch.equal(ck["raw_state_dict"][k], raw[k]) for k in raw)
    assert ck["ema"] == {"decay": 0.9, "steps": 5}
    # the tensors are copies: later EMA steps do not change the written dicts
    snap = {k: v.clone() for k, v in ck["state_dict"].items()}
    _perturb(m, 77)
    m.ema_update()
    assert all(torch.equal(ck["state_dict"][k], snap[k]) for k in snap)
    # inside the validation loop the live tensors hold the EMA; the raw weights are still what `raw_state_dict` gets
    raw2 = {k: v.clone() for k, v in m.state_dict().items()}
    ema2 = m.ema_state_dict()
    m.on_validation_start()
    assert all(torch.equal(m.state_dict()[k], ema2[k]) for k in ema2 if k in m._ema)
    ck2 = {"state_dict": m.state_dict()}
    m.on_save_checkpoint(ck2)
    assert all(torch.equal(ck2["raw_state_dict"][k], raw2[k]) for k in raw2)
    assert all(torch.equal(ck2["state_dict"][k], ema2[k]) for k in ema2)
    m.on_validation_end()
    assert all(torch.equal(m.state_dict()[k], raw2[k]) for k in raw2) and m._ema_backup is None


class _Probe(Callback):
    """Records, around the validation loop, whether the model held the EMA or the raw weights."""

    def __init__(self):
        self.raw_before, self.during, self.after = None, [], []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.raw_before = {k: v.clone() for k, v in pl_module._ema_live().items()}

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        live = pl_module._ema_live()
        self.during.append((all(torch.equal(live[k], pl_module._ema[k]) for k in live),
                            all(torch.equal(live[k], self.raw_before[k]) for k in live)))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.during:
            live = pl_module._ema_live()
            self.after.append(all(torch.equal(live[k], self.raw_before[k]) for k in live))


@pytest.fixture(scope="module")
def ema_runs(tmp_path_factory):
    """Real CPU Lightning runs of the EMA + AdamW arm (dt_dz_l3_reg's keys), 8 optimiser steps (accumulate 2, 4 epochs of 4
    micro-batches, validation every epoch): (A) uninterrupted; (B) 4 steps, then `fit(ckpt_path=last.ckpt)` to 8."""
    root = tmp_path_factory.mktemp("ema_runs")
    cfg = _fit_cfg(train={"EMA_DECAY": 0.9, "WEIGHT_DECAY": 0.01})
    out = {"cfg": cfg}

    torch.manual_seed(0)
    a = _randomize(MNISTModel(copy.deepcopy(cfg)))
    probe = _Probe()
    tr, ck = _trainer(root / "A", 8, callbacks=[probe])
    tr.fit(a, *_loaders())
    out.update(A=a, A_ck=ck, probe=probe, A_dir=root / "A")

    torch.manual_seed(0)
    b = _randomize(MNISTModel(copy.deepcopy(cfg)))
    tr, ck1 = _trainer(root / "B1", 4)
    tr.fit(b, *_loaders())
    shutil.copy(ck1.last_model_path, root / "B1_step4.ckpt")
    out["B1_last"] = str(root / "B1_step4.ckpt")
    b1_ckpt = torch.load(out["B1_last"], map_location="cpu")
    out["B1_ckpt"] = b1_ckpt
    torch.manual_seed(123)                                   # a resumed run must not depend on the construction seed
    b2 = MNISTModel(copy.deepcopy(cfg))
    seen = {}

    class _AtStart(Callback):
        def on_train_start(self, trainer, pl_module):
            seen["live"] = {k: v.clone() for k, v in pl_module.state_dict().items()}
            seen["ema_steps"] = pl_module._ema_steps
            seen["pending"] = None if pl_module._ema_pending is None else {k: v.clone() for k, v in pl_module._ema_pending.items()}
            seen["global_step"] = int(trainer.global_step)

    tr, ck2 = _trainer(root / "B2", 8, callbacks=[_AtStart()])
    tr.fit(b2, *_loaders(), ckpt_path=out["B1_last"])
    out.update(B=b2, B_ck=ck2, at_start=seen)
    return out


def test_ema_real_fit_validation_runs_on_the_ema_weights_and_restores_the_raw_ones(ema_runs):
    p = ema_runs["probe"]
    assert len(p.during) >= 2
    assert all(ema and not raw for ema, raw in p.during), p.during          # validation saw the EMA, not the raw weights
    assert p.after and all(p.after), p.after                                # and the raw weights were back for training


def test_ema_real_fit_saved_state_dict_is_the_ema_and_raw_state_dict_the_trained_weights(ema_runs):
    a, ck = ema_runs["A"], ema_runs["A_ck"]
    c = torch.load(ck.last_model_path, map_location="cpu")
    assert c["global_step"] == 8 and c["ema"] == {"decay": 0.9, "steps": 8}
    final_raw, final_ema = a.state_dict(), a.ema_state_dict()
    assert list(c["state_dict"]) == list(c["raw_state_dict"]) == list(final_raw)
    assert all(torch.equal(c["raw_state_dict"][k].cpu(), final_raw[k].cpu()) for k in final_raw)
    assert all(torch.equal(c["state_dict"][k].cpu(), final_ema[k].cpu()) for k in final_ema)
    assert any(not torch.equal(c["state_dict"][k], c["raw_state_dict"][k]) for k in final_ema)     # the EMA lags the weights
    grid = sorted(Path(ema_runs["A_dir"]).glob("c-step=*.ckpt"))
    assert len(grid) == 4                                                                           # every 2nd step: 2, 4, 6, 8
    steps = [torch.load(g, map_location="cpu")["ema"]["steps"] for g in grid]
    assert steps == [2, 4, 6, 8]                                         # a checkpoint at step N holds the EMA after N updates


def test_ema_load_from_checkpoint_returns_the_ema_weights(ema_runs):
    """Evaluation (evalx, select_checkpoint, the distillation teacher): no Trainer, so no raw weights are swapped in."""
    c = torch.load(ema_runs["A_ck"].last_model_path, map_location="cpu")
    m = MNISTModel.load_from_checkpoint(ema_runs["A_ck"].last_model_path, cfg=copy.deepcopy(ema_runs["cfg"]), map_location="cpu")
    sd = m.state_dict()
    assert all(torch.equal(sd[k], c["state_dict"][k]) for k in sd)
    assert any(not torch.equal(sd[k], c["raw_state_dict"][k]) for k in sd)


def test_ema_resume_restores_the_raw_weights_and_the_ema_continues_without_a_seam(ema_runs):
    c1, at = ema_runs["B1_ckpt"], ema_runs["at_start"]
    assert at["global_step"] == 4 and at["ema_steps"] == 4
    # at the start of training the model holds the RAW weights of the checkpoint, not the EMA ones
    assert all(torch.equal(at["live"][k], c1["raw_state_dict"][k]) for k in at["live"])
    assert any(not torch.equal(at["live"][k], c1["state_dict"][k]) for k in at["live"])
    # and the EMA was restored from the checkpoint's state_dict
    assert at["pending"] is not None and all(torch.equal(at["pending"][k], c1["state_dict"][k]) for k in at["pending"])
    # the split run equals the uninterrupted run bit for bit: raw weights, EMA, checkpoints
    a, b = ema_runs["A"], ema_runs["B"]
    _assert_same_state(a, b)
    ea, eb = a.ema_state_dict(), b.ema_state_dict()
    assert all(torch.equal(ea[k], eb[k]) for k in ea) and a._ema_steps == b._ema_steps == 8
    ca = torch.load(ema_runs["A_ck"].last_model_path, map_location="cpu")
    cb = torch.load(ema_runs["B_ck"].last_model_path, map_location="cpu")
    for key in ("state_dict", "raw_state_dict"):
        assert all(torch.equal(ca[key][k], cb[key][k]) for k in ca[key]), key
    assert ca["ema"] == cb["ema"]


def test_ema_checkpoint_written_at_the_end_of_validation_still_holds_the_raw_weights(tmp_path):
    """ModelCheckpoint's `on_validation_end` (a callback hook, run BEFORE the module's) writes `last.ckpt` while the live
    tensors hold the EMA: its raw_state_dict must still be the trained weights, its state_dict the EMA."""
    cfg = _fit_cfg(train={"EMA_DECAY": 0.9})
    torch.manual_seed(0)
    m = _randomize(MNISTModel(copy.deepcopy(cfg)))
    probe = _Probe()
    tr, ck = _trainer(tmp_path / "v", 2, every=1000, callbacks=[probe])         # one validation, after the 2nd optimiser step
    tr.fit(m, *_loaders())
    assert len(probe.during) == 1 and probe.during[0] == (True, False)
    c = torch.load(ck.last_model_path, map_location="cpu")
    assert c["global_step"] == 2 and c["ema"]["steps"] == 2
    live = m._ema_live()
    ema = m.ema_state_dict()
    assert all(torch.equal(c["raw_state_dict"][k], v) for k, v in live.items())
    assert all(torch.equal(c["raw_state_dict"][k], probe.raw_before[k]) for k in probe.raw_before)
    assert all(torch.equal(c["state_dict"][k], ema[k]) for k in ema)
    assert any(not torch.equal(c["state_dict"][k], c["raw_state_dict"][k]) for k in live)


def test_ema_validate_with_a_checkpoint_keeps_the_ema_weights(ema_runs, tmp_path):
    """`trainer.validate(ckpt_path=...)` is not FITTING: the checkpoint's state_dict (the EMA) is what gets loaded."""
    c = torch.load(ema_runs["A_ck"].last_model_path, map_location="cpu")
    m = MNISTModel(copy.deepcopy(ema_runs["cfg"]))
    tr = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False,
                    enable_checkpointing=False, default_root_dir=str(tmp_path))
    tr.validate(m, _loaders()[1], ckpt_path=ema_runs["A_ck"].last_model_path)
    sd = m.state_dict()
    assert all(torch.equal(sd[k], c["state_dict"][k]) for k in sd)


def test_ema_resuming_a_non_ema_checkpoint_starts_the_ema_at_its_weights(tmp_path):
    """A checkpoint without raw_state_dict (any earlier run): its weights are the raw weights and the EMA's start."""
    cfg0 = _fit_cfg()
    torch.manual_seed(0)
    m0 = _randomize(MNISTModel(copy.deepcopy(cfg0)))
    tr, ck = _trainer(tmp_path / "p", 4)
    tr.fit(m0, *_loaders())
    c0 = torch.load(ck.last_model_path, map_location="cpu")
    assert "raw_state_dict" not in c0
    m1 = MNISTModel(copy.deepcopy(_fit_cfg(train={"EMA_DECAY": 0.9})))
    seen = {}

    class _At(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if "live" not in seen:                      # after the module's on_train_start built the EMA
                seen["live"] = {k: v.clone() for k, v in pl_module.state_dict().items()}
                seen["ema"] = {k: v.clone() for k, v in pl_module._ema.items()}
                seen["steps"] = pl_module._ema_steps

    tr, _ = _trainer(tmp_path / "q", 8, callbacks=[_At()])
    tr.fit(m1, *_loaders(), ckpt_path=ck.last_model_path)
    assert seen["steps"] == 0
    assert all(torch.equal(seen["live"][k], c0["state_dict"][k]) for k in seen["live"])
    assert all(torch.equal(seen["ema"][k], c0["state_dict"][k]) for k in seen["ema"])
    c1 = torch.load(tmp_path / "q" / "last.ckpt", map_location="cpu")
    assert c1["ema"]["steps"] == 4 and "raw_state_dict" in c1


def test_ema_an_ema_free_run_that_resumes_an_ema_checkpoint_trains_from_the_raw_weights(ema_runs, tmp_path):
    at = {}

    class _At(Callback):
        def on_train_start(self, trainer, pl_module):
            at["live"] = {k: v.clone() for k, v in pl_module.state_dict().items()}

    m = MNISTModel(copy.deepcopy(_fit_cfg(train={"WEIGHT_DECAY": 0.01})))             # same optimiser, no EMA
    tr, _ = _trainer(tmp_path / "r", 6, callbacks=[_At()])
    tr.fit(m, *_loaders(), ckpt_path=ema_runs["B1_last"])
    c1 = ema_runs["B1_ckpt"]
    assert all(torch.equal(at["live"][k], c1["raw_state_dict"][k]) for k in at["live"])
