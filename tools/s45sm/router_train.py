"""Router training and decoding variants (GPU: two-GPU DDP pair among 2-7; fresh-only outputs; seed 3407).

One trainer for every router the study trains, chosen by the registry spec (`kind`, `heldout`, `aug`, `kd`, `params`,
`anneal_of` + `anneal_steps`):
  sr   semkine/s45sm.py SRNet (137,028 parameters), S45-RT T0P's recipe: S45's training dataset and augmentation (with
       S45-AT dz's depth scale mode when `dz`) with the XAG per-event sources, minus every sequence of the held-out subject
       (`heldout`; null = all nine training subjects, the deployment router); zgz is never read. Two-GPU DDP, 512 packets per
       GPU, Adam 1e-3, 500-step linear warmup then constant, BF16 autocast, seed 3407; 50 steps (preflight) or `steps`.
       `aug` adds router-only augmentation (per packet a rotation of +-rot_deg and a scale in `scale` about the packet's
       event centroid, then each event kept with a per-packet probability in `keep`; labels travel with their events).
       `kd` adds distillation from the frozen T0P (part KL at temperature T, rest-template point smooth-L1 where the
       teacher says on-hand, on-hand BCE; total = GT + lambda_part * part + lambda_coord * coord + lambda_onhand * onhand).
  srp, dw   the slim patch routers (semkine/s45sm_srp.py 137,787 parameters, semkine/s45sm_dw.py 137,291): T0P's own
       targets and loss (semkine/s45rt_router.py: patch cross-entropy, rest-template point, on-hand BCE) and `kd` on the
       279-patch distribution (lambda_patch). The parameter count is asserted against `params`.
  anneal_of / anneal_steps   continue a trained router's own recipe from its final checkpoint, unchanged, for n steps with a
       fresh Adam whose learning rate is warmed up over 100 steps to the base rate and then decays linearly to 0 at step n;
       structure and parameter count stay the base router's (kinds sr, srp, dw).
Writes routers/<name>[_preflight]/<name>_final.ckpt and its completion JSON. `decode_variant` (no training) links a
part-first decoding variant (decode_of; kinds srph / dwh / t0ph of semkine/s45sm_hier.py) to its base router's checkpoint.
Seed placement is per variant, as it was (router training sets no deterministic flag): the T0P teacher of an `sr` router
is built after the student; the patch routers' teacher before the output directory; an annealed router loads its weights
and never reseeds.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from .paths import S45_CONFIG, now, require, save, sha  # noqa: E402  (also puts the repository and model/ on sys.path)
from .protocol import verify_protocol  # noqa: E402
from .registry import router_checkpoint, router_dir, router_ready, router_spec  # noqa: E402
import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.strategies import DDPStrategy  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from config import load_config  # noqa: E402
from semkine import s45rt_router as RT  # noqa: E402
from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_dw as DW  # noqa: E402
from semkine import s45sm_srp as SRP  # noqa: E402
from semkine.dataset import build_dataset  # noqa: E402
from semkine.events import EV_BATCH, EV_X, EV_Y, collate_packets  # noqa: E402
from semkine.s45sm import HEIGHT as RT_H, WIDTH as RT_W  # noqa: E402
from tools.s45tr.common import XAG_SOURCES  # noqa: E402

LR, WARMUP = 1e-3, 500
ANNEAL_WARMUP = 100
XAG_FIELDS = ("xag_source_index", "xag_source_ms", "xag_part", "xag_face", "xag_bary", "xag_valid")
DECODE_KINDS = {"srph": "srp", "dwh": "dw", "t0ph": "t0p"}
SAME = ("kind", "heldout", "dz", "kd", "aug")


def dataset(cfg, heldout):
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    if heldout is None:
        require(len(ds.sequences) == 72, "training split changed")
        return ds
    held = np.array([s.startswith(heldout + "_") for s, _ in ds.sequences])
    keep = np.flatnonzero(~held[ds.index[:, 0]])
    require(held.sum() == 8 and 0 < keep.size < len(ds), "held-out subject selection changed")
    return Subset(ds, keep.tolist())


@torch.no_grad()
def augment(batch, aug, generator):
    """Router-only augmentation (see the module docstring); returns a new batch with consistent per-event fields."""
    ev = batch.events.clone()
    B = batch.batch_size
    dev = ev.device
    b = ev[:, EV_BATCH].long()
    rot = (torch.rand(B, generator=generator, device=dev) * 2 - 1) * float(aug["rot_deg"]) * torch.pi / 180
    lo, hi = aug["scale"]
    sc = lo + (hi - lo) * torch.rand(B, generator=generator, device=dev)
    klo, khi = aug["keep"]
    kp = klo + (khi - klo) * torch.rand(B, generator=generator, device=dev)
    counts = batch.counts.clamp_min(1).float()
    cx = torch.zeros(B, device=dev).index_add_(0, b, ev[:, EV_X]) / counts
    cy = torch.zeros(B, device=dev).index_add_(0, b, ev[:, EV_Y]) / counts
    dx, dy = ev[:, EV_X] - cx[b], ev[:, EV_Y] - cy[b]
    c, s = torch.cos(rot)[b], torch.sin(rot)[b]
    x = cx[b] + sc[b] * (c * dx - s * dy)
    y = cy[b] + sc[b] * (s * dx + c * dy)
    keep = (x >= 0) & (x <= RT_W - 1) & (y >= 0) & (y <= RT_H - 1)
    keep &= torch.rand(ev.shape[0], generator=generator, device=dev) < kp[b]
    ev[:, EV_X], ev[:, EV_Y] = x.round(), y.round()
    fields = {k: v for k, v in batch.__dict__.items()}
    fields["events"] = ev[keep]
    for name in XAG_FIELDS:
        if fields.get(name) is not None:
            fields[name] = fields[name][keep]
    kept = torch.bincount(b[keep], minlength=B)
    fields["ptr"] = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(kept, 0)])
    return type(batch)(**fields)


def patch_parts(teacher):
    """(P,) part of every T0P patch (patches never cross parts; checked)."""
    pof, fp = teacher.patch_of_face.cpu(), teacher.face_part.cpu()
    out = torch.full((teacher.n_patches,), -1, dtype=torch.long)
    for k in range(teacher.n_patches):
        parts = fp[pof == k].unique()
        require(parts.numel() == 1, "a T0P patch crosses parts")
        out[k] = parts[0]
    return out


def teacher_part_prob(patch_part, logits, T):
    """The teacher's part distribution at temperature T: its patch softmax summed over each part's patches."""
    prob = torch.softmax(logits.float() / T, -1)
    out = torch.zeros(*prob.shape[:-1], SM.N_PARTS, dtype=prob.dtype, device=prob.device)
    return out.scatter_add(-1, patch_part.to(prob.device).expand_as(prob), prob)


def kd_loss_part(out, tout, patch_part, kd):
    """Distillation of an `sr` router from T0P: part KL (the teacher's patches summed per part), coord, on-hand."""
    mask = out["mask"]
    T = float(kd["T"])
    t_prob = teacher_part_prob(patch_part, tout["patch"], T)[mask]
    s_logp = torch.log_softmax(out["part"][mask].float() / T, -1)
    part = (torch.xlogy(t_prob, t_prob) - t_prob * s_logp).sum(-1).mean() * T * T
    t_on = torch.sigmoid(tout["onhand"].float())
    onhand = F.binary_cross_entropy_with_logits(out["onhand"][mask].float(), t_on[mask])
    sel = mask & (t_on >= 0.5)
    coord = (F.smooth_l1_loss(out["coord"][sel].float(), tout["coord"][sel].float()) if bool(sel.any())
             else onhand * 0.0)
    parts = {"kd_part": part, "kd_coord": coord, "kd_onhand": onhand}
    total = (float(kd["lambda_part"]) * part + float(kd["lambda_coord"]) * coord
             + float(kd["lambda_onhand"]) * onhand)
    return total, parts


def kd_loss_patch(out, tout, kd):
    """Distillation of a patch router (srp, dw) from T0P: patch KL on the 279-patch distribution, coord, on-hand."""
    mask = out["mask"]
    T = float(kd["T"])
    t_prob = torch.softmax(tout["patch"][mask].float() / T, -1)
    s_logp = torch.log_softmax(out["patch"][mask].float() / T, -1)
    patch = (torch.xlogy(t_prob, t_prob) - t_prob * s_logp).sum(-1).mean() * T * T
    t_on = torch.sigmoid(tout["onhand"].float())
    onhand = F.binary_cross_entropy_with_logits(out["onhand"][mask].float(), t_on[mask])
    sel = mask & (t_on >= 0.5)
    coord = (F.smooth_l1_loss(out["coord"][sel].float(), tout["coord"][sel].float()) if bool(sel.any())
             else onhand * 0.0)
    parts = {"kd_patch": patch, "kd_coord": coord, "kd_onhand": onhand}
    total = (float(kd["lambda_patch"]) * patch + float(kd["lambda_coord"]) * coord
             + float(kd["lambda_onhand"]) * onhand)
    return total, parts


def factor(step, total):
    """Learning-rate factor of an annealed router: linear warmup over ANNEAL_WARMUP steps, then linear decay to 0 at `total`."""
    return min(1.0, (step + 1) / ANNEAL_WARMUP) * max(0.0, 1.0 - step / total)


class _RouterBase(pl.LightningModule):
    def __init__(self, net, aug=None, teacher=None, kd=None):
        super().__init__()
        self.net = net
        self.aug = aug
        self.generator = None
        self.kd = kd
        self._teacher = [teacher]                                                     # not a submodule: no DDP, no state

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def _augmented(self, batch):
        if self.aug:
            if self.generator is None or self.generator.device != batch.events.device:
                self.generator = torch.Generator(device=batch.events.device)
            self.generator.manual_seed(3407 * 1000003 + int(self.global_step) * 17 + int(self.global_rank))
            batch = augment(batch, self.aug, self.generator)
        return batch

    def _teacher_out(self, batch, out):
        teacher = self._teacher[0]
        if teacher.faces.device != batch.events.device:
            self._teacher[0] = teacher = teacher.to(batch.events.device)
        with torch.no_grad():
            tout = teacher(batch.events, batch.ptr, batch.delta_t_s)
        require(torch.equal(tout["src"], out["src"]) and torch.equal(tout["mask"], out["mask"]),
                "teacher and student nodes differ")
        return tout

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class SRModule(_RouterBase):
    """Kind sr: semkine/s45sm.py targets and loss; an optional T0P teacher (part KL)."""

    def __init__(self, net, aug=None, teacher=None, kd=None):
        super().__init__(net, aug, teacher, kd)
        self.patch_part = patch_parts(teacher) if teacher is not None else None

    def training_step(self, batch, idx):
        batch = self._augmented(batch)
        out = self.net(batch.events, batch.ptr, batch.delta_t_s)
        tgt = SM.targets(self.net, batch, out["src"], out["mask"])
        total, parts = SM.loss(self.net, out, tgt)
        if self._teacher[0] is not None:
            tout = self._teacher_out(batch, out)
            kd_total, kd_parts = kd_loss_part(out, tout, self.patch_part, self.kd)
            total = total + kd_total
            parts.update(kd_parts)
        self.log("train_loss", total, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log("train_" + k, v, sync_dist=True)
        with torch.no_grad():
            v = tgt["valid"]
            if bool(v.any()):
                self.log("train_part_acc", (out["part"].argmax(-1) == tgt["part"])[v].float().mean(), sync_dist=True)
                err = (out["coord"].float() * 0.01 - tgt["point"]).norm(dim=-1)[v]
                self.log("train_coord_err_mm", err.median() * 1000, sync_dist=True)
        return total


class PatchModule(_RouterBase):
    """Kinds srp / dw: semkine/s45rt_router.py targets and loss; an optional T0P teacher (patch KL)."""

    def training_step(self, batch, idx):
        batch = self._augmented(batch)
        out = self.net(batch.events, batch.ptr, batch.delta_t_s)
        tgt = RT.targets(self.net, batch, out["src"], out["mask"])
        total, parts = RT.loss(self.net, out, tgt)
        if self._teacher[0] is not None:
            tout = self._teacher_out(batch, out)
            kd_total, kd_parts = kd_loss_patch(out, tout, self.kd)
            total = total + kd_total
            parts.update(kd_parts)
        self.log("train_loss", total, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log("train_" + k, v, sync_dist=True)
        with torch.no_grad():
            v = tgt["valid"]
            if bool(v.any()):
                self.log("train_patch_acc", (out["patch"].argmax(-1) == tgt["patch"])[v].float().mean(), sync_dist=True)
                err = (out["coord"].float() * 0.01 - tgt["point"]).norm(dim=-1)[v]
                self.log("train_coord_err_mm", err.median() * 1000, sync_dist=True)
        return total


class _Anneal:
    anneal_steps = None

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        total = int(self.anneal_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: factor(s, total))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class SRAnneal(_Anneal, SRModule):
    pass


class PatchAnneal(_Anneal, PatchModule):
    pass


#: kind -> build, load, save_checkpoint, training module, anneal module, name of the parameter assertion
KINDS = {
    "sr": {"build": SM.build, "load": SM.load, "save": SM.save_checkpoint, "module": SRModule, "anneal": SRAnneal},
    "srp": {"build": SRP.build, "load": SRP.load, "save": SRP.save_checkpoint, "module": PatchModule, "anneal": PatchAnneal},
    "dw": {"build": DW.build, "load": DW.load, "save": DW.save_checkpoint, "module": PatchModule, "anneal": PatchAnneal}}


class Finish(pl.Callback):
    """Writes the checkpoint and the completion JSON at the end of training (rank 0)."""

    def __init__(self, target, phase, save_fn, meta, anneal=False):
        self.target, self.phase, self.save_fn, self.meta, self.anneal = target, phase, save_fn, meta, anneal

    def on_train_end(self, trainer, module):
        if trainer.global_rank == 0:
            params = SM.n_params(module.net)
            step = int(trainer.global_step)
            if self.anneal:
                self.save_fn(module.net, self.target, step=step, phase=self.phase, lr=LR, anneal_warmup=ANNEAL_WARMUP,
                             params=params, **self.meta)
                extra = self.meta
            else:
                self.save_fn(module.net, self.target, step=step, phase=self.phase, lr=LR, warmup=WARMUP, params=params,
                             **self.meta)
                extra = {}
            save(self.target.with_suffix(".json"), {"status": "COMPLETE", "step": step, "phase": self.phase,
                                                    "checkpoint_sha256": sha(self.target), "params": params, **extra,
                                                    "completed_at_utc": now()})


def anneal_spec(name):
    spec = router_spec(name)
    require("anneal_of" in spec and int(spec["anneal_steps"]) > ANNEAL_WARMUP, "not an annealed router: " + name)
    base = router_spec(spec["anneal_of"])
    require(base["kind"] in KINDS and "anneal_of" not in base, "annealed routers continue a trained sr / srp / dw router")
    require({k: spec.get(k) for k in SAME} == {k: base.get(k) for k in SAME}, "an annealed router keeps its base recipe")
    return spec, base


def decode_spec(name):
    spec = router_spec(name)
    require(spec["kind"] in DECODE_KINDS and "decode_of" in spec, "not a decoding variant: " + name)
    base = router_spec(spec["decode_of"])
    require(base["kind"] == DECODE_KINDS[spec["kind"]] and "decode_of" not in base,
            "a decoding variant decodes a trained router of its base kind")
    return spec, base


def decode_variant(router):
    """A decoding variant has no weights of its own: routers/<name>/<name>_final.ckpt is a symbolic link to the base
    router's ready checkpoint (same bytes, so router_ready verifies the same SHA), with a completion JSON naming the base."""
    verify_protocol()
    spec, base = decode_spec(router)
    source = router_ready(spec["decode_of"]).resolve()
    out, target = router_dir(router), router_checkpoint(router)
    require(not out.exists(), "Fresh-only output already exists: " + str(out))
    out.mkdir(parents=True)
    os.symlink(source, target)
    save(target.with_suffix(".json"), {"status": "COMPLETE", "phase": "formal", "step": None,
                                       "checkpoint_sha256": sha(target), "decode_of": spec["decode_of"],
                                       "kind": spec["kind"], "source": str(source), "completed_at_utc": now()},
         exclusive=True)
    require(router_ready(router) == target, "decoding variant not ready")
    print("READY", router, "->", source, flush=True)
    return target


def _teacher(spec):
    require(router_spec(spec["kd"]["teacher"])["kind"] == "t0p", "the teacher is a T0P router")
    teacher, _ = RT.load(router_ready(spec["kd"]["teacher"]))
    for q in teacher.parameters():
        q.requires_grad_(False)
    return teacher.eval()


def _fresh_output(router, phase):
    out, target = router_dir(router, phase), router_checkpoint(router, phase)
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        require(not out.exists(), "Fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    return out, target


def _loader(train):
    return DataLoader(train, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True,
                      prefetch_factor=2, drop_last=True, collate_fn=collate_packets)


def _fit(module, loader, out, steps, finish):
    trainer = pl.Trainer(accelerator="gpu", devices=2, precision="bf16", max_steps=steps,
                         strategy=DDPStrategy(find_unused_parameters=False), enable_progress_bar=False,
                         logger=pl.loggers.TensorBoardLogger(str(out), name="logs"), log_every_n_steps=20,
                         enable_checkpointing=False, callbacks=[finish], num_sanity_val_steps=0)
    trainer.fit(module, train_dataloaders=loader)


def _data(spec):
    cfg = load_config(str(S45_CONFIG))
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    if spec.get("dz"):
        cfg["AUG"]["DOMRAND"]["SCALE_MODE"] = "depth"                  # the arms' (S45-AT dz) augmentation
    return cfg


def train(router, phase):
    """Train a router fresh (see the module docstring); `anneal_of` routers continue their base."""
    verify_protocol()
    spec = router_spec(router)
    if "anneal_of" in spec:
        return anneal(router, phase)
    kind = spec["kind"]
    require(kind in KINDS, f"routers of kind {kind} are not trained here")
    k = KINDS[kind]
    if spec.get("kd"):
        require(router_spec(spec["kd"]["teacher"])["kind"] == "t0p", "the teacher is a T0P router")
    teacher = _teacher(spec) if spec.get("kd") and kind != "sr" else None       # patch routers: before the seeds
    out, target = _fresh_output(router, phase)
    pl.seed_everything(3407, workers=True)
    cfg = _data(spec)
    loader = _loader(dataset(cfg, spec["heldout"]))
    torch.manual_seed(3407)
    net = k["build"](cfg["MANO"]["NPZ"])
    if kind != "sr":
        require(SM.n_params(net) == int(spec["params"]), f"{kind} parameter count differs from the registration")
    elif spec.get("kd"):
        teacher = _teacher(spec)                                                  # sr: after the student
    steps = int(spec["steps"]) if phase == "formal" else 50
    module = k["module"](net, spec.get("aug"), teacher, spec.get("kd"))
    meta = {"heldout": spec["heldout"]}
    if kind != "sr" or spec.get("kd"):
        meta["kd"] = module.kd
    _fit(module, loader, out, steps, Finish(target, phase, k["save"], meta))


def anneal(router, phase):
    spec, base = anneal_spec(router)
    k = KINDS[base["kind"]]
    base_ckpt = router_ready(spec["anneal_of"])
    net, _ = k["load"](base_ckpt)
    net.train()
    for q in net.parameters():
        q.requires_grad_(True)
    if "params" in spec:
        require(SM.n_params(net) == int(spec["params"]), "parameter count differs from the registration")
    teacher = _teacher(base) if base.get("kd") else None
    out, target = _fresh_output(router, phase)
    pl.seed_everything(3407, workers=True)
    cfg = _data(base)
    loader = _loader(dataset(cfg, base["heldout"]))
    steps = int(spec["anneal_steps"]) if phase == "formal" else 50
    module = SRAnneal(net) if base["kind"] == "sr" else PatchAnneal(net, base.get("aug"), teacher, base.get("kd"))
    module.anneal_steps = steps
    meta = {"anneal_of": spec["anneal_of"], "base_checkpoint_sha256": sha(base_ckpt), "heldout": base["heldout"]}
    _fit(module, loader, out, steps, Finish(target, phase, k["save"], meta, anneal=True))
