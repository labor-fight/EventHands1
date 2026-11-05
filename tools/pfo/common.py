"""PFO (pfasrdz-focused optimisation, docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103): constants, locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 2-7 only; GPU 0-1 are never used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STUDY = "PFO_20261103"
OUT = ROOT / "outputs/pfo/20261103"
CONFIGS = ROOT / "configs/pfo_20261103"
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
PAIRS = ((2, 3), (4, 5), (6, 7))
SM = ROOT / "outputs/s45sm/20261031"
SM_CONFIGS = ROOT / "configs/s45sm_20261031"
XAG_SOURCE_ROOT = ROOT / "outputs/xag/sources"
SR1 = SM / "routers/sr1/sr1_final.ckpt"
T0P = ROOT / "outputs/s45rt/20261031/t0p/t0p_final.ckpt"
SRDWKDA = SM / "routers/srdwkda/srdwkda_final.ckpt"
ZGZ = ("zgz_global", "zgz_local")
COUNTS = {"zgz_global": 1386, "zgz_local": 1204}
TRAIN_LOCAL = ("ch_local", "lfz_local", "lpc_local", "lr_local", "ly_local", "lyh_local", "lyq_local", "ycy_local",
               "ylf_local")

# S43 (s43n_sp, W50 raw protocol main row outputs/semkine/s43n_sp_tf_pert_main_row.json): the comparator.
S43_SEEDS = (3407, 3408, 3409)
S43_RUN = ROOT / "outputs/semkine/s43n_sp_s{seed}"
S43_CONFIG = ROOT / "configs/s43/s43n_sp.yaml"
S43_ROW = ROOT / "outputs/semkine/s43n_sp_tf_pert_main_row.json"


def _sm_arm(arm, config_arm=None):
    run = f"s45sm_{arm}_s3407"
    return {"kind": "routed", "checkpoint": SM / "runs" / run / f"{run}-step=6000.ckpt",
            "config": SM_CONFIGS / f"s45sm_{config_arm or arm}_s3407.yaml"}


#: existing networks the study reads (zero training): name -> checkpoint, config and kind
NETS = {
    "pfasrdz": _sm_arm("pfasrdz"),
    "pfat0pdz_cos": _sm_arm("pfat0pdz_cos"),
    "xt0cos_srdwa": _sm_arm("pfat0pdz_cos", "xt0cos_srdwa"),
    "pfagtdz": {**_sm_arm("pfagtdz"), "kind": "label"},
    **{f"s43_s{s}": {"kind": "s43", "checkpoint": Path(str(S43_RUN).format(seed=s)) / f"s43n_sp_s{s}-step=6000.ckpt",
                     "config": Path(str(S43_RUN).format(seed=s)) / "config_resolved.yaml"} for s in S43_SEEDS},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(value):
    return json.loads(path(value).read_text())


def save(destination, value, *, exclusive=False):
    p = path(destination)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with p.open("x") as stream:
            stream.write(data)
        return p
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(data)
    tmp.replace(p)
    return p


def require(value, message):
    if not value:
        raise RuntimeError(message)


def require_one_gpu():
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
