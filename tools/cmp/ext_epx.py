"""CMP extension arms epxf -> epx (docs/EXPERIMENT_RECORDS.md#cmp-amend6-20261110): EvHandPose with its hand crop,
trained for the online tracker it is evaluated as.

`epc` (amendment 1) diverges online: trained on GT-centred boxes of the true size, it never reads depth from the crop, so
the decoded depth copies the box's and compounds; and it never saw a sequence shorter than 12 packets, which is what the
first updates of a segment are. epxf / epx keep epcf / epc's network, losses, optimiser, schedule, budget and evaluation
(CMP.EVAL.AS: epc, the cropped EvHandPose's adapter of tools/cmp/nets.py: box from the tracker's own state) and change
the training samples only:
  box      semkine/cmp_seq_cropx.py: GT + one prior-noise draw per packet, box side x log-uniform zoom in [1 / 1.5, 1.5]
  history  semkine/cmp_evhpx.py: sequences of 1 .. 12 packets (12 with probability 0.5)
"""
from __future__ import annotations

ZOOM = 1.5
FULL = 0.5
ARMS = {
    "epxf": {"model": "epxf", "stage": "seq", "steps": 3000, "lr": 5.0e-4, "warmup": 200, "batch": 32, "accumulate": 1,
             "workers": 16, "seq": {"K": 1, "VOXEL_BINS": 5}, "deterministic": "warn", "crop": True},
    "epx": {"model": "epx", "stage": "seq", "init": ("epxf", 3000), "steps": 7500, "lr": 5.0e-4, "warmup": 500,
            "batch": 16, "accumulate": 2, "workers": 16, "seq": {"K": 12, "VOXEL_BINS": 5, "ANNOT_LEN": 3}, "crop": True},
}
EVAL = {"epx": {"AS": "epc", "WINDOW_MS": 50, "HISTORY": 12}}
CODE = ("tools/cmp/ext_epx.py", "semkine/cmp_seq_cropx.py", "semkine/cmp_evhpx.py")


def configure(cfg, arm, seed, phase):
    cfg["CMP"]["CROP"]["TRACK"] = {"NOISE": "prior", "ZOOM": ZOOM, "FULL": FULL}
    if arm == "epx":
        cfg["TRAIN"]["WEIGHT_DECAY_L2"] = 1.0e-4                    # the official weight decay, as ep / epc
    return cfg


def build_dataset(ds, cfg, split):
    if split != "train":
        return ds
    from semkine import cmp_seq_cropx as SX
    c = cfg["CMP"]
    s, crop = c["SEQ"], c["CROP"]
    return SX.upgrade(ds, int(s["K"]), int(s["PACKET_MS"]), int(s["VOXEL_BINS"]), int(crop["SIZE"]), float(crop["RATE"]),
                      float(crop["SHIFT"]), float(crop["TRACK"]["ZOOM"]))
