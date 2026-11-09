"""TPD stage STR (docs/EXPERIMENT_RECORDS.md#tpd-str-prereg-20261106): rolk's recipe -- B2's fine-tune of the same seed
with K = 5 packets and the gradient through the prior, 1500 steps, peak 1e-3, cosine to 0 -- in the streaming input form
(semkine/tpd_str.py) instead of the causal one. The only change against rolk is the trunk's nodes and tokens (and the
history the batches carry for them); paired with rolk 3407 (same initial weights, same packets draw for draw).

`enter()` (before any other tools.tpd import): ROL-k's namespace (amendment 3: seed 3409, PREFETCH_FACTOR 1 for K >= 5;
arm rolk), then arm `str` added to the synthesis registry, configs under configs/tpd_20261105/str/:
  config  rolk's config of the seed and phase + STR {BUDGET: 2048, WINDOW_MS: 50, TRAIN_CAP: 6144}
          + names / output directory / PFO.ARM / PFO.SECTION
  data    semkine/tpd_str.py datasets (rollout packet 0 with its history, every packet with its stream counter; plain
          validation packets with the raw history) and collate
  model   semkine/tpd_str.py STRModel on rolk's model class (ROL's learning-rate floor 0)
"""
from __future__ import annotations

SECTION = "tpd-str-prereg-20261106"
ARM = "str"
SPEC = {"routes": "router", "causal": True, "k": 5, "grad": True}
STR = {"BUDGET": 2048, "WINDOW_MS": 50, "TRAIN_CAP": 6144}
_STATE = {"on": False}


def enter():
    if _STATE["on"]:
        return
    from tools.tpd import ns_rolk
    ns_rolk.enter()
    from semkine import tpd_final as TF
    from semkine import tpd_str as TS
    from tools.tpd import arms_final as F
    from tools.tpd.common import CONFIGS
    rolk_build = F.build_config

    def build(arm, seed, phase="formal"):
        if arm != ARM:
            return rolk_build(arm, seed, phase)
        cfg = rolk_build(ns_rolk.ARM, seed, phase)
        name = F.run_name(arm, seed, phase)
        cfg["TRAIN"].update(OUTPUT_DIR=str(F.run_dir(arm, seed, phase)), RUN_NAME=name)
        cfg["EVAL"]["OUTPUT_DIR"] = str(F.run_dir(arm, seed, phase) / "eval")
        cfg["STR"] = dict(STR)
        cfg["PFO"].update(ARM=arm, SECTION=SECTION)
        return cfg
    F.build_config = build
    F.ARMS[ARM] = dict(SPEC)
    F.CONFIGS = CONFIGS / "str"
    TF.upgrade_rollout, TF.upgrade_plain, TF.collate_final = TS.upgrade_rollout, TS.upgrade_plain, TS.collate_str
    rolk_class = TF.final_model_class
    TF.final_model_class = lambda: TS.str_model_class(rolk_class())
    _STATE["on"] = True
