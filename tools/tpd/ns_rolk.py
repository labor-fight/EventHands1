"""TPD stage ROL-k (docs/EXPERIMENT_RECORDS.md#tpd-rolk-prereg-20261106): rol5 -- B2's fine-tune of the same seed with
K = 5 packets and the gradient through the prior, 1500 steps, peak 1e-3, cosine to 0 -- trained in the causal deployment
form. Descriptive candidate for the stop point's "best three-seed deployable row"; not the goal's final configuration
(which trains from scratch, L1-style).

`enter()` (before any other tools.tpd import): amendment 3's namespace (seed 3409; PREFETCH_FACTOR 1 for K >= 5), then
arm `rolk` added to the synthesis registry (tools/tpd/arms_final.py), configs under configs/tpd_20261105/rolk/:
  config  the ROL registry's rol5 config of the seed and phase (MODEL.INIT_FROM = B2's accepted step-6000 checkpoint of
          the seed; tools/tpd/arms_rol.py, unchanged) + FINAL {CAUSAL: true, ROUTES: router, ROLLOUT: true}
          + names / output directory / PFO.ARM / PFO.SECTION
  model   semkine/tpd_final.py's FinalModel (causal nodes and tokens, n_prev per packet, ROL's rollout step) with ROL's
          learning-rate floor (cosine to 0) instead of B2's 0.02
"""
from __future__ import annotations

SECTION = "tpd-rolk-prereg-20261106"
ARM = "rolk"
SPEC = {"routes": "router", "causal": True, "k": 5, "grad": True}
FINAL = {"CAUSAL": True, "ROUTES": "router", "ROLLOUT": True}
_STATE = {"on": False}


def enter():
    if _STATE["on"]:
        return
    from tools.tpd import ns_amend3
    ns_amend3.enter()
    from semkine import tpd_final as TF
    from tools.tpd import arms_final as F
    from tools.tpd import arms_rol as A
    from tools.tpd.common import CONFIGS
    native = F.build_config

    def build(arm, seed, phase="formal"):
        if arm != ARM:
            return native(arm, seed, phase)
        cfg = A.build_config("rol5", seed, phase)
        name = F.run_name(arm, seed, phase)
        cfg["TRAIN"].update(OUTPUT_DIR=str(F.run_dir(arm, seed, phase)), RUN_NAME=name)
        cfg["EVAL"]["OUTPUT_DIR"] = str(F.run_dir(arm, seed, phase) / "eval")
        cfg["FINAL"] = dict(FINAL)
        cfg["PFO"].update(ARM=arm, SECTION=SECTION)
        return cfg
    F.build_config = build
    F.ARMS[ARM] = dict(SPEC)
    F.CONFIGS = CONFIGS / "rolk"
    native_cls = TF.final_model_class

    def model_class():
        if "cls" not in _STATE:
            class ROLKModel(native_cls()):
                LR_FLOOR = 0.0                       # ROL's fine-tune schedule: cosine to 0
            _STATE["cls"] = ROLKModel
        return _STATE["cls"]
    TF.final_model_class = model_class
    _STATE["on"] = True
