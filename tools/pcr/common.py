"""PCR (prior-conditioned router, docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105): constants, locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 4-7 only while IDP's last training holds GPUs 2-3 (GPU 0-1 never).
Read-only reuse: PFO's frozen engine and r_s2d; IDP's L1 (seed 3407) checkpoint / spec / eval config / readings; PFO2's
B2 (seed 3407); S45-SM's pfasrdz closed-loop state bank. Products under outputs/pcr/20261105/, configs under
configs/pcr_20261105/.
"""
from __future__ import annotations

from tools.pfo.common import (COUNTS, PY, ROOT, XAG_SOURCE_ROOT, ZGZ, now, path, read, require, save, sha)  # noqa: F401

STUDY = "PCR_20261105"
SECTION = "pcr-prereg-20261105"
OUT = ROOT / "outputs/pcr/20261105"
CONFIGS = ROOT / "configs/pcr_20261105"
PHYSICAL_GPUS = (4, 5, 6, 7)
PAIRS = ((4, 5), (6, 7))
SEED = 3407
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
NOISE_RA = 0.8
IDP_OUT = ROOT / "outputs/idp/20261105"
PFO2_OUT = ROOT / "outputs/pfo2/20261103"
R_S2D = ROOT / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"
PFASRDZ_BANK = ROOT / "outputs/s45sm/20261031/banks/pfasrdz"
#: frozen trunks read with the new routers (zero training): name -> (spec, eval config)
TRUNKS = {"l1": (IDP_OUT / "specs/idp_l1_s3407.json", ROOT / "configs/idp_20261105/eval/idp_l1_s3407.yaml"),
          "b2": (PFO2_OUT / "specs/pfo2_fgs_s3407.json", ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml")}
#: L1 with r_s2d through IDP's entry (the reference of engine check K)
L1_DEPLOYED_E1 = IDP_OUT / "readings/idp_l1_s3407/e1"

__all__ = ["CONFIGS", "COUNTS", "IDP_OUT", "L1_DEPLOYED_E1", "NOISE_RA", "OUT", "PAIRS", "PFASRDZ_BANK", "PFO2_OUT",
           "PHYSICAL_GPUS", "PY", "R_S2D", "ROOT", "S43_ENGINE", "SECTION", "SEED", "STUDY", "TRUNKS", "XAG_SOURCE_ROOT",
           "ZGZ", "now", "path", "read", "require", "save", "sha"]
