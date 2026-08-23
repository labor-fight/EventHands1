# SemKine claim matrix (S17)

Registered before the remaining training arms finish. A claim is `PASS` only after the
gate is measured on this machine. `PENDING` is not a euphemism for PASS.

| ID | Claim | Gate | Verdict | Evidence |
|---|---|---|---|---|
| C0 | Frozen `track_render51` step 1000 reproduces 19.2576 mm RA on the legacy val split | bit-identical recursive RA | **PASS** | [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) |
| C1 | Subject-disjoint protocol + buckets do not change the legacy metric when off | extended evaluator vs legacy ≤ 1e-6 | **PASS** | `tests/test_s1_eval.py` |
| C2 | Overall SOTA vs strongest domrand dense baseline, paired CI excludes 0 | full val/test, ≥2 seeds, pooled mean not best-of-N | PENDING | S1 Track+domrand still selecting |
| C3 | RDOR beats δ-trust=0.5, paired CI excludes 0 | same-checkpoint inference | PENDING | `tools/run_semkine.py`; S7 ceiling at 50 ms is ~0.2 mm |
| C4 | S2 raw encoder RA degradation vs A2, 95% CI upper bound < 0.5 mm | paired per-run bootstrap | PENDING | `configs/semkine/s2_raw_track.yaml` |
| C5 | Quiet / single-finger / 30–60 s drift buckets (−15 / −50 / −30 %) | support ≥ 200, not synthetic-only for a main claim | PENDING | `data/hand_data51/buckets/` |
| C6 | S4 absolute FK: non-aligned MPJPE must not worsen > 2 mm vs selected loss | paired retrain, ≥2 seeds | PENDING | `configs/semkine/s4_abs_fk.yaml` |
| C7 | S10 same-checkpoint ablation freezes fingers exactly; live head moves them | architectural, no training | **PASS** | `tests/test_s10_active_head.py` |
| C8 | S6 geometric field: bary, LBS, background, containment, 15-joint localisation | see S6 log | **PASS (geometry)** | `tests/test_s6_kssf.py` |
| C9 | S8 Jacobian matches autograd and central differences; support-contrast + normal equations | see S8 log | **PASS** | `tests/test_s8_jacobian.py` |
| C10 | S12 filter calibrated (Spearman, NIS coverage, PSD) | unit + synthetic trajectory | **PASS (math)** | `tests/test_s12_filter.py` |
| C11 | S13 tangent fusion recovers injected drift; SE(3) metric bias documented | unit | **PASS** | `tests/test_s13_anchor.py` |

Machine-readable copy: `outputs/semkine/s17/claim_matrix.json` after `python tools/run_s17_final.py`.
