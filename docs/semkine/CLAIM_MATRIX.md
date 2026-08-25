# SemKine claim matrix (S17 + KEG S18–S22)

Registered before the arms finished. A claim is `PASS` only after the gate is measured on this
machine. `PENDING` is not a euphemism for PASS, and a `FAIL` here is a result, not a to-do.

## Framework claims (S0–S17)

| ID | Claim | Gate | Verdict | Evidence |
|---|---|---|---|---|
| C0 | Frozen `track_render51` step 1000 reproduces 19.2576 mm RA on the legacy val split | bit-identical recursive RA | **PASS** | [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) |
| C1 | Subject-disjoint protocol + buckets do not change the legacy metric when off | extended evaluator vs legacy ≤ 1e-6 | **PASS** | `tests/test_s1_eval.py` |
| C2 | Active mainline beats the strongest domrand dense baseline, paired CI excludes 0 | full test split, pooled mean not best-of-N | **FAIL (null)** | `outputs/semkine/mainline/mainline_test_s1_track_domrand_s3407-step=3500.json`: −0.041 mm [−0.206, +0.077] |
| C3 | RDOR beats δ-trust = 0.5, paired CI excludes 0 | same-checkpoint inference | **FAIL (null)** | same file: +0.079 mm [−0.141, +0.290]; S7 put the oracle ceiling at ~0.2 mm |
| C4 | S2 raw encoder RA degradation vs A2, 95% CI upper bound < 0.5 mm | paired per-run bootstrap | SUPERSEDED | replaced by the KEG arm; `configs/semkine/s2_raw_track.yaml` never carried a main claim |
| C5 | Quiet / single-finger / 30–60 s drift buckets (−15 / −50 / −30 %) | support ≥ 200, not synthetic-only for a main claim | NOT CLAIMED | `data/hand_data51/buckets/`; no arm cleared C2, so bucket deltas have no winner to attribute |
| C6 | S4 absolute FK: non-aligned MPJPE must not worsen > 2 mm vs selected loss | paired retrain, ≥ 2 seeds | PENDING | `configs/semkine/s4_abs_fk.yaml` |
| C7 | S10 same-checkpoint ablation freezes fingers exactly; live head moves them | architectural, no training | **PASS** | `tests/test_s10_active_head.py` |
| C8 | S6 geometric field: bary, LBS, background, containment, 15-joint localisation | see S6 log | **PASS (geometry)** | `tests/test_s6_kssf.py` |
| C9 | S8 Jacobian matches autograd and central differences; support-contrast + normal equations | see S8 log | **PASS** | `tests/test_s8_jacobian.py` |
| C10 | S12 filter calibrated (Spearman, NIS coverage, PSD) | unit + synthetic trajectory | **PASS (math)** | `tests/test_s12_filter.py` |
| C11 | S13 tangent fusion recovers injected drift; SE(3) metric bias documented | unit | **PASS** | `tests/test_s13_anchor.py` |

## KEG claims (S18–S22)

| ID | Claim | Gate | Verdict | Evidence |
|---|---|---|---|---|
| G1 | KEG reaches 50 ms parity with the dense arm under distillation | paired val_core CI, 2 seeds | NOT MEASURED | `configs/semkine/s19_keg_50ms.yaml`; the arm was paused for memory and superseded by G2, which answers the same question without the teacher |
| G2 | KEG beats the dense control at short steps, where rasterisation drops events | paired CI excludes 0 at 5/10/20/50 ms, 2 seeds each | **FAIL** | `s20_grid_step5.json`, `s20_grid_step102050.json`: 16 of 16 pairings fail, +6.1 to +12.0 mm the wrong way |
| G3 | The twelve state-conditioned KSSF channels are load-bearing | same checkpoint, one config bit | **PASS** | `s22_g3_kssf_ablation.json`: −85.8 mm [−101.5, −66.5] and −86.5 mm [−103.5, −65.6] |
| G4 | filter + RDOR + anchor help once the step is short | paired CI at 5 ms on the KEG arm | **NOT ENTITLED** | G2 failed, so there is no short-step arm worth stacking on; the 50 ms measurement on the dense arm is C2/C3 and is null |
| G5 | KEG's arithmetic scales with event count, the dense trunk's does not | report-only, no latency claimed | **REPORTED** | `s22_efficiency.json`: 22.5x fewer MAC on the densest sequence, 375x on the sparsest |
| G-a | The deletion-free halo lift costs nothing per window | within 1.1 mm of 11.59 mm single-step | **PASS** | `closed_loop_sensitivity_halo_50ms.json`: 12.11 and 11.16 mm |
| G-b | Removing the visibility deletion flattens the conditioning-sensitivity curve | g-slope at 26 mm < 0.20 | **FAIL** | same file: 0.294 and 0.327 against the gated lift's 0.30 and the dense arm's 0.148 |
| G-c | The halo lift closes the closed-loop gap | recursive ≤ 21.9 mm or amplification ≤ x2.10 | **FAIL** | same file: 27.77 / 27.90 mm, x2.29 / x2.50 |

## What the KEG line establishes

Not the claim it was built for. G2 fails decisively and in the opposite shape to the prediction:
KEG is rate-invariant (29–31 mm from 5 to 50 ms) but about a fixed point 8 mm worse than the dense
control, and the dense control loses only 2 mm when the step shortens 10x, so the 50 ms window was
never the binding constraint on this dataset.

What is established, with pre-registered gates and paired intervals, is a mechanism: a
state-conditioned sparse frontend pays for its conditioning in the closed loop. Its single-step
error is at parity (0.2–1.2 mm of the dense arm, inside the 1.1 mm cross-training floor), and its
error against conditioning error grows twice as fast (g-slope 0.29–0.33 against 0.148), which is
what sets the loop's fixed point. Four candidate causes were tested and rejected on their own
numbers — event dropping (G5/G2), routing discontinuity (R5, and a retrained soft arm), divergence
(`run_drift_curve.py`), and evidence deletion (G-b) — and the surviving explanation is the one G3
independently confirms from the other side: the geometry channels are load-bearing *because* they
are read at the previous state, and that is exactly why the loop amplifies.

Machine-readable copy: `outputs/semkine/s17/claim_matrix.json` after `python tools/run_s17_final.py`.
