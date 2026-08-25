# SemKine / KEG-Track — status report (S0–S22)

This is the honest state of the roadmap, not a paper draft. Numbers are measured on this machine
or explicitly marked otherwise. The headline is a negative result with a mechanism behind it.

## The headline

**The asynchronous state-conditioned frontend does not beat the dense one on this dataset, and the
reason is not event dropping.** KEG loses to the LNES control by +6.1 to +12.0 mm recursive
RA-MPJPE at every step size from 5 to 50 ms, on both seeds, sixteen paired comparisons with every
interval excluding zero the wrong way (G2). The premise that motivated the whole line — the dense
rasteriser discards 70–88% of events, so a sparse lift should win as the window shortens — is not
supported: the dense control loses only ~2 mm when the step shortens 10x, so the 50 ms window was
never the binding constraint here.

What the line does establish, with pre-registered gates:

- **As a per-window estimator the sparse frontend is at parity.** Teacher-forced single-step error
  10.9–12.1 mm against the dense arm's 10.9, inside the 1.1 mm cross-training resolution S0
  registered. The frontend is not the deficit.
- **The deficit is entirely closed-loop, and it is a sensitivity, not a divergence.** Error against
  *measured* conditioning error grows at slope 0.29–0.33 for KEG and 0.148 for the dense arm. Both
  loops are stable and both forget where they were initialised; KEG simply settles at a higher fixed
  point because its per-step error responds twice as fast to a wrong conditioning state.
- **The state conditioning is simultaneously what makes it work and what makes it fragile.**
  Silencing the twelve KSSF channels on the same checkpoint costs 85.8 mm (G3, PASS): the geometry
  is load-bearing. It is load-bearing *because* it is read at the previous state, which is the same
  property that sets the slope. G3 and G-b are one fact from two sides.
- **Arithmetic scales with events, as designed** (G5, report-only): 22.5x fewer MAC than the dense
  trunk on the densest validation sequence, 375x on the sparsest. No asynchronous runtime is
  implemented, so no latency or energy is claimed, and none of it offsets G2.

Four candidate causes were tested and rejected on their own numbers: event dropping (G2/G5),
routing discontinuity (three independent measurements), divergence (`run_drift_curve.py`), and
evidence deletion (G-b, the halo lift). The surviving explanation is exposure bias on top of a
steep conditioning-sensitivity curve — both arms' recursive steady states sit far above their
independent-noise fixed points (32.6 vs 14.3 mm; 20.9 vs 12.0), and inside the loop both arms'
per-step updates collapse to ~11% of the needed motion.

## What is frozen

- **S0.** `track_render51` step 1000 = **19.25760436702419 mm** recursive RA-MPJPE on the legacy val
  split, bit-identical to the historical record. Noise floor: ±0.4 mm same-config scatter, ≈1.1 mm
  cross-training, 2.43 mm observed replicate range. 19.26 is a best-of-4 minimum; the distribution
  centre is 20.6–20.7 mm. `jitter_static` is retired.
- **S1 protocol.** Subject-disjoint splits (`train` 5 / `val` 2 / `test` 3), `zgz_local` held out as
  a named stress sequence, bucket manifests from training-split quantiles. Evaluator parity with the
  legacy path to 1e-6 with buckets and δ-trust off. µs residuals extracted, `raw_packed` contract
  tested.
- **The dense baseline.** Domain randomisation is the one intervention that moved the metric
  (≈3.7 mm), by fixing the low-event-rate sequences that dominate the split. Selected arms:
  `s1_track_domrand` s3407 step 3500, 20.87 mm val_core / 22.77 mm on the 18-sequence test split.
- **The active mainline is null at 50 ms.** Filter, RDOR router and triggered anchor each move the
  test metric by ≤ 0.04 mm with every interval containing zero (C2, C3). S7's oracle ceiling
  (~0.2 mm) predicted this. The router does cut the update rate to 0.65 at no accuracy cost, so the
  honest framing is a compute result, not an accuracy one. `delta_trust 0.5` is the only visible
  effect and it is on jitter (11.5 → 7.2), not error.

## Where each stage stands

| Stage | Module | Status |
|---|---|---|
| S2 | `semkine/encoder.py` raw scan + sparse-cell + distillation hook | implemented; superseded as a claim by KEG |
| S3 | `semkine/lie.py` | PASS (21 gates) |
| S4 | `ABS_FK_WEIGHT` + `TRANS_BETA` on `so3_trans_fk` | implemented; retrain never carried a claim |
| S6 | `semkine/kssf.py` | PASS geometry (31 gates); mechanism confirmed later by G3 |
| S7 | `semkine/oracle.py` | INCONCLUSIVE at 50 ms; restated as variance, not sparsity |
| S8 | `semkine/jacobian.py` | PASS |
| S9 | `semkine/router.py` | PASS offline; null in tracking (C3) |
| S10 | unique-pathway head on `MNISTModel` | PASS architecture |
| S11 | `semkine/gnn.py` | implemented; TreeConv reused by KEG |
| S12 | `semkine/filter.py` | PASS math; null in tracking (C2) |
| S13 | `semkine/anchor.py` | PASS; 1079 anchors fired, no measurable effect |
| S14 | `semkine/gn.py` | implemented |
| S15 | `semkine/packetizer.py` | implemented |
| S16 | `semkine/frontends.py` | implemented |
| S18 | `semkine/keg.py` | PASS mechanism & budget (20 gates); 141 k params against `raw_scan`'s 166 k |
| S19 | `configs/semkine/s19_keg_50ms.yaml` | NOT MEASURED; paused for memory, superseded by G2 |
| S20 | mixed-window retrain, both arms, both seeds | **FAIL (G2)** at 5 / 10 / 20 / 50 ms |
| S21 | `s21_keg_halo` deletion-free lift | G-a PASS, **G-b FAIL**, G-c FAIL |
| S22 | G3 ablation, G5 accounting, mainline test | G3 PASS, G4 not entitled, G5 reported |

## Claims that will not be made

- Any accuracy win for the sparse/asynchronous frontend over the dense baseline, at any step size.
- Any latency or energy claim. The MAC ratio is arithmetic on paper; no asynchronous runtime exists.
- RDOR, the Lie filter or the triggered anchor as accuracy improvements at 50 ms.
- Bucket claims. No arm cleared the overall gate, so there is no winner to attribute a bucket to.

## What the evidence says to do next

The measurements point at the training distribution, not the architecture. Both arms' per-step
updates collapse inside their own loops while moving 0.77–0.93 of the needed step under
teacher-forced noise, which is exposure bias in its textbook form (Ross & Bagnell 2011; Bengio et
al. 2015). Every noise schedule in this project — S1's and S20b's included — was tuned on
teacher-forced metrics, i.e. on the wrong distribution. A short closed-loop unroll during training
is the one intervention aimed at that term rather than at a symptom, and it applies to the dense
arm as much as to the sparse one; the dense arm's x1.74 echo excess is the larger absolute prize
because it starts from a lower fixed point.

The first attempt at it failed instructively: mixing self-conditioning at a constant probability
from step 0 feeds the network conditioning states from an untrained model, and its optimum is to
ignore the previous state entirely, which it did (42.5 mm single-step, 144.5 mm recursive, kept as
`outputs/semkine/archive/s22_constP_fail_s340*`). The annealed form is training on both seeds; its
deciding gate is not the error but whether the recursive per-step update stops collapsing, since
that defect is shared with the dense arm. Verdicts land in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## How to reproduce the verdicts

```bash
# G2, the main claim, at every step size
python tools/run_keg_arms.py --arm keg_s3407=... --arm lnes_s3407=... --step-ms 5
python tools/run_keg_arms.py --arm keg_s3407=... --arm lnes_s3407=... --step-ms 10,20,50

# G3, same checkpoint, one config bit
python tools/run_keg_arms.py --arm keg_s3407=... --arm keg_s3407_nokssf=... --step-ms 50

# the closed-loop decomposition that produced the mechanism
python tools/run_closed_loop_probe.py --arm keg=... --prev-noise 0,0.5,1,2,4
python tools/run_drift_curve.py --arm keg=... --noise-scale 1.0,4.0

# G5, report-only
python tools/run_keg_efficiency.py
```
