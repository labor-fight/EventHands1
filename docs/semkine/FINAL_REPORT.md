# SemKine S17 — status report

This is the honest state of the roadmap, not a paper draft. Numbers below are either
measured on this machine or explicitly marked pending.

## What is frozen

- **S0.** `track_render51` step 1000 = **19.25760436702419 mm** recursive RA-MPJPE on the
  legacy val split, bit-identical to the historical record. Noise floor: ±0.4 mm same-config
  scatter, ≈1.1 mm cross-training, 2.43 mm observed replicate range. 19.26 is a best-of-4
  minimum; the distribution centre is 20.6–20.7 mm. `jitter_static` is retired.
- **S1 protocol.** Subject-disjoint splits (`train` 5 / `val` 2 / `test` 3), `zgz_local`
  held out as a named stress sequence, bucket manifests written from training-split
  quantiles. Evaluator parity with the legacy path holds to 1e-6 when buckets and δ-trust
  are off. µs residuals extracted; `raw_packed` contract tested.
- **S1 absolute arms (val_core, selection by recursive RA).**

  | arm | seed | selected step | val_core RA (mm) |
  |---|---:|---:|---:|
  | abs_base | 3407 | 3000 | 20.71 |
  | abs_base | 3408 | 3000 | 20.33 |
  | abs_domrand | 3407 | 4000 | 19.08 |

  These are *selection* numbers on `val_core`, not reported test numbers. Track arms are
  training; their closed-loop selection is the paper's main dense baseline.

## What the modules do

The method is event-driven kinematic active state estimation. The frontend is a baseline
family, not a contribution.

| Stage | Module | Status |
|---|---|---|
| S2 | `semkine/encoder.py` raw scan + sparse-cell fallback + distillation hook | implemented; tracking parity pending train |
| S3 | `semkine/lie.py` | PASS (21 gates) |
| S4 | `ABS_FK_WEIGHT` + `TRANS_BETA` on `so3_trans_fk` | implemented; retrain pending |
| S6 | `semkine/kssf.py` | PASS geometry (31 gates) |
| S7 | `semkine/oracle.py` | INCONCLUSIVE at 50 ms on 2 sequences; restated as variance, not sparsity |
| S8 | `semkine/jacobian.py` | PASS; Schur-on-motion gate rewritten |
| S9 | `semkine/router.py` | PASS offline; tracking vs δ-trust pending |
| S10 | unique-pathway head on `MNISTModel` | PASS architecture |
| S11 | `semkine/gnn.py` | implemented; battle pending |
| S12 | `semkine/filter.py` | PASS math; adjoint transport correction recorded |
| S13 | `semkine/anchor.py` | PASS; default root metric is decoupled R³×SO(3) |
| S14 | `semkine/gn.py` | implemented; Cholesky-only amended to lstsq |
| S15 | `semkine/packetizer.py` | implemented |
| S16 | `semkine/frontends.py` | implemented; battle pending |
| mainline | `semkine/mainline.py` | routing as zero information, one Fisher matrix |

## Claims that will not be made yet

- Any millimetre-level improvement over the S1 Track+domrand baseline.
- RDOR as a 50 ms win. S7 put the oracle ceiling over δ-trust at ~0.2 mm on two sequences,
  inside the 0.4 mm replicate floor. The active mainline is to be judged at short windows.
- Bucket claims on synthetic-only quiet / occlusion data as main-paper results.

## How to finish the measurements

```bash
# after Track+domrand selection
bash tools/run_s1_delta_trust.sh <selected_track_ckpt> 0.5 val
python tools/run_semkine.py --ckpt <selected_track_ckpt> --step-ms 50
python tools/run_semkine.py --ckpt <selected_track_ckpt> --step-ms 5
python tools/run_s17_final.py
```

S2 training (attempt 1, then fallback):

```bash
python semkine/train.py --config configs/semkine/s2_raw_track.yaml --seed 3407
```
