# Failure cases and amended gates

Recorded so they are not rediscovered as bugs.

## Protocol

- **`jitter_static`.** Supported by 1 frame on `zgz_global` and 0 on `zgz_local`. Removed
  from every gate from S1 on. Replaced by the `quiet` bucket.
- **Legacy overall metric.** `zgz_local` supplies 1204 of 2590 val frames at ~1/20 the event
  rate of `zgz_global`. Subject-disjoint splits and named stress buckets are mandatory.
- **`val_loss` selection.** Spearman −0.07 vs closed-loop RA. Checkpoints are saved on a
  fixed step grid and selected by recursive RA on `val_core`.

## Geometry and information

- **Splat IoU ≥ 0.995.** The point splat cannot cover a surface. Replaced by containment of
  splatted vertices in the raster plus agreement with a brute-force rasteriser.
- **Root-only Schur −60% finger IG.** Measured drop was 5.1%. `Lambda` does not know which
  DoF moved. Replaced by support contrast and normal-equation recovery.
- **Cholesky-only Fisher solve.** One view leaves hidden joints with no rows. S14 uses a
  ridge least-squares solve on the active set.

## Losses and heads

- **`so3_trans_fk` without an absolute term.** Root-relative FK is blind to translation;
  SmoothL1 at β=1 m puts centimetre errors in the quadratic region. Absolute MPJPE degraded
  63.7 → 83.6 mm. Remedy: `ABS_FK_WEIGHT` and/or `TRANS_BETA≈0.01`.
- **Zero-init additive semantic / joint head.** Direct 51D path steals the gradient
  (KSGN +0.16 mm). S10 makes the per-joint decoder the only path; same-checkpoint ablation
  is the non-laziness check.

## Active estimation

- **Motion is not sparse at 50 ms.** Median joint motion 4.54°; 2.5% of joints move <1°.
  "Skip joints that are not moving" describes nothing in this data. Restated as a
  bias–variance trade at short windows. A FAIL at 50 ms alone does not stop the active
  mainline; FAIL at short steps too would.
- **Oracle vs δ-trust at 50 ms.** Oracle 0.10 reaches 18.52 mm vs δ-trust 18.73 mm on two
  `zgz` sequences; CI vs δ-trust crosses 0. Almost all realisable gain is a global constant.
- **Over-skipping.** Oracle 0.40 drives jitter to 0.225 mm and RA to 25.3 mm. Jitter is not
  an objective.

## Manifold

- **SE(3) fusion of a 0.6 rad wrist disagreement** lands 5.9 mm off the translational
  midpoint. Correct under that metric, wrong for a millimetre vertex error. Default fusion
  metric is decoupled R³×SO(3).
- **Error transport uses the adjoint, not Jr⁻¹.** The first filter derivation failed the
  transport test at ratio 2.0.

## Degenerate labels

- ~0.9% of frames have the fitted hand straddling the image plane (min vertex depth ≈ −5 cm).
  KSSF drops those faces via a 5 cm near plane. Do not use those frames as method evidence;
  report the rate in the per-sequence table.
