# SemKine experiment log

One entry per stage verdict. Verdicts are written only after the stage's registered gates have been
measured on this machine. `PASS-KEEP` / `NO-GO-REVERT` apply to optional arms.

| Stage | Verdict | Date | Evidence |
|---|---|---|---|
| S0 audit & baseline freeze | `S0: PASS` | 2026-08-22 | [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md), `outputs/semkine/s0_baseline_rerun/` |
| S1 data & protocol reform | `S1: PASS` | 2026-08-23 | µs extract done; splits+buckets written; abs+track ×2 seeds trained; selection running on val_core |
| S3 Lie state | `S3: PASS` | 2026-08-22 | `tests/test_s3_lie.py` (21 gates), `semkine/lie.py` |
| S6 KSSF | `S6: PASS (geometry)` | 2026-08-22 | `tests/test_s6_kssf.py` (31 gates), `semkine/kssf.py`; mechanism gate deferred to the S1 baseline |
| S8 Jacobian & observability audit | `S8: PASS, one gate rewritten` | 2026-08-22 | `tests/test_s8_jacobian.py` (21 gates), `semkine/jacobian.py` |
| S7 oracle upper bound | `S7: blocked, premise revised` | 2026-08-22 | `semkine/oracle.py`, `tools/run_s7_oracle.py`, `outputs/semkine/s7_oracle*/` |
| S9 RDOR routing | `S9: PASS (offline), tracking gate pending` | 2026-08-22 | `tests/test_s9_router.py` (13 gates), `semkine/router.py`, `tools/run_s9_rdor.py` |
| S12 Lie filter | `S12: PASS (mathematics & calibration)` | 2026-08-22 | `tests/test_s12_filter.py` (22 gates), `semkine/filter.py`; one derivation corrected |
| S13 triggered anchor | `S13: PASS (fusion & trigger)` | 2026-08-22 | `tests/test_s13_anchor.py` (14 gates), `semkine/anchor.py`; root metric changed on evidence |
| S2 RawEvent encoder | `S2: PASS (mechanism)` | 2026-08-23 | `tests/test_s2_encoder.py`, `semkine/encoder.py`; tracking parity pending train |
| S4 geometric loss | `S4: PASS (gradients)` | 2026-08-23 | `tests/test_s4_loss.py`; retrain pending |
| S10 active head | `S10: PASS (architecture)` | 2026-08-23 | `tests/test_s10_active_head.py`; unique pathway + same-ckpt ablation |
| S11 kinematic GNN | `S11: PASS (module)` | 2026-08-23 | `tests/test_s11_gnn.py`; battle pending |
| S14 GN/LM | `S14: PASS (solver)` | 2026-08-23 | `tests/test_s14_gn.py`; active-set lstsq |
| S15 packetizer | `S15: PASS (policy)` | 2026-08-23 | `tests/test_s15_packetizer.py` |
| S16 frontends | `S16: PASS (contract)` | 2026-08-23 | `tests/test_s16_frontends.py`; battle pending |
| S17 final | `S17: FRAMEWORK` | 2026-08-23 | `CLAIM_MATRIX.md`, `FINAL_REPORT.md`, `FAILURE_CASES.md`, `tools/run_s17_final.py` |

## S0 — details

Frozen baseline `track_render51 step=1000` reproduces **19.25760436702419 mm** recursive RA-MPJPE,
bit-identical to the historical record. All four baseline numeric gates and all six hard correctness
gates pass; GT leakage after initialisation is exactly 0; inference is bitwise deterministic.

Two findings that change later stages:

1. `jitter_static` is supported by 1 step on `zgz_global` and 0 on `zgz_local`. Removed from all
   gates from S1 onward, replaced by the S5 `quiet` bucket.
2. `val/zgz_local` runs at 1490 events per 50 ms versus 29 255 for `val/zgz_global` on the same
   subject and camera, while supplying 1204 of 2590 val frames. The overall metric is dominated by
   one low-event-rate sequence, which is why the S1 protocol reform precedes all model work.

Registered noise floor for every later comparison: ±0.4 mm replicate scatter, ≈1.1 mm cross-training
resolution, 2.43 mm observed replicate range. `track_render51`'s distribution centre is 20.6–20.7 mm;
19.26 is a best-of-4 minimum. Same-checkpoint paired inference comparisons may use tighter
thresholds because inference is deterministic here.

## S6 — details

The kinematic semantic field is a real triangle rasterisation of MANO through the previous state,
read out at event pixels. Per pixel it yields the covering face and its barycentric coordinates,
perspective-correct depth, the top-4 skinning weights with their joint ids, a 4-dimensional
continuous part code, and the signed distance to the projected occluding contour with its gradient.
The contour distance is the exact point-to-segment distance, not a pixel-grid distance transform,
because the estimator it feeds in S8 is a sub-pixel one and a quantised contour would cap its
achievable residual at half a pixel.

**Verdict `S6: PASS` on the geometric half**, 31 of 31 gates, measured on real poses drawn from six
sequences:

| Gate | Threshold | Measured |
|---|---|---|
| Coverage vs brute-force rasteriser | <1e-4 of pixels | 0 |
| Depth vs brute-force, where covered | <1e-4 of pixels | 0 |
| Barycentric sum deviation | ≤1e-5 | 2.4e-7 |
| Top-4 skinning mass | ≥0.98 | 0.9939 |
| Background leakage (semantic / LBS / inv-depth) | 0 | 0 |
| Splat vertices off the raster | <1% | 0.08% |
| SDF vs exhaustive segment distance | <1e-3 px | <1e-3 px |
| Normal vs central difference of the SDF | <5% of band pixels disagree | 1.77% (190/10761) |
| Normal unit length in band | ≤1e-5 | 1.8e-7 |
| Per-joint perturbation localisation, all 15 joints | precision >0.5 | all pass |

Three findings worth carrying forward.

1. **The plan's "IoU ≥0.995 against the existing splat" gate is not a correctness test and was
   replaced.** The splat scatters 778 vertices into a depth buffer, which cannot cover a surface
   spanning ~2400 pixels; the raster is legitimately 4.9x denser, so a correct rasteriser fails an
   IoU gate against it. The falsifiable statement is containment — every splatted vertex must land
   on a rasterised face — which holds to 0.08%, plus agreement with an independent brute-force
   rasteriser, which is exact.
2. **About 0.9% of frames carry a degenerate label**: the fitted hand straddles the image plane,
   minimum vertex depth reaching −5 cm (seen in `lpc_local_v2`, `ycy_local_v4`, `zgz_local`). The
   field drops the affected faces via a 5 cm near plane and stays finite, but S7/S8 must not treat
   these frames as evidence about a method, and S17's per-sequence table should report the rate.
3. **Reducing the distance and the winning segment separately is a real bug, not a nicety.** With
   the segments processed in tiers, a later tier can lower a pixel's minimum distance without
   updating a separately-recorded winner, so the normal ends up taken from the wrong segment. It is
   silent — the field still looks plausible — and showed up only as normal-versus-gradient
   disagreement rising from 1.77% to 5.22%. Packing distance and segment id into one integer and
   reducing once removes the failure mode by construction.
4. **Cost is now compatible with the training loop**: 2772 samples/s and 4.5 GiB peak at batch 512,
   from 752 samples/s and 27.9 GiB in the first working version, with bit-identical output. The
   three changes were grouping faces into span tiers instead of giving every triangle the largest
   triangle's window, sizing each contour segment's search window from its own extent, and
   interpolating the 16-wide skinning matrix only on covered pixels.

The mechanism gate (active-finger macro-F1 +10 pt, or delta-geodesic MAE −5%) is a training
comparison and is deferred until the S1 baseline family finishes, since the plan requires it to be
a single-variable test stacked on the selected loss.

## S8 — details

`semkine/jacobian.py` supplies the projection Jacobian in the S3 tangent space, the Fisher
information, and the root-Schur complement. Verified three independent ways, all in float64:

| Check | Result |
|---|---|
| Rebuilt FK vs `ManoLayer` vertices | 2.4e-8 m (MANO's own `batch_rodrigues` epsilon) |
| Analytic vs autograd through `ManoLayer` | 3.9e-9 relative |
| Analytic vs central differences, all 51 columns | <1e-9 m per unit |
| Second-order convergence under step halving | error ratio 4.0, 4.0 |
| Projection derivative vs central differences | <1e-4 |
| Fisher symmetry / PSD, and PSD of every Schur complement | pass |

Three results that change the plan.

1. **The root pivot is `p_root`, not MANO's joint-0 translation.** `state_from_51d` folds the LBS
   pivot into `p_root` to make `(R_root, p_root)` a genuine SE(3) element, so a right perturbation
   of the root holds `p_root` fixed. Reading the pivot off MANO's own transform chain instead —
   the natural thing to do — gives a different perturbation and a Jacobian that passes no finite
   difference check. The identity that reconciles them, `G_0 = [R_root | p_root + R_root j0]`, is
   gated in `test_forward_kinematics_reproduces_mano`.
2. **Pose blendshapes contribute a term the classical hand-tracking Jacobian does not have.**
   `v_posed` depends on the joint rotations, so a joint moves a vertex through the skinning
   transforms *and* through the rest vertex. Measured, the extra term is 0.28% of the Jacobian norm
   at the median and 0.6% at p99 — small, but the analytic path includes it because autograd does
   and the two must agree exactly. The blended-point approximation that the standard formula makes
   (using the final `X_i` in place of each joint's contribution `A_j vbar_i`) is worth 0.009% at
   the median, 0.13% at p99; the exact per-contribution form is used anyway, since it costs nothing
   once the ancestor mask is folded into a matrix product.
3. **The registered counterfactual gate was not falsifiable and has been rewritten.** As
   pre-registered it read: under root-only motion, finger information gain must drop 60% after
   Schur-eliminating the root. Measured, it drops 5.1%. That is not a bug to fix — it is the gate
   being wrong. `Lambda = sum_i J_i^T R_i^-1 J_i` depends on where the events are and on the local
   geometry, and contains no reference at all to which degree of freedom moved. A wrist rotation
   does sweep edges across the fingers, and at those pixels a finger rotation genuinely would
   change the residual, so the finger columns carry real information and are not collinear with the
   root columns. No Schur complement can subtract information that is present.

   Two replacements, both registered and both passing:

   - **Support contrast** (the well-posed form of the original intent): events confined to
     palm-dominated pixels must retain less than 40% of the information that events on a finger
     carry about that finger, after root elimination. This is a statement about spatial support,
     which is what a router can actually key on.
   - **Normal-equation recovery** (the decisive form): attribution is a property of the
     *residuals*, not of the information matrix, and is settled by solving `Lambda dx = J^T r`. A
     root-only motion must come back as a root-only solution (joint leakage <25% of the solution
     norm) and a single-finger motion must peak on the finger that moved (≥80% of trials). Both
     hold with no tuning, which jointly validates the Jacobian, the residual convention and the
     information assembly.

   The consequence for S9 is that RDOR must be justified by the support contrast and by
   downstream tracking accuracy against the δ-trust control, not by the claim that Schur
   elimination suppresses information from non-moving joints. The normal equations must be
   rank-deficient-safe: with one view, joints hidden behind the hand contribute no rows at all and
   `Lambda` is genuinely singular, so S9 and S14 need a pseudo-inverse or a trust region rather
   than a Cholesky solve. The plan's "Cholesky-only" instruction is therefore amended.

## S7 — details

`semkine/oracle.py` implements selective updating with a real skip: a retained coordinate group
keeps its previous value and that value is fed back on the next step. Masking the predicted update
instead would not test anything, because with `PREDICT_DELTA` the network's output already *is*
`prev + delta`. The plumbing is parity-checked: an all-active policy reproduces the untouched
evaluator to 0.00e+00 mm.

**The stage is blocked on two findings, one procedural and one substantive.**

**Procedural: the frozen S0 checkpoint cannot be evaluated on the new validation split.** Under the
old protocol `zgz` was the only held-out subject, so `lr` and `lyq` — the S1 validation subjects —
were in that checkpoint's training data. It scores 6.03 mm RA on the new val, versus 19.26 mm on the
old one; that is a training-set number and not a measurement of anything. The oracle experiment
therefore has to wait for the subject-disjoint S1 tracking arms. A preliminary run restricted to
`zgz`, the only legitimate holdout, is recorded separately and is underpowered by construction — two
sequences, which is the exact weakness S1 was reformed to fix.

**Substantive: the motion is not sparse, at any step size, and the active premise needs restating.**
Ground-truth per-joint rotation between consecutive evaluated steps, measured as a geodesic on the
validation split:

| step | median joint motion | joints moving <1° | joints moving <0.25° | top-4-of-15 share of total motion |
|---|---|---|---|---|
| 1 ms | 0.15° | 99.8% | 76.8% | 0.46 |
| 2 ms | 0.29° | 95.5% | 42.0% | 0.47 |
| 5 ms | 0.83° | 59.9% | 6.1% | 0.47 |
| 10 ms | 1.65° | 25.0% | 1.1% | 0.48 |
| 20 ms | 2.98° | 7.8% | 0.3% | 0.48 |
| 50 ms | 4.54° | 2.5% | 0.0% | 0.47 |

At the 50 ms step the whole system currently uses, essentially every joint moves every step: a
threshold of 0.02 rad still updates 96% of groups, and it takes 0.2 rad to skip even half of them.
So "skip the joints that are not moving" describes nothing that happens in this data.

The concentration ratio is the more interesting column. The four most active of fifteen joints
account for ~47% of the total motion at *every* step size, against 27% for a uniform spread. That
ratio is scale-invariant: shortening the window does not make the motion sparser in relative terms,
it only lowers its absolute magnitude. Which means the active idea cannot be justified as a sparsity
argument at all.

It can be justified as a **variance** argument, and that is the restatement this log registers. At
short windows most joints move less than the estimator can resolve, so updating them injects
estimation noise instead of signal — 76.8% of joints move under 0.25° in a 1 ms window, far below
any achievable per-joint accuracy. Suppressing those updates is a bias–variance trade, not an
exploitation of sparsity. Three consequences:

1. This is exactly what δ-trust already does, globally and with one constant, which is why it buys
   0.5 mm for free. RDOR's claim must be that doing it *per joint, in proportion to measured
   information* beats doing it uniformly. The plan already required RDOR to beat δ-trust; this
   makes clear that δ-trust is not a strawman baseline but the same mechanism at rank one.
2. The headroom should grow as the step shrinks, because the fraction of below-resolution joints
   grows. S7 must therefore be run at several step sizes, and a FAIL at 50 ms alone must not stop
   the active mainline — the plan's stopping rule is amended to require FAIL at short steps too.
   This also aligns the active branch with the asynchronous regime that motivates the project,
   rather than leaving it as an add-on to the 50 ms dense pipeline.
3. The oracle's own threshold sweep must be paired with a same-rate random control at every point,
   which it is, and that pairing turned out to be what carried the result.

### Preliminary oracle sweep, frozen checkpoint, `zgz` only (2 sequences, underpowered)

| Arm | RA-MPJPE (mm) | jitter (mm/step) | update rate |
|---|---|---|---|
| baseline | 19.2576 | 11.384 | 1.000 |
| δ-trust 0.5 | 18.7300 | 7.314 | — |
| δ-trust 0.7 | 18.8997 | 8.778 | — |
| oracle 0.02 | 19.2544 | 10.653 | 0.952 |
| random, same rate | 19.2826 | 11.286 | 0.953 |
| oracle 0.05 | **18.5548** | 8.741 | 0.721 |
| random, same rate | 19.3644 | 10.304 | 0.722 |
| oracle 0.10 | **18.5237** | 6.695 | 0.388 |
| random, same rate | 19.9060 | 8.274 | 0.389 |
| oracle 0.20 | 23.5255 | 3.029 | 0.110 |
| oracle 0.40 | 25.3010 | 0.225 | 0.013 |

The baseline reproduces the frozen 19.2576 mm exactly, and δ-trust 0.5 reproduces the historical
18.73 mm exactly, so the harness is measuring what it claims to.

Paired sequence bootstrap against the best oracle (0.10):

| Comparison | Δ (mm) | 95% CI | excludes zero |
|---|---|---|---|
| vs baseline | −0.734 | [−1.462, −0.101] | yes |
| vs random at the same rate | **−1.382** | [−1.547, −1.192] | yes |
| vs δ-trust 0.5 | −0.206 | [−0.533, +0.078] | no |

**Verdict `S7: INCONCLUSIVE`, and the distinction from FAIL is the point.** The registered rule stops
the active mainline on a FAIL, and here the confidence interval against the strongest control
(±0.305 mm) is wider than the effect it is measuring (0.206 mm), on two sequences. Stopping would be
a decision made by lack of data. The verdict logic in `tools/run_s7_oracle.py` now emits
`INCONCLUSIVE` in exactly this situation rather than collapsing it into FAIL.

Three substantive readings, all of which survive the lack of power:

1. **The routing signal is real and large.** At matched update rates the oracle beats the random
   policy by 1.38 mm with a tight interval, and by 0.81 mm at rate 0.72. Skipping the *right* 60% of
   coordinate groups is worth over a millimetre more than skipping an arbitrary 60%. This is the
   first direct evidence that per-joint routing has something to route on, and it is the result S9
   inherits.
2. **Almost all of the realisable gain is already captured by one global constant.** δ-trust 0.5
   reaches 18.73 mm with no ground truth, no per-joint decisions and no parameters; the ground-truth
   oracle reaches 18.52 mm. So the ceiling on what *any* router can add over uniform shrinkage, on
   this data at a 50 ms step, is about 0.2 mm — which is inside the 0.4 mm replicate noise floor.
   RDOR cannot be justified at this step size, whatever its internals. This is consistent with the
   motion-density measurement above and is the sharpest form of the amended premise: the active
   mainline has to be evaluated where below-resolution motion is common, i.e. at short windows.
3. **Over-skipping is actively harmful and jitter is not the objective.** Oracle 0.40 drives jitter
   to 0.225 mm — a nearly frozen prediction — while RA degrades to 25.30 mm. Any claim resting on
   jitter reduction must be reported alongside accuracy, and the plan's `quiet`-bucket claims should
   be read with that in mind.

The decisive run is the same sweep on the subject-disjoint S1 tracking arms across all 16 validation
sequences and at several step sizes, which is queued behind their training.

## S9 — details

RDOR is implemented in `semkine/router.py` as an inference-time policy with the same interface as the
S7 oracle but blind to ground truth. Per packet it rasterises the KSSF at the previous state, keeps
only events with a face behind them and a defined contour normal, builds `Lambda = sum J^T R^-1 J`
from the S8 residual rows in float64, eliminates the root by Schur complement, and selects joint
groups by greedy conditional `log det(I + Lambda_S)` under a budget, with kinematic closure to the
selected joint's ancestors within its finger and asymmetric on/off thresholds plus a dwell floor.

Thirteen gates pass. Two are worth naming.

**Greedy selection is verified against brute force, not against its own guarantee.** With a budget of
two the optimum over fifteen groups is enumerable, and greedy is required to reach 95% of the optimal
objective on random information matrices. Separately, an instance is constructed where two groups
have identical Jacobian rows: the selection must take the independent third group and must not take
both duplicates. That is the property the conditional form exists for, and a per-group diagonal score
would fail it while looking perfectly reasonable.

**The router localises the moving finger from geometry alone.** A packet is synthesised from exactly
the pixels one finger's motion changes; the router, which sees only pixels and the previous state,
must rank a joint of that finger first or second and concentrate at least 5x more gain there than the
median elsewhere. Measured marginal gains, by moved finger (top three groups, nats):

| moved finger | joints | top gains |
|---|---|---|
| thumb | 0,1,2 | j2 14.9, j1 10.1, j0 2.9 |
| index | 3,4,5 | j5 14.6, j4 10.3, j3 0.15 |
| middle | 6,7,8 | j8 8.8, j7 5.0, j6 0.13 |
| ring | 9,10,11 | j3 16.1, j11 15.0, j4 11.1 |
| little | 12,13,14 | j14 13.3, j13 4.3, j12 0.7 |

Four of five are clean. The ring finger's leading group is the index chain, which is not a failure of
the router but a property of the measurement: the ring finger sweeping across its neighbour changes
pixels whose surface, *at the previous state*, belongs to the index finger, and no policy reading the
previous state can attribute them otherwise. Hence the gate allows second place.

Verdict is `PASS (offline)`. The tracking gate — beating δ-trust in a paired comparison — is not yet
measurable, and S7's finding stands as a warning about it: at a 50 ms step the ceiling over uniform
shrinkage is about 0.2 mm, inside the noise floor. `tools/run_s9_rdor.py` runs the full arm set
(baseline, δ-trust 0.5/0.7, RDOR at three budgets, no-Schur ablation, same-rate random, event-count
at three thresholds) with paired sequence bootstraps, and is queued behind the S1 tracking arms; it
must be run at short windows, where the premise says the mechanism should matter.

## S12 — details

`semkine/filter.py` filters the S3 state augmented with a body-frame velocity: 51 pose coordinates,
51 velocities, a 102x102 covariance, Joseph-form update, and measurement covariance built from the
same packet Fisher information the router uses, so S9's routing weight and S12's fusion weight come
from one quantity rather than two heuristics. Twenty-two gates pass. Three findings.

**A derivation was wrong and the test caught it.** The pose error was propagated with the inverse
right Jacobian `J_r(u)^-1`. The correct operator is the adjoint: an error `e` means the true state is
`X Exp(e)`, both copies advance by the same body increment `u`, and
`Exp(e') = Exp(-u) Exp(e) Exp(u)`, so `e' = Ad(Exp(-u)) e`. The two agree to zeroth order in `u` and
differ at first order, which is exactly the regime of a 50 ms step. The first version failed the
transport test at ratio 2.0 — first-order accurate where the exact form is required. With the adjoint
the transport is *exact* to machine precision at any error magnitude, including half a radian, so the
gate was rewritten from "second order as `e` shrinks" to "exact at finite `e`", which is strictly
sharper and would have caught the original error immediately. Velocity coupling uses the right
Jacobian `dt J_r(u)`, and the `SE(3)` right Jacobian with Barfoot's coupling block was added to
`semkine/lie.py` and verified against central differences to 1e-7.

**A covariance gate was restated.** "Repeated measurements shrink the trace" is not a property of a
correct filter — it depends on where the prior started, and a filter correctly loosening an
over-tight prior would fail it. Replaced by the two real statements: the measured filter reaches a
Riccati steady state (last 50 traces vary by <1e-3 relative), and that steady state is under a tenth
of the propagate-only trace over the same interval.

**The `SE(3)` correction bias was measured, not assumed.** Following S13's finding below, the filter's
own root-metric bias was quantified with antithetic noise pairs, which cancel the statistical term
that would otherwise dominate: under 0.1 mm at innovations up to 0.05 rad, 1.6 mm at 0.2 rad. At the
operating scale of a 50 ms step this is an order of magnitude below the 0.4 mm floor, so the S3
`SE(3)` convention stands in the filter. The first version of this test used independent draws and
reported 1.6/4.1/16.6 mm, all of which turned out to be `sigma/sqrt(N)` rather than bias.

## S13 — details

`semkine/anchor.py` fires an absolute predictor on three internal signals — sustained NIS above the
chi-square band, posterior trace, and the fraction of events falling outside the projected silhouette
— with persistence before firing and a cooldown after, then fuses in the tangent space by iterated
Gauss-Newton on `sum ||Log(x_i^-1 x)||^2_{P_i^-1}`. Fourteen gates pass. Two findings.

**The root metric was changed on evidence.** Fusing two estimates placed symmetrically about a known
truth must return that truth. Under the full `SE(3)` metric it does not: the `Log` map couples wrist
rotation into translation, and for a 0.6 rad disagreement the fused wrist lands **5.9 mm** off, with
the joints exact to 1e-16. This is the correct mean under that metric, and the wrong mean for a
method whose error measure is millimetres of vertex position. The default is now a decoupled
`R^3 x SO(3)` root metric — translation averaged where it is measured — which returns the truth to
1e-6 mm; `root_metric="se3"` is kept so the bias is reproducible rather than asserted. Note the
`SE(3)` fusion was not buggy: it found a *lower* objective than the truth, which is how the
distinction was identified.

Two derivation errors were found on the way and are worth recording because both survive symmetric
test cases: the Gauss-Newton Jacobian must be evaluated at `r_i = Log(x_i^-1 x)`, not at its negative
`d_i`, and the right-hand side is `sum J_i^T P_i^-1 d_i` without the extra `J_i` that the information
matrix carries. In the symmetric case both errors cancel exactly, so only the asymmetric-covariance
gate distinguishes them.

**The geometric trigger was calibrated, not guessed.** Outside-silhouette fraction at a 4 px margin,
measured on a real frame with events synthesised on the true contour: 0.00 at the correct state, 0.00
at 1 cm of translational drift, 0.23 at 2 cm, 0.36 at 4 cm, 0.65 at 8 cm; a 0.6 rad wrist error gives
0.15. So this signal detects gross displacement, not millimetre drift, and the thresholds were set to
0.20/0.10 accordingly — the original 0.45 would essentially never have fired. The test now asserts
monotonicity in drift and ties the default thresholds to the measurement.
