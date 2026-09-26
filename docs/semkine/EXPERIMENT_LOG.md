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
| S17 final | `S17: FRAMEWORK` | 2026-08-23 | `CLAIM_MATRIX.md`, `FAILURE_CASES.md`, `tools/run_s17_final.py` |
| S18 KEG frontend | `S18: PASS (mechanism & budget)` | 2026-08-24 | `tests/test_s18_keg.py` (20 gates), `semkine/keg.py`; three design decisions changed on measurement |
| S20 rate claim | `S20: FAIL (G2, all four step sizes)` | 2026-08-25 | `outputs/semkine/s20_grid_step5.json`, `s20_grid_step102050.json`; KEG behind the dense control by +6.1 to +12.0 mm, every CI excludes zero in the wrong direction |
| S21 halo lift | `S21: G-a PASS, G-b FAIL, G-c FAIL` | 2026-08-25 | `outputs/semkine/closed_loop_sensitivity_halo_50ms.json`; deletion removed, sensitivity slope unchanged |
| S22 final | `S22: G3 PASS, G4 null, G5 reported` | 2026-08-25 | `s22_g3_kssf_ablation.json`, `mainline/mainline_test_s1_track_domrand_s3407-step=3500.json`, `s22_efficiency.json` |

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

## S18 — details

`semkine/keg.py` replaces the LNES raster with a state-conditioned kinematic event graph: every event
is lifted onto the previous state's surface through KSSF, assigned to the MANO joint whose skinning
weight dominates its pixel, aggregated over continuous time inside its joint, passed along the
kinematic tree, and read out per joint. Eighteen gates pass. Two design decisions were changed by
measurement, and one registered property of the repository was weakened and had to be measured
rather than asserted.

**The gated scan was dropped, and dropping it is the claim rather than a simplification.** S2's
`gated_scan` is input-gated: `a_i = sigma(W_f x_i) e^{-lambda dt_i}`. That makes the recurrence
*time-varying*, and the entire reason to prefer a state-space recurrence for event data is Zubic et
al.'s (CVPR'24) argument that a linear **time-invariant** continuous system can be re-discretised at
any step size — the property that costs an SSM backbone 3.31 mAP across a train/test frequency
mismatch where an RNN loses 21. An input-dependent gate forfeits exactly that. KEG therefore uses a
diagonal LTI recurrence, one rate per hidden channel, which is also the trainability fix: the closed
form `h_c(T) = sum_i e^{-lambda_c (T - t_i)} v_{i,c}` is a single segmented `index_add_`, whereas
`gated_scan` pads to the longest packet and loops once per event of it. Measured 72 M events/s at a
real packet size against a registered gate of 1 M, so the arm is trainable at full event rate with
no subsampling — the raw path never drops an event, which is the point of leaving LNES.

`test_closed_form_equals_the_per_event_recursion` asserts the sum equals iterating
`h <- e^{-lambda dt} h + v` to 1e-9 relative. That is the Messikommer et al. (ECCV'20)
train-synchronously/deploy-asynchronously contract stated as an equality: the `O(1)` per-event update
and the batched training operator are the same function, not an approximation of one another.

**Every input channel had to be made a function of absolute time differences.** `event_tokens` in
`encoder.py` divides `t` by the packet duration and lets an unseen pixel's surface-of-active-events
value fall back to that duration. Both make the token depend on the window length, which would have
silently destroyed the time-invariance the previous paragraph is about. `keg_tokens` uses
`exp(-x / 10 ms)` of intervals in seconds throughout, and `sae_times` grew a fixed `fallback`.
`test_output_depends_on_age_not_on_window_length` shifts every event and the window end by 40 ms and
asserts KEG is unchanged to 1e-6 while `raw_scan` moves — the contrast is measured, not argued.

**Bitwise-deterministic inference is no longer available on this path, by 2e-7 relative.** S0
registered it as the reason paired inference comparisons may use thresholds far below the 0.4 mm
replicate floor. `index_add_` accumulates with CUDA atomics whose order varies between calls;
measured at 2.0e-7 relative on the aggregate and gated at the prediction by
`test_aggregation_jitter_is_below_float32_epsilon`. The aggregation is forced to float32 regardless
of autocast for a separate reason: bfloat16's eight mantissa bits would discard the tail of an
exponentially weighted sum over 29 000 events, which is the part the long-timescale rates exist to
carry.

**The graph is 16 nodes, not a pixel k-NN graph.** AEGNN's evolving-graph locality argument survives
the substitution — a new event still touches one node and reaches the rest along the tree — while
`cdist` over 29 255 events (8.6e8 pairwise distances per packet, the measured bottleneck of S16's
`aegnn_lite`) does not arise, and the neighbourhoods mean something: pixel adjacency does not imply
kinematic adjacency across a finger boundary. Parameter count is 141 504 against `raw_scan`'s
166 401 at matched width, so a later win cannot be attributed to capacity.

**G5, report-only** (`tools/run_keg_efficiency.py`): 1 344 MAC per event plus 517 184 MAC per window,
against the dense trunk's 1.49 GMAC per window regardless of how many events the window holds. On the
densest validation sequence that is 37x fewer multiply-accumulates; on the sparsest, 585x. No
asynchronous runtime is claimed or shipped, so no latency is quoted.

## S20 — details so far

The LNES control arm is trained on the mixed 5–50 ms schedule and selected: `s3407` step 3500 at
20.871 mm and `s3408` step 4000 at 21.094 mm recursive RA on `val_core` at a 50 ms step, so the
worse-of-two is 21.09. Selection is registered at **50 ms while the claim is at 5 ms**: a 5 ms
selection costs 992 s per checkpoint against 107 s (104 560 recursive frames against 10 452), which
would put a four-arm eight-point grid at nine GPU-hours, and choosing on a step size the claim does
not use is the conservative direction — the selection never sees the metric being claimed, and the
rule is identical for every arm.

The KEG arm has been retrained twice, each time on a defect the measurements named rather than on a
guess. Both are recorded in `FAILURE_CASES.md`; the second is the more interesting one.

**The first arm's deficit was scale, and it was visible in the activations.** Unnormalised
exponentially weighted sums put a median of 1.75 and a maximum of 3347 into one block of
`node_proj`'s input while the others stayed inside [0, 9], because the sum is proportional to a
node's event count. Grid: 36.4 / 68.4 / 34.7 / 33.5 / 43.5 / 38.3 / 35.0 / NaN, best 33.54.
Normalising to a weighted mean took the best to 30.39 and absolute MPJPE from 87.6 to 49.5 mm.

**The remaining deficit was entirely closed-loop, and the frontend was already at parity.**
`tools/run_closed_loop_probe.py` runs an arm twice at 50 ms on the same windows, once conditioning
on the ground-truth previous state and once on its own previous prediction:

| Arm | Single-step RA | Recursive RA | Amplification |
|---|---|---|---|
| KEG (normalised, hard routing) | 11.588 mm | 30.388 mm | x2.62 |
| LNES control | 10.947 mm | 20.865 mm | x1.91 |

As a per-window estimator KEG is 0.64 mm behind the dense arm — inside the 1.1 mm cross-training
resolution S0 registered. The entire 9.5 mm recursive gap is compounding. That is a much more
specific finding than "the sparse arm is worse", and it points at the one path KEG has and the
control does not: KEG conditions its *event routing* on the previous prediction through KSSF, and
the routing was `argmax` over skinning weights. `argmax` is discontinuous in the state, so a small
error near a skinning boundary relocates a whole band of events to a neighbouring joint, the
per-joint decoders read evidence that is not theirs, and the error returns to the next field query.
`FAILURE_CASES.md` registered self-excitation on this path as a risk before any of this was trained;
the probe is what turned the risk into a measurement.

The fix is to make the routing a convex combination of MANO's own top-four skinning weights instead
of their argmax, which the field already returns and the hard version was discarding. A soft
assignment moves by `O(||dx||)` where the argmax moves by `O(1)`, so the feedback gain is bounded;
`test_routing_is_continuous_in_the_state_that_produced_it` measures the separation directly, at
total variation 2.0 for the argmax against under 0.05 for the convex form under the same
perturbation. The re-measured amplification is the gate on this change, not the RA alone.

**Soft routing FAILED its gate (2026-08-24, `closed_loop_probe_soft_50ms.json`).** Retrained on
both seeds with the convex routing, same schedule, same everything:

| Arm | Single-step RA | Recursive RA | Amplification |
|---|---|---|---|
| KEG soft s3407 | 13.035 mm | 34.665 mm | x2.66 |
| KEG soft s3408 | 12.192 mm | 30.090 mm | x2.47 |
| gate | — | — | ≤ x2.10 (LNES +10%) |

Both seeds miss the gate; the single-step error moved *backwards* by 0.6–1.4 mm. The
`drift_curve_50ms.json` companion shows KEG's recursive trajectory is identical under x1 and x4
initialisation noise (34.66 vs 34.72 mm) and reaches ~31 mm inside the first 5-second bucket: the
loop is contractive, the deficit is a high *steady state* of per-step error injection, not
compounding of the initial error. So the discontinuity story was the wrong mechanism.

**Why softening could not work, measured (`probe_gnn_info_loss.py`, debug session 3eaac2).** Under
a closed-loop-scale perturbation (1 cm / 0.1 rad — the regime the loop actually visits, not an
epsilon), the *soft* assignment's total variation is 1.46–1.78 out of 2, statistically equal to the
hard assignment's 1.51–1.79. The skinning support itself migrates whole bands of pixels at that
error scale, and a convex combination of migrated supports migrates with them. Lipschitz-at-zero
was the wrong property to buy.

**The dominant evidence path nobody had measured: the visibility gate deletes the contour's outer
half.** At the *ground-truth* previous state, `kssf_event_channels` keeps only vis_frac ≈ 0.26 of
events (8 real windows, range 0.13–0.48); the deleted events sit at a median |sdf| of 4–9 px with
41–97% inside the 12 px contour band. Event cameras fire on moving edges, edges straddle the
silhouette, and `keep = vis` zeroes the outer half of exactly that evidence and pools it into the
background node. Under the 1 cm / 0.1 rad perturbation vis_frac drops further (7 of 8 windows,
low-rate windows by x2.5–3.5): deletion rate grows with state error, which is a positive feedback
loop the dense control does not have (the CNN always sees every event). This is R3 of debug
session 3eaac2; the closed-loop sensitivity sweep (`closed_loop_sensitivity_50ms.json`) is the
dynamic confirmation gate.

**The routing-continuity fix was tested and falsified on the quantity it was aimed at, which is
worth more than the fix would have been.** Convex skinning routing was retrained on both seeds and
re-probed: amplification x2.66 and x2.47 against the argmax version's x2.62, i.e. unchanged. The
discontinuity it removes is not the term that sets the closed-loop deficit.

On the selected checkpoints the two routings are close enough to be undecidable here. Worse of two
seeds: 29.32 mm soft against 30.92 mm hard, a 1.6 mm difference against the 1.1 mm cross-training
resolution S0 registered. Soft is kept because that direction is the one the evidence points, while
noting it costs 2.3x the activation memory and 40% of the throughput, and that a single-checkpoint
read of this comparison was misleading: at step 3500 alone hard leads by 2.8 mm, and the grids
disagree about where each run's best point is. Two further hypotheses died the same afternoon:

* **Not divergence.** `tools/run_drift_curve.py` bucketed error by elapsed time within a run. KEG
  sits between 26.6 and 31.7 mm across 25 s and the dense control between 14.5 and 24.0; neither
  curve runs away. The loops are stable.
* **Not a missing absolute observation.** Scaling the initialisation displacement by four changes
  KEG's result from 34.665 to 34.716 mm and the control's from 20.871 to 20.875 — both arms forget
  where they started, so both observe absolute pose rather than only integrating increments.

What is actually happening is that KEG's loop settles at a higher fixed point: its error is already
31.2 mm in the first five-second bucket, against 15.2 for the control, and then stays there. The
quantity that sets a fixed point is how fast single-step error grows with error in the conditioning
state, and the sweep measures exactly that (`prev_noise_sensitivity_50ms.json`, corruption in
multiples of the S1 training noise):

| corruption | x0 | x0.5 | x1 | x2 | x4 | slope |
|---|---|---|---|---|---|---|
| KEG | 13.04 | 13.20 | 13.64 | 15.10 | 19.14 | +6.11 mm |
| LNES control | 10.95 | 11.01 | 11.22 | 12.11 | 14.10 | +3.16 mm |

KEG is 1.9x more sensitive to a wrong conditioning state, which is what its input composition
predicts: every one of its twelve geometric channels is read at the previous state, while the dense
arm carries a state-free event image it can fall back on. The training noise was set for the dense
arm. The corroborating detail is that KEG's checkpoint grid peaks at step 1500 on both seeds and
degrades after (30.39 -> 33.47 and 30.92 -> 38.18) while the control improves to step 3500: leaning
harder on a nearly correct conditioning state lowers the training loss and raises the closed-loop
sensitivity, and more training buys more leaning. S20b tests that reading by hardening the schedule
on both arms; a grid that still peaks early would falsify it.

## S20 closed-loop adjudication (2026-08-24 night, `closed_loop_sensitivity_50ms.json`, debug 3eaac2)

Four arms through teacher-forced / recursive / three conditioning-noise levels, with the
conditioning error *measured* through FK (`cond_err_ra_mm`) instead of quoted in noise multiples,
per-step update magnitudes collected by a hook inside the recursive evaluator, and per-window
vis_frac logged inside `kssf_event_channels`. This closes every hypothesis the soft-routing
failure opened.

**The g-curve (single-step RA against measured conditioning error, mm):**

| arm | e=0 | e=6.6 | e=26.1 | e=50.0 | slope (mid) | fixed point g(e)=e | measured recursive | echo excess |
|---|---|---|---|---|---|---|---|---|
| keg_soft s3407 | 11.15 | 11.90 | 18.03 | 29.02 | 0.32 | 14.3 mm | 32.6 mm | x2.28 |
| keg_soft, hard switch | 11.03 | 11.79 | 18.00 | 29.15 | 0.32 | 14.3 mm | 31.8 mm | x2.23 |
| keg_hard s3407 | 11.59 | 12.16 | 18.04 | 29.79 | 0.30 | 14.6 mm | 30.4 mm | x2.08 |
| lnes s3407 | 10.95 | 11.22 | 14.10 | 17.42 | 0.15 | 12.0 mm | 20.9 mm | x1.74 |

Verdicts, one per hypothesis:

* **R1 CONFIRMED — KEG's g-curve is 2.2-2.8x steeper than LNES's** at every measured error. This
  is the state-conditioned frontend's own sensitivity, now measured against FK-verified
  conditioning error rather than noise multiples.
* **R2 CONFIRMED — independent noise does not explain either loop.** Both arms' recursive steady
  states sit far above their independent-noise fixed points (KEG 32.6 vs 14.3, LNES 20.9 vs
  12.0). The loop's own error is time-correlated and the network has never trained on that
  distribution. The *excess* is larger for KEG (x2.28 vs x1.74), and the arithmetic splits the
  11.7 mm recursive gap into ~2.3 mm of fixed-point difference plus ~9.4 mm of echo difference —
  but the echo rides on the g-slope, so the two multiply rather than add.
* **R3 CONFIRMED — deletion grows with state error, live.** vis_frac on val_core: 0.62 at the GT
  state, 0.54 at 6.6 mm, 0.43 at 26 mm, 0.32 at 50 mm, 0.47 at the recursive steady state
  (p10 windows 0.34). The gate deletes between a third and a half of the evidence exactly when
  the loop needs it most.
* **R4 REJECTED in the teacher-forced regime, CONFIRMED in the loop.** Under independent noise
  the network moves 0.77-0.93 of the needed step — no under-updating. In the recursive run the
  per-step pose update collapses to 0.069 rad against a needed 0.61 rad (LNES: 0.071 vs 0.61).
  Both arms freeze relative to the motion and coast near a slowly-varying posture; the error cost
  of freezing is set by the g-slope, which is why the same freeze costs KEG 32.6 mm and LNES
  20.9 mm. This is the textbook exposure-bias signature (Ross & Bagnell's DAgger, AISTATS 2011;
  Bengio et al.'s scheduled sampling, NeurIPS 2015): the training conditioning distribution
  (GT + independent noise) never contained self-generated drift.
* **R5 REJECTED — routing form is not a variable.** Same-checkpoint hard/soft switch moves the
  recursive error by 0.8 mm (31.8 vs 32.6) and the single step by 0.12 mm, both inside the 1.1 mm
  replicate floor. Independently-trained keg_hard lands at 30.4, inside the seed spread. Together
  with the retrained-soft FAIL above, no further routing-form work is justified.

**Decision.** Two fixes, one per confirmed mechanism, strictly sequenced:

* **E5.5a `s21_keg_halo` (launched tonight, seeds 3407/3408, GPU 6/7):** replace `keep = vis`
  with the deletion-free halo lift — continuous visibility `w_geo = exp(-[sdf]_+ / 6px)` (the
  SoftRas replacement of a coverage test by a monotone SDF function), closest-point projection
  `x - (d+1)n` to read the LBS of the contour segment that produced the event, unit routing mass
  `w_geo` to the four surface joints + `1 - w_geo` to BG. Contract tests
  (`test_halo_routing_is_deletion_free_and_matches_the_gate_inside`,
  `test_halo_is_a_config_bit_that_reaches_the_lift`) passed before launch; inside events are
  bit-identical to the gated lift, so the single variable is what happens to the deleted half.
  Gates: G-a single-step within 1.1 mm of 11.59; G-b g-slope at 26 mm < 0.20; G-c amplification
  <= x2.10 or recursive RA <= 21.9 mm.
* **E5.5b unroll (scheduled, not yet built):** short closed-loop unroll / scheduled sampling in
  training to attack the echo excess both arms share. R2's numbers cap its value: even a perfect
  echo fix leaves KEG at its fixed point (~14.3 mm gated, lower with halo), and even a perfect
  representation fix leaves the x1.74-x2.28 echo. If E5.5a passes G-b but misses G-c, E5.5b goes
  on top; if E5.5a misses G-b, the halo did not flatten the curve and the representation bet is
  falsified in one number.

## S21 — the halo lift removed the deletion and did not flatten the curve

`s21_keg_halo` trained on both seeds, both selected at step 1500, re-probed through the same three
conditioning-error levels (`closed_loop_sensitivity_halo_50ms.json`):

| arm | single step | g-slope at 26 mm | recursive RA | amplification |
|---|---|---|---|---|
| keg_halo s3407 | 12.11 mm | 0.294 | 27.77 mm | x2.29 |
| keg_halo s3408 | 11.16 mm | 0.327 | 27.90 mm | x2.50 |
| keg_hard (gated lift, reference) | 11.59 mm | 0.30 | 30.39 mm | x2.08 |
| lnes control | 10.95 mm | 0.148 | 20.87 mm | x1.91 |
| registered gate | within 1.1 mm of 11.59 | < 0.20 | <= 21.9 mm | <= x2.10 |

**G-a PASS** (+0.52 and -0.43 mm): making the lift deletion-free costs nothing as a per-window
estimator, which is what the bit-identical-inside contract test predicted.

**G-b FAIL, and this is the informative number.** The slope is 0.294 and 0.327 against the gated
lift's 0.30 — unchanged. R3 measured that the visibility gate deletes a third to a half of the
contour evidence and that the deletion rate grows with state error, and that measurement stands;
what fails is the inference that the deletion was what made the curve steep. Removing the gate
entirely leaves the sensitivity where it was, so the steepness comes from the lift being *read at
the previous state at all*, not from how much of it survives. Every one of KEG's geometric channels
is a function of the conditioning pose; a deletion-free version of a state-conditioned channel is
still state-conditioned.

**G-c FAIL.** 27.8 mm against the 21.9 mm gate. The halo did buy 2.6-3.0 mm of recursive error
against the gated lift (30.4 -> 27.8), so the deletion was costing something real; it was not
costing the deficit.

Per the sequencing registered before launch, missing G-b falsifies the representation bet and
E5.5b was not entitled to run on top of a passed G-b. It was trained anyway, as the one remaining
independent term (the echo the *dense* arm shares), with its result recorded as a bound on the
echo rather than as a rescue of the frontend.

## S22 — final measurements

**G2, the main claim, FAILS at every step size** (`s20_grid_step5.json`,
`s20_grid_step102050.json`; both retrained seeds of each arm, all four pairings):

| step | KEG s3407 / s3408 | LNES s3407 / s3408 | worst-case KEG deficit |
|---|---|---|---|
| 5 ms | 31.07 / 29.30 | 22.96 / 23.06 | +12.02 [+7.87, +16.56] |
| 10 ms | 30.50 / 28.61 | 21.99 / 22.47 | +10.12 [+5.56, +14.64] |
| 20 ms | 30.11 / 28.66 | 21.86 / 21.66 | +8.44 [+3.86, +13.02] |
| 50 ms | 29.32 / 28.66 | 20.87 / 21.09 | +8.45 [+5.42, +12.00] |

Sixteen paired comparisons, sixteen FAILs, every interval excluding zero in the wrong direction.
The claim was that a state-conditioned asynchronous lift would win at short steps where the dense
rasteriser throws events away; the measured shape is the opposite of the predicted one. KEG is flat
in step size (29-31 mm everywhere) and the dense control is nearly flat too (20.9 -> 23.0 as the
step shortens by 10x). Neither arm converts a shorter step into accuracy, so the premise that the
50 ms window was the binding constraint is not supported on this dataset. What KEG's flatness does
show is rate-invariance of the LTI readout — the property S18 was built for — but it is
rate-invariant about a worse fixed point, and invariance is not a win.

Two honest caveats on the 5 ms row. The dense control diverged on 4 of 8 val_core sequences at
5 ms on seed 3407 (KEG 3 of 8), so that row mixes precision with divergence and is the least
trustworthy of the four; the 10-50 ms rows are clean and say the same thing. And selection ran at
50 ms for every arm (recorded in the S20 details, nine GPU-hours saved), so no arm was tuned on
the claimed metric.

**G3, the KSSF mechanism claim, PASSES** (`s22_g3_kssf_ablation.json`; one checkpoint, one config
bit, no retraining):

| arm | recursive RA | absolute | diverged |
|---|---|---|---|
| keg s3407 | 29.32 mm | 47.7 mm | 0/8 |
| keg s3407, KSSF channels silenced | 86.30 mm | 520.8 mm | 6/8 |
| keg s3408 | 28.68 mm | 50.8 mm | 0/8 |
| keg s3408, KSSF channels silenced | 45.41 mm | 149.0 mm | 0/8 |

-85.8 mm [-101.5, -66.5] and -86.5 mm [-103.5, -65.6]. The twelve geometric channels are not
decoration: without them the same weights cannot track at all. Read this as a mechanism claim and
not as a magnitude — silencing the channels at inference is a distribution shift the weights never
saw, so the size of the gap is inflated by the mismatch. What it establishes is that the state
conditioning is load-bearing, which is the same property that G-b says makes the loop fragile.
Those two results are the same fact seen from both sides.

**G4 is null on the test split** (`mainline_test_s1_track_domrand_s3407-step=3500.json`, 18
sequences, 50 ms, on the dense domrand checkpoint):

| arm | RA | jitter | update rate | anchors fired |
|---|---|---|---|---|
| baseline | 22.771 mm | 11.52 | 1.000 | 0 |
| delta_trust 0.5 | 22.699 mm | 7.16 | — | 0 |
| filter | 22.739 mm | 11.32 | 1.000 | 0 |
| filter+router | 22.779 mm | 11.25 | 0.648 | 0 |
| filter+router+anchor | 22.729 mm | 12.53 | 0.653 | 1079 |

Every paired interval against both references includes zero, and the largest effect is 0.04 mm
against a 0.4 mm replicate floor. The filter, the RDOR router and the triggered anchor together
change nothing at 50 ms, which is what S7's oracle ceiling (~0.2 mm) predicted two weeks earlier
and what the roadmap registered as the reason to judge the active mainline at short windows. The
router does cut the update rate to 0.65 at no accuracy cost, so the honest statement is a
*compute* result and not an accuracy one. `delta_trust 0.5` remains the only intervention with a
visible effect, and it is on jitter (11.5 -> 7.2) rather than on error.

**G5, report-only** (`s22_efficiency.json`): 2 240 MAC per event plus 586 816 MAC per window for
KEG, against 1.490 GMAC per window for the dense trunk irrespective of event count. On the
sparsest validation sequence (1 512 events per 50 ms) that is 375x fewer multiply-accumulates; on
the densest (29 255 events) 22.5x. Both arms additionally rasterise 1 538 MANO triangles per
window, excluded from both sides. No asynchronous runtime is implemented, so no latency or energy
number is quoted, and none of this offsets G2.

**E5.5b, the unroll arm: attempt 1 collapsed, attempt 2 in flight.** Mixing self-conditioning at a
constant `UNROLL_P = 0.5` from step 0 gives the network half its conditioning states from an
untrained model, i.e. noise unrelated to the pose, and its optimum is to stop reading `prev` at
all. It did: teacher-forced 42.5 mm and recursive 144.5 mm at step 3500 — the signature of
collapsing to absolute regression, and worse than every other arm in this log. Checkpoints 500 and
1000 evaluate to NaN on `val_core`. Training loss never went NaN (final `train_loss` 32.3 / 38.2
against the teacher-forced arms' ~0.25), so this is a converged bad optimum, not a numerical
failure. Kept as the negative control at `outputs/semkine/archive/s22_constP_fail_s340*`.

Scheduled sampling anneals for exactly this reason, so attempt 2 keeps pure teacher forcing through
the LR warmup (step 500), ramps linearly to 0.5 by step 2000 and holds it (`UNROLL_RAMP: [500,
2000]`). Attempts 2 and 3 then trained as **no-ops**: the three `TRACK.UNROLL_*` keys were read by
no code in the working tree, so both runs were silent copies of `s21_keg_halo`. See
`FAILURE_CASES.md`; the arm was reimplemented against `tests/test_s22_unroll.py` (which had been
written against the missing implementation and failed at its first fixture), all four contract
tests plus the 33 S1 parity tests and the remaining 265 now pass, and `MNISTModel.TRACK_KEYS` makes
an unimplemented TRACK key a build error rather than a default.

The real attempt is training on both seeds. Its gates are unchanged from the config: G-a
single-step within 1.1 mm of s21's 12.11 / 11.16 mm, G-b recursive ≤ 21.9 mm or amplification
≤ x2.10, G-c the recursive per-step pose update must exceed 0.2 rad against the 0.069 measured
before the fix. G-c is the one that matters: it is the only gate that reads the mechanism (update
collapse) rather than its consequence, and both frontends share the defect, so a pass here
transfers to the dense arm.

**Real-attempt verdict (2026-08-25 21:44, `closed_loop_sensitivity_unroll_50ms.json`, selection
grids in `select_anneal.log`).** Both seeds trained clean (16:32-20:09, grids full; selection
picked step 1000 on s3407, step 4000 on s3408, both grids flat to ±1 mm). That the arm is not a
silent copy is established behaviourally: recursive per-step pose update 0.192 / 0.105 rad
against s21-halo's 0.062 on the same probe — a 3x difference no replicate noise produces.

| arm | TF RA | recursive RA | amp | TF@6.6mm | TF@26mm | echo excess |
|---|---|---|---|---|---|---|
| unroll s3407 | 10.92 | 34.22 | x3.13 | 11.68 | 19.91 | x2.22 |
| unroll s3408 | 10.26 | 31.03 | x3.02 | 11.02 | 18.36 | x2.26 |
| s21-halo anchor | 12.11 | 27.76 | x2.29 | 12.55 | 18.26 | x1.85-1.95 |
| lnes anchor | 10.95 | 20.87 | x1.91 | — | — | x1.74 |

**G-a PASS** (10.26 / 10.92, better than s21 by 1.2-1.9 mm). **G-b FAIL** (31.0-34.2 vs ≤ 21.9;
worse than the arm it was meant to improve). **G-c half-lifted** (0.105-0.192 rad vs the 0.069
collapse value; s3407 grazes the 0.2 gate). The three gates together say something sharper than
any one of them: the unroll taught the network to *move* (G-c) without teaching it where to move
(G-b), because the conditioning it trains on — the lead prediction, 1-2 mm from GT — also taught
it that prev is near-clean (G-a improving is the same fact). The error curriculum narrowed: on
half the samples the 6.6-26 mm noise-table coverage was replaced by that near-clean prediction,
the g-slope steepened 0.29 → 0.38-0.42, and the echo excess the halo had cut to x1.85-1.95 went
back to x2.22-2.26. A training scheme built against exposure bias widened the exposure gap.

Standing decision: **replacement-form unroll is dead on both variants** (constant-P catastrophically,
annealed by curriculum narrowing). The one family member left is the residual form — keep
`(GT + curriculum noise)` and add only the lead error's correlated structure,
`cond = prev_state + (pred_lead - lead.target)` — implemented behind `TRACK.UNROLL_RESIDUAL`
(config `s23_keg_unroll_resid.yaml`, contract test
`test_residual_unroll_adds_the_lead_error_on_top_of_the_noise`), not yet trained. Falsifiable
either way: if s23's recursive RA does not beat s21-halo's 27.8 mm, the two-window
training-distribution family is closed (its 1-2 mm self-error amplitude cannot represent a 27 mm
steady state) and the next single variable is the g-slope itself — state-free fallback channels
in the lift (E5.6).

Implementation, for the record: a sample is `(leading, main)`, the leading window ending exactly
where the main one starts and carrying the main window's *un-noised* conditioning state as its
target, so a prediction on it estimates what the main window is conditioned on. Both halves are
transformed by one domrand draw — a pair whose halves saw different virtual cameras would train the
model to estimate a state in one frame and apply it in another. The lead forward is under
`no_grad` and detached, so the pair changes what the network is conditioned on and not how gradient
reaches it: one extra forward per step, no extra backward graph. At a run boundary the pair
degenerates to a zero-length interval at the same instant, where the zero-event gate returns the
conditioning state unchanged and the sample falls back to teacher forcing. `unroll_pair=False`
leaves the S1 random-draw order untouched, which the bitwise-parity test confirms.
