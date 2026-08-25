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
## Sparse aggregation

- **An unnormalised exponentially weighted sum reads out the event rate, not the pose.** KEG's
  first readout fed `node_proj` the raw `sum_i exp(-lambda_c a_i) v_i`, whose magnitude is
  proportional to how many events the node holds. Measured on the trained `s20_keg_mixed` s3407
  checkpoint, that block spanned median 1.75 / p99 292 / max 3347 within a single batch while the
  `peak` and `count` blocks stayed inside [0, 9] — three to four orders of magnitude of scale
  attached to a nuisance variable that varies 20x between validation sequences (S0's `zgz_local`
  finding) and another 10x across the 5–50 ms mixed schedule. Symptoms: a non-monotonic checkpoint
  grid (36.4, 68.4, 34.7, 33.5, 43.5, 38.3, 35.0, NaN), a NaN in recursive rollout from finite
  weights, and 33.5 mm against the LNES control's 20.9 at the same step. Remedy: divide by the same
  kernel applied to a constant channel, i.e. an exponentially weighted *mean*, bounded by the token
  embedding and invariant to event count; keep the mass as `log1p`. Gated by
  `test_readout_does_not_scale_with_how_many_events_a_node_holds`. Do not reintroduce a raw sum
  into a linear layer on this data.
- **A distillation arm is not memory-comparable to the arm it distils into.** S19 inherited S20's
  256x4 batch and ran out of memory at 38 GB: its 30–300 ms schedule holds up to six times the
  events per packet and `INPUT_MODE: both` adds a dense LNES for the frozen teacher. Effective batch
  and LR are the quantities the parity gate needs fixed, so 64x16 restores both.
- **Do not edit a shell script while it is running.** Bash reads a script incrementally by byte
  offset; an edit shifts the offsets and the shell resumes mid-token. Cost one confusing
  "unbound variable" at a line whose variable was plainly bound. `run_keg_queue.sh` now honours
  `KEG_REPO` so it can be copied out of the tree before launch.
- **An interrupted run leaves its partial grid in place.** Lightning writes `-v1` duplicates beside
  the old files, so `select_checkpoint.py`'s glob would score a grid whose points come from two
  different trajectories. `run_keg_queue.sh` moves a partial grid aside before starting;
  `tools/dedupe_ckpt_grid.py` repairs one after the fact.
- **State-conditioned routing must be continuous in the state.** KEG assigned each event to the
  single joint with the largest skinning weight at its pixel, read from KSSF at the *previous
  prediction*. In the closed loop that is a discontinuous function of the estimate: near a skinning
  boundary an arbitrarily small state error relocates a whole event to another node (total variation
  2), the per-joint decoders read a neighbour's evidence, and the error feeds the next field query.
  Measured amplification from single-step to recursive at 50 ms: x2.62 against the dense control's
  x1.91, while the single-step errors were 11.59 mm and 10.95 mm — the representation was already at
  parity and the whole 9.5 mm recursive deficit was compounding. Remedy: route over MANO's top-four
  skinning weights as a convex combination, which the field already returns. Near-ties are only
  ~0.3% of surface pixels at a 0.02 margin, so this is not about the static case; it is about drift,
  which is not an epsilon perturbation. Gated by
  `test_routing_is_continuous_in_the_state_that_produced_it`.
- **Recursive error and single-step error are different quantities and want opposite remedies.**
  Reading only the recursive number would have prompted a bigger frontend, which the single-step
  measurement says would have bought 0.64 mm at most. Run `tools/run_closed_loop_probe.py` before
  attributing a tracking deficit to representation.
- **Lipschitz-at-zero is not closed-loop stability.** Softening the LBS routing (convex top-4
  instead of argmax) fixed the epsilon-perturbation discontinuity (TV 2.0 → <0.05) and changed
  nothing at the error scale the loop actually visits: at 1 cm / 0.1 rad the soft assignment's TV
  is 1.46–1.78 out of 2, equal to the hard one's, because the skinning support itself migrates.
  Retrained both seeds: amplification x2.66 / x2.47 against a ≤x2.10 gate, single-step *worse* by
  0.6–1.4 mm (evidence dilution across four joints). Do not buy continuity arguments about
  state-conditioned representations without measuring them at the loop's own error scale.
- **A visibility hard gate deletes the outer half of the contour evidence.** Events fire on moving
  edges; edges straddle the predicted silhouette; `keep = vis` zeroes all 12 geometric channels of
  the outside half and routes it to the background node. Measured at the ground-truth state:
  vis_frac ≈ 0.26 (i.e. ~74% of events deleted), deleted events at median |sdf| 4–9 px, 41–97%
  inside the 12 px band. Under a 1 cm / 0.1 rad state error the deletion grows (7/8 windows,
  low-rate sequences by x2.5–3.5) — a positive feedback path unique to the state-conditioned
  frontend. The dense control never deletes an event. Dynamic confirmation on val_core
  (`closed_loop_sensitivity_50ms.json`): vis_frac 0.62 at GT, 0.54 / 0.43 / 0.32 at 6.6 / 26 /
  50 mm conditioning error, 0.47 at the recursive steady state.
- **An independent-noise fixed point does not predict the closed-loop steady state — for either
  frontend.** Sweeping measured conditioning error against single-step error and solving
  g(e) = e gives 14.3 mm for KEG and 12.0 mm for LNES; the loops actually settle at 32.6 and
  20.9 mm. The excess is the loop's *time-correlated* error, a distribution independent noise
  cannot represent. Signature inside the recursive run: per-step updates collapse to 11% of the
  needed motion (0.069 vs 0.61 rad, both arms), while under teacher-forced noise the same
  networks move 0.77–0.93 of the needed step. Exposure bias, textbook form. Consequence: any
  training-noise schedule tuned on teacher-forced metrics (including S1's and S20b's) optimises
  the wrong distribution; only unrolled/scheduled-sampling training sees the real one.
- **Routing form was measured three ways and is not a variable.** Same-checkpoint hard/soft
  switch: 0.8 mm recursive, 0.12 mm single-step (inside the 1.1 mm replicate floor).
  Independently trained hard routing: 30.4 mm, inside the seed spread. Retrained soft routing:
  FAIL on both seeds. Three negatives, one conclusion: stop touching how the weights are shaped
  and fix what evidence they are allowed to carry.
- **Removing a mechanism's loudest symptom is not removing the mechanism.** The visibility gate
  deletes a third to a half of the contour evidence and deletes *more* as the state error grows,
  which is a real measurement and looked like the cause of the closed loop's steepness. The halo
  lift removes the deletion entirely — deletion-free by construction, bit-identical to the gated
  lift on inside events — and the sensitivity slope at 26 mm conditioning error came back 0.294 and
  0.327 against the gated lift's 0.30. Unchanged. It bought 2.6–3.0 mm of recursive error, so the
  deletion was costing something, just not the thing it was blamed for. What remains is that the
  channels are read at the previous state at all; a deletion-free state-conditioned channel is
  still state-conditioned. Register the gate on the *mechanism quantity* (here the slope), not on
  the symptom, or a partial win reads as a confirmation.
- **An archived negative control looks exactly like a misfiled result.** `s22_keg_halo_unroll`'s
  first attempt collapsed and was deliberately renamed to `archive/s22_constP_fail_s340*` and kept,
  with a `README_s22_constP.txt` beside it stating what it was and — in capitals — not to move it
  back, because `run_keg_queue.sh` skips a config whose run directory already holds a full
  checkpoint grid. On resuming, `training_metadata.json` inside those directories said
  `s22_keg_halo_unroll`, the name said `constP_fail`, and the mismatch read as sloppy filing; the
  grids were renamed into the live path, which both hid a legitimate archive and put a
  "already done, skipping" trap in front of the retraining queue. The live corrected run was
  killed in the same cleanup. Nothing was lost, but ~25 minutes of GPU time went into scoring a
  grid that was already published as a failure. Read the note in the archive before deciding a
  directory is misnamed; provenance lives in the note, not in the metadata, when the point of the
  directory is that the run inside it is *wrong*.
- **A queue whose skip rule is "the grid is full" cannot distinguish done from failed.** That is
  why the archive note exists, and it is a design smell: the skip should key on a
  success marker written at the end of a run, not on file count. Fixed —
  `run_keg_queue.sh` now writes `.run_complete` on a clean exit and refuses to guess when it finds
  a full grid without one.
- **A config key that nothing reads is an arm that does not exist, and it trains to completion
  looking healthy.** `TRACK.UNROLL_PAIR / UNROLL_P / UNROLL_RAMP` were in two configs, documented
  in their headers, referenced by four tests — and read by no code in the working tree:
  `SemKineDataset` had no `unroll_pair` argument, `MNISTModel` had no `unroll_p`. `build_dataset`
  and the model both take TRACK keys through `.get(...)` with defaults, so nothing complained. Two
  runs of ~4 GPU-hours each finished as silent copies of the arm they were supposed to differ from,
  their loss curves indistinguishable from it because they *were* it. The detection was accidental:
  a grep for `unroll` returned only tests, configs and docs. `MODEL` keys had been whitelisted
  against exactly this failure since S18; `TRACK` had not, and the gap was the whole cost.
  Fixed by `MNISTModel.TRACK_KEYS`, which refuses to build on a TRACK key no code consumes.
  The general rule: every config section that can name an experiment needs a whitelist, and the
  cheap check before launching any arm is to run its contract tests, not to read its config.
- **Tests can outlive the code they test.** `tests/test_s22_unroll.py` was a complete, correct
  specification of a feature that was not present; it failed at the first fixture with
  `TypeError: unexpected keyword argument 'unroll_pair'`. Nobody ran it, so a documented,
  configured, tested arm was launched twice as a no-op. Running one test file would have cost
  three seconds against eight GPU-hours. The upside: the spec was precise enough that
  reimplementing against it took one pass and the four tests passed unmodified.
- **`cd X && cmd &` backgrounds the `cd` too.** The `&` binds the whole `&&` list, so a second
  command on the same line runs in the *previous* working directory and writes its log somewhere
  unintended. Two selection jobs were launched this way and only the first landed. Use absolute
  paths in anything backgrounded, and `setsid` if it must outlive the shell — a plain `nohup ... &`
  child was still killed by SIGTERM when its process group went away.
