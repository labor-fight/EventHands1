# S27: retention ranks recursive error but cannot be intervened on

**Date:** 2026-08-27
**Status:** hypothesis falsified by intervention. The `GAIN_REG` line is closed on both of its two bullets.
**Protocol:** `val_core` under `_retired_splits_semkine_5v2v3.json` (8 sequences, 10452 frames,
subjects `lr`/`lyq` held out), 50 ms, strict recursion with one ground-truth initialisation per
valid run, checkpoint chosen on the fixed 500-step grid by that same recursive metric.

---

## 1. The reference numbers, re-derived under the current code

| arm | seed 3407 | seed 3408 |
|---|---:|---:|
| `s1_track_domrand` (warm-started dense tracker, no penalty) | **16.6375** | **17.7440** |

These reproduce the `16.64 / 17.74` quoted in the master verdict to four decimals, so the evaluator,
the split and the selection policy used below are the repository's own.

Two facts about that grid are worth carrying forward, because they bound what any claim can mean:
the s3407 grid runs `21.81 / 18.00 / 21.94 / 16.97 / 18.11 / 107.01 / 16.64 / 18.10` with a median
of 18.11, so selection is worth 1.47 mm on its own, and one grid point diverges outright.

## 2. What the retention penalty optimises, and what the loop actually meets

`TRACK.GAIN_REG_W` penalises `E_delta[ ||f(x+delta) - f(x)||_J^2 / ||delta||_J^2 ]` with `delta`
drawn from the isotropic curriculum noise. For a 51-D input that expectation is a Frobenius
quantity, `(1/51) sum_i sigma_i^2`. The recursion applies the same Jacobian to its own error every
step, so the error concentrates on the dominant direction and the amplification it meets is
`sigma_max`. These are not the same number and need not move together.

`tools/tmp_probe_spectral_gain.py` measures three gains on a frozen checkpoint, all at the amplitude
the loop is currently carrying and all in root-aligned joint space: `g_rand` (one isotropic
curriculum draw), `g_self` (the model's own accumulated error direction), and `g_spec` (a power
iterate `u <- A u`, warm-started along the recursion -- not an analogy for the loop but literally
the loop's own update on its error).

Nine frozen checkpoints (`outputs/semkine/spec_gain_{lnes,lowg,keg}.json`):

| ranks recursive RA at | all nine arms | dense family only (n=7) |
|---|---:|---:|
| `g_rand` -- what the penalty optimises | +0.367 (p=0.33) | **-0.357** |
| `g_self` | +0.533 (p=0.14) | +0.036 |
| `g_spec` | **+0.883 (p=0.0016)** | **+0.786 (p=0.036)** |

And over the S24 lambda sweep the penalty is a perfect controller of the wrong number:

| lambda | 0.03 | 0.11 | 0.32 | 3.16 | Spearman(lambda, .) |
|---|---:|---:|---:|---:|---:|
| `g_rand` | 0.219 | 0.188 | 0.148 | 0.085 | **-1.000** |
| `g_spec` | 0.726 | 0.633 | 0.743 | 0.718 | **+0.000** |
| `b` (single-step, mm) | 15.86 | 17.33 | 18.17 | 19.75 | **+1.000** |
| recursive RA (mm) | 21.27 | 19.44 | 20.45 | 21.18 | -0.200 |

The `g_spec / g_rand` ratio grows 3.3 -> 8.4 across the sweep. The mechanism is legible: the
cheapest way to shrink an average over 51 directions is to flatten the fifty that do not matter, so
the optimiser answers the penalty by making the operator *more* anisotropic, leaving the one
direction that governs the loop untouched, and pays for it in bias.

## 3. The intervention, and why it failed

S27 re-points the same penalty along a persistent power iterate
(`TRACK.GAIN_REG_DIR: spectral`), one warm-started iteration per optimiser step advanced from the
displacement the penalty forward already produces, so the cost is unchanged.

Two implementation contracts had to be added, both discovered by running rather than by reasoning:

1. **The iterate must stay out of the metric's null space.** Root translation is *exactly* the null
   space of root-aligned FK -- a 0.05 m shift moves the root-aligned joints by 2e-5 mm, against
   29 mm for root rotation and 22 mm for pose. An isotropic draw is safe by accident, because only
   3 of its 51 components are null and the other 48 dominate the denominator. Power iteration
   maximises the ratio by construction, so the null space is its global optimum: the first launch
   reported retention 30-38 within twenty steps, which is a division by ~zero, not a spectral
   radius. Pinned by `test_power_iterate_never_enters_the_metrics_null_space`.
2. **Splatting a reversed view does not reverse duplicate-write order** (found later, in the S28
   encoder, same class of bug: an operation whose ordering is undefined silently returning the
   wrong quantity).

Results, same protocol:

| arm | direction | lambda | RA mm |
|---|---|---:|---:|
| `s1_track_domrand_s3407` (control) | none | - | **16.638** |
| `s27_ws_spec_0p03_s3407` | spectral | 0.032 | 17.510 |
| `s24_lowg_0p03_bnfix_s3407` (scratch control) | isotropic | 0.032 | 19.823 |
| `s27_spec_0p01_s3407` | spectral | 0.010 | 19.945 |
| `s27_spec_0p03_s3407` | spectral | 0.032 | 23.552 |
| `s27_spec_0p11_s3407` | spectral | 0.105 | 31.161 |
| `s27_barrier_s3407` / `s27_ws_barrier_s3407` | spectral, barrier | - | 90.15 / 81.17 |

Recursive error rises **monotonically** with the strength of the spectral penalty.

## 4. Why, measured rather than argued

Re-probing the trained arms (`outputs/semkine/spec_gain_postfix_*.json`) separates two candidate
explanations and rejects the obvious one:

| arm | `g_rand` | `g_spec` | b (mm) | selected RA |
|---|---:|---:|---:|---:|
| `s1_track_domrand_s3407` | 0.241 | 0.449 | 15.42 | 16.64 |
| `s27_ws_spec_0p03_s3407` | 0.219 | **0.452** | 17.94 | 17.51 |
| `s27_spec_0p01_s3407` | 0.264 | **0.768** | 15.66 | 19.95 |
| `s27_spec_0p11_s3407` | 0.455 | **1.007** | 18.85 | 31.16 |

The penalty **did not reduce `g_spec` at evaluation at all** -- 0.452 against the control's 0.449 --
even though the retention it logged during training fell from 1.02 to 0.13-0.34
(`.cursor/debug-*.log`, `hypothesisId: H1-fix`). The two are different quantities. Training reads
the Jacobian along one persistent direction shared across the batch and advanced once per step; the
probe re-derives the locally dominant direction per sample. A single slowly-moving global vector is
cheap to be insensitive to, and the logged `u . u_prev` of 0.7-0.88 says it had all but stopped
moving. The network nulled the probe vector, not the operator.

So S27 is not evidence that `sigma_max` is the wrong target. It is evidence that **a single global
power iterate is not an estimator of it**, and that the penalty attached to that estimator costs
real accuracy (b rises 15.4 -> 17.9 mm at the mildest setting that moved anything).

## 5. The consequence that redirects the project

From the same probe, on the *selected control* checkpoint:

```
b (single-step, teacher-forced)  15.42 mm
recursive                        17.17 mm      ->  the loop contributes 11%
```

The entire retention programme is competing for 11% of the error. `b` is the other 90%. This is
consistent with, and explains, both null results: S24 moved its own target 2.6x and recursion did
not follow; S27 moved recursion, in the wrong direction, by damaging `b`.

Two bullets were pre-registered for the G line and both have now been fired. The line is closed for
accuracy. What remains is `b`, and there the defect is already measured
(`outputs/semkine/lnes_capacity.json`): at 50 ms, LNES puts 23474 events into 2367 slots and
**discards 73.8% of them by overwriting**, onto a surface that is then 97.3% empty. That is a
capacity defect in the state-independent half of the model -- the one half that, by the
architecture law in 3.5 of the master verdict, can be widened without feeding the loop. S28 tests
it.

## 6. What is kept, and what was reverted

The arm was falsified, so its code is gone rather than left switched off: `TRACK.GAIN_REG_DIR`,
`TRACK.GAIN_REG_FORM`, the `gain_u` power iterate, the observable-subspace mask and the barrier
branch have all been removed, and `configs/semkine/s27_*.yaml` deleted. `TRACK.GAIN_REG_W` is back
to exactly its pre-S27 behaviour. A config key that survives its own refutation is how this project
previously trained two arms that were silent copies of their control.

One compatibility shim remains and is load-bearing: `on_load_checkpoint` drops a `gain_u` key if it
finds one, because the S27 and S28 checkpoint grids were written while the buffer existed and
Lightning loads strictly.

Kept:

* `tools/tmp_probe_spectral_gain.py`: `g_spec` is the best available *ranking* statistic for
  recursive error even though it is not an intervention target, and it is the only probe that
  distinguishes "the penalty worked" from "the penalty's estimator was nulled". It is also what
  measured `b` against `rec`, which is the finding in section 5.
* Claim discipline: `g_spec` ranks (Spearman +0.883) but does **not** predict magnitude --
  `b/(1-g_spec)` over-predicts with a 9.1x spread across arms, against 1.24x for `b/(1-g_rand)`.
  The loop error is not fully aligned with the top singular vector. Neither number may be quoted as
  a predicted steady state.
