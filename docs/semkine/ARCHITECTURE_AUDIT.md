# S0 — Architecture audit and baseline freeze

Stage S0 of the SemKine roadmap. No model behaviour was changed in this stage. Every number below
was measured on this machine with the frozen checkpoint; nothing is copied from historical reports.

Verification artefacts:

- `outputs/semkine/s0_baseline_rerun/track_metrics_step50.json` — official recursive evaluation
- `outputs/semkine/s0_baseline_rerun/s0_checks.json` — SHA256 registry, parity, leakage, latency

## 1. Real data flow

```
AEDAT4 (µs, /data1/lyq/data/hand_data/hand_data/<split>/<seq>/event/<seq>_events.aedat4)
  │  tools/prepare_hand_data.py : ts // 1000  → µs DISCARDED, 1 ms binning
  ▼
data/hand_data51/<split>/<seq>_events.npy   (N,3) uint8  [x, y, p]
                          _offsets.npy      (n_ms+1,) int64   ms → event slice
                          .meta             51 float64 per ms + magic (4,13)
                          _aux.npz          betas, camera_K, valid_ms, valid_runs_ms, category
  ▼
HandData51Dataset._build_lnes  → LNES (180,240,2) float32, last-writer-wins
  ▼
MNISTModel.forward(x, prevpos, betas, camera_K)
  ├─ _render_prev(prevpos) → sil + inv  (no_grad, point-splat z-buffer)
  ├─ cat([LNES, render]) → conv1(4→3) → resnet18 → 51D
  ├─ + prev_mlp(prevpos)            (PREVPOS_EMBED, zero-init last layer)
  ├─ ZERO_EVENT_GATE: empty LNES ⇒ delta := 0
  └─ out = delta + prevpos          (PREDICT_DELTA)
  ▼
decode_to_mano_inputs (residual45 + hands_mean) → ManoLayer → 778 verts / 21 joints
  ▼
eval_track.track_sequence : per valid run, prev = GT[start] + noise, then prev ← pred
```

### Exact code locations

| Concern | File | Symbol / lines |
|---|---|---|
| raw event read (memmap) | `model/fastevc.py` | `HandData51Dataset.__init__` L298–306 |
| polarity handling | `model/fastevc.py` | `_build_lnes` L430 `np.clip(ev[:,2],0,1)`; augment L462–473 |
| random window sampling | `model/fastevc.py` | `__getitem__` L436–450 (log-uniform 30–300 ms, clamped to run) |
| sequence reset / run bounds | `model/fastevc.py` | `sample_ends` L330–343 (window history must lie in one run) |
| LNES construction | `model/fastevc.py` | `_build_lnes` L411–432 |
| prev MANO silhouette / inv-depth | `model/model.py` | `_render_chunk` L456–505, `_render_prev` L507–516 |
| `PREV_RENDER` | `model/model.py` | L306, L521–526 |
| `PREDICT_DELTA` | `model/model.py` | L304, L531–536 |
| `ZERO_EVENT_GATE` | `model/model.py` | L307, L533–535 |
| 51D layout | `model/pose_repr.py` | `Slices` L17–44 |
| MANO decode (mean/residual) | `model/pose_repr.py` | `decode_to_mano_inputs` L81–106 |
| betas / camera_K resolution | `model/model.py` | `_resolve_betas_K` L412–437, `_intrinsics` L439–447 |
| train entry | `model/train_abs.py` | `main` L57–204 |
| absolute eval | `model/eval_abs.py` | per-frame, `prev` zeroed |
| recursive eval | `model/eval_track.py` | `track_sequence` L87–185 |
| checkpoint selection | `model/train_abs.py` | L120–141 (fixed grid if `SAVE_EVERY_N_STEPS`, else top-3 by `val_loss`) |

### Shapes, units, coordinate and time semantics

| Quantity | Shape | dtype | Unit / convention |
|---|---|---|---|
| stored event | `(N,3)` | uint8 | `x∈[0,239]`, `y∈[0,179]` (already scaled ×0.375 from 640×480), `p∈{0,1}` |
| event time | — | — | **only the millisecond index**, via `offsets`; sub-ms discarded at preprocessing |
| LNES | `(180,240,2)` | float32 | channel = polarity; value = `(ms_index − window_start)/window ∈ (0,1]`; 0 = no event |
| 51D state | `(51,)` | float32 | `[t(3) m | R_root(3) axis-angle rad | residual45(3×15) axis-angle rad]` |
| local pose | `(45,)` | float32 | **residual w.r.t. `hands_mean`**; `local_full = residual + hands_mean` |
| betas | `(10,)` | float32 | per-sequence GT, read-only |
| camera_K | `(3,3)` | float32 | 640×480 intrinsics; render multiplies by `RENDER_SCALE = 0.375` |
| MANO verts / joints | `(778,3)` / `(21,3)` | float32 | metres, camera frame, OpenPose-21 ordering, wrist = index 0 |
| LBS weights | `(778,16)` | float32 | `ManoLayer.weights`, available for KSSF |
| eval step | — | — | 50 ms = LNES window; drift buckets 5000 ms |

Measured units sanity: `mean|t| = 0.2372 m`, `mean z = 0.5621 m` — metres confirmed.

Polarity convention: the stored polarity is used directly as the LNES channel index. The only
transformations are the two training augmentations (global flip, per-pixel swap), both applied once
inside `__getitem__` after LNES construction.

## 2. Recursive protocol verification (core S0 question)

`eval_track.py` touches the GT array `pos51` at exactly two sites, confirmed by static audit:

```
prev = pos51[a].copy() + sample_init_noise(cfg, rng, noise_scale)   # run start init, once
gts.append(pos51[end])                                             # metric only
```

Inside a run the chain is strictly `prev_t = pred` (L109–110). Each `valid_run` re-initialises
independently; there is no reset inside a run.

**Runtime leakage proof.** Adding 1234.5 to every row of `pos51` after initialisation and re-running
the forward pass with the same `(x, prev)` gives a bit-identical prediction, and the rebuilt LNES is
bitwise identical (LNES never reads `pos51`):

| Check | Result | Gate |
|---|---|---|
| LNES identical after GT corruption | True | must be True |
| `max\|pred − pred_after_corruption\|` | `0.000e+00` | exactly 0 |

## 3. Frozen baseline (measured here, not copied)

Config `configs/eventhands_track_render51.yaml`, checkpoint
`outputs/hand_data51/track_render51/track_render51-step=1000-val_loss=val_loss=0.8015.ckpt`,
step 50 ms, `--init-noise-scale 1.0`, `--seed 0`, split `val`.

```
overall RA-MPJPE  = 19.25760436702419 mm
overall RA-MPVPE  = 15.679647244633856 mm
overall abs MPJPE = 62.214436154752164 mm
overall abs MPVPE = 61.82468463794605 mm
n_frames = 2590
```

This is **bit-identical** to the historical record (19.25760436702419), so the checkpoint, config,
data and evaluator are all the ones the historical numbers were produced with.

| Sequence | category | frames | runs | RA-MPJPE | jitter_all | jitter_static | n_static |
|---|---|---:|---:|---:|---:|---:|---:|
| `zgz_global` | global | 1386 | 1 | 11.6899 | 11.084 | 4.545 | **1** |
| `zgz_local` | local | 1204 | 41 | 27.9692 | 11.683 | `None` | **0** |

`zgz_global` absolute-MPJPE drift, 5 s buckets (mm): 24.5, 35.0, 24.7, 28.2, 30.0, 28.8, 25.7,
39.2, 43.6, 44.1, 44.6, 51.3, 29.0, **28.9 @ ~65 s**.

### Baseline numeric gates

| Gate | Range | Measured | Verdict |
|---|---|---|---|
| recursive RA-MPJPE | [19.06, 19.46] | 19.2576 | PASS |
| recursive RA-MPVPE | [15.48, 15.88] | 15.6796 | PASS |
| static jitter | [4.35, 4.75] | 4.545 | PASS (but see §5) |
| global abs error @ ~65 s | [27.4, 30.4] | 28.9 | PASS |

### Hard correctness gates

| Gate | Result |
|---|---|
| checkpoint loads, no missing/unexpected keys | PASS (`missing=[]`, `unexpected=[]`, 11 209 429 params) |
| same input twice in eval mode, `max abs diff ≤ 1e-6` | PASS (`0.000e+00`, batch 32) |
| sequence/run counts match manifest | PASS (74 sequences; val = 91 runs, 2590 steps @ 50 ms) |
| no cross-sequence state contamination | PASS by construction — `track_sequence` re-inits per run; `set_hand_context` is per sequence |
| GT leakage after init | PASS (exactly 0) |
| 51D order/length/unit and MANO decode match evaluator | PASS (`local_full = residual + hands_mean` verified; 778/21 shapes) |

## 4. Latency (relative reference only)

Same process, batch = 1, 50 warm-up iterations then 500 timed iterations with
`torch.cuda.synchronize()`, GPU 0 (L20):

| p50 | p95 | min | peak allocated |
|---:|---:|---:|---:|
| 3.783 ms | 3.815 ms | 3.731 ms | 292 MiB |

The historical 4.96 ms is **not** treated as a value to reproduce: it came from a different
measurement session with a different alignment convention. All later stages compare against the
3.783 ms p50 measured here, in-process.

## 5. Registered noise floor and selection discipline

These are the numbers every later stage must respect. Source: `docs/experiment_history.md` §12.4,
confirmed by the `n_static` counts above.

| Quantity | Value | Consequence |
|---|---|---|
| same-config replicate scatter | ±0.4 mm | any single-run difference below this is noise |
| cross-training resolution (this budget) | ≈1.1 mm | single-variable retraining claims need >1.1 mm |
| replicate range observed (`ch_inv`) | 2.43 mm | best-of-N minima are not expected performance |
| `track_render51` distribution centre | 20.6–20.7 mm | **19.26 is a best-of-4 minimum**, not the mean |

Cause of the non-determinism across training runs: CUDA `scatter_add` / `scatter_reduce`
accumulation order in the rasterizer. Note this is *training* non-determinism only — inference is
bitwise deterministic on this machine, which is what makes paired inference-time comparisons
(e.g. δ-trust) valid at much smaller thresholds.

**`jitter_static` is removed from all gates from S1 onward.** The measurement above shows it is
supported by 1 step on `zgz_global` and 0 steps on `zgz_local`. It is replaced by the S5 `quiet`
bucket, which is constructed to have real support.

**Checkpoint selection discipline.** `val_loss` is anti-correlated with the closed-loop metric
(experiment_history §8.5: Spearman −0.07 vs closed-loop RA; `val_j3d` gives +0.84). Therefore:
save on a fixed step grid (`TRAIN.SAVE_EVERY_N_STEPS`), score every saved checkpoint with the
recursive evaluator, and select by recursive RA on validation. Never select by `val_loss`.

**Statistical discipline.** Cross-training comparisons require ≥2 replicates, pooled mean ± sem over
all grid checkpoints (not best-of-N), and per-sequence paired bootstrap CIs. Only same-checkpoint
inference-time changes may use small thresholds.

## 6. Registry

SHA256 (first 16 hex) of the frozen artefacts; full digests in `s0_checks.json`.

| Path | SHA256 |
|---|---|
| `outputs/hand_data51/track_render51/track_render51-step=1000-val_loss=val_loss=0.8015.ckpt` | `8f5fa67b6e333d7d…` |
| `configs/eventhands_track_render51.yaml` | `ff0afcc2bf7407db…` |
| `model/model.py` | `11d8d36b31008858…` |
| `model/fastevc.py` | `4cfc975035fa16d8…` |
| `model/eval_track.py` | `d5e1dc3fd8e08f37…` |
| `model/pose_repr.py` | `88f597863b97b734…` |
| `model/mano_layer.py` | `f54872b63b0d9284…` |
| `model/train_abs.py` | `1eb6bacf4d0cb707…` |
| `assets/mano_right.npz` | `20cc6d7a55503176…` |

Data manifest: `data/hand_data51/splits.json` — train 72 trials, val 2 trials (`zgz_global`,
`zgz_local`). Per-sequence run/valid-ms counts are in `s0_checks.json` under `sequences`.

`outputs/hand_data51` in this repo is a symlink to `../EventHands/outputs/hand_data51`, so the
config-relative checkpoint paths resolve to the historical artefacts without copying 13 GB.

## 7. LNES saturation (measured, for S2 motivation)

Per 50 ms window, first valid run, up to 200 windows. Drop fraction = `1 − unique(y,x,p)/n_events`,
i.e. the events LNES's last-writer-wins overwrites.

| sequence | events / 50 ms | LNES drop |
|---|---:|---:|
| `val/zgz_global` | 29 255 | 0.884 |
| `val/zgz_local` | **1 490** | 0.749 |
| `train/ch_global` | 27 185 | 0.874 |
| `train/ch_local` | 4 673 | 0.699 |

The `zgz_local` event rate is ~1/20 of `zgz_global` on the same subject and camera. Combined with
the fact that `zgz_local` supplies 1204 of 2590 val frames and carries 27.97 mm of the 19.26 mm
overall, **the historical overall metric is dominated by one low-event-rate sequence**. This is the
direct justification for the S1 protocol reform.

Caveat carried forward from `experiment_history.md` §10: the saturation statistic is real, but the
inference that "direction information is destroyed before the network" was falsified by the
teacher-forced probe. S2's expected gain from raw events is therefore rated *moderate*, not
*decisive*.

## 8. Verdict

All hard correctness gates and all four baseline numeric gates pass.

`S0: PASS`
