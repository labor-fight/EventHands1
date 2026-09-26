# S37 architecture, failure evidence, geometry, visibility and evaluation audit

Date: 2026-09-25. Auditor: real delegated agent `/root/audit_s37`; six requested review perspectives (1, 2, 5, 6, 18, 20), **not six independently instantiated experts**. Scope: current worktree, retained artifacts and bounded CPU witnesses. No GPU, optimizer step, model training, checkpoint selection, legacy-record editing or new accuracy experiment was performed by this agent.

Evidence labels: **observed** = inspected code/artifact or reproduced bounded witness; **reported** = historical document assertion not re-executed; **inferred** = reasoning conditional on cited facts; **proposed** = unexecuted experiment/design. Historical documents can report valid old experiments without proving a broader causal conclusion. No external literature search was performed by this subagent; novelty and publication claims belong to the separate literature audit.

Instructions read: repository `AGENTS.md`; user-specified `/data1/lyq/code/skill/creative-thinking-for-research/SKILL.md` and `/data1/lyq/code/skill/research-opportunity-graph-skill/SKILL.md`; their relevant generative-lenses, evaluation/output, scoring and research-playbook references, Event4D pitfalls ledger and worked-case/graph excerpts. Those examples informed protocol discipline only; stereo assumptions and benchmark conclusions were not imported into the hand task.

## 1. Decision-changing facts

1. **Observed:** S37 already consumes raw packed events without LNES in its main path, but is a complete-packet GNN, not an incremental runtime. `semkine/event_gnn.py:23-25` explicitly disclaims async execution; `:202-233` rebuilds features, graph and pooling. Sparse input migration is real; complete asynchronous migration is unproven.
2. **Observed:** Current token/selection semantics defeat an append-only prefix cache. `semkine/encoder.py:90-98,147-169` uses packet-wide observed span as the missing-SAE fallback; `semkine/event_gnn.py:127-145` selects nodes using final packet count. Appending an event can change old tokens and old selected node identities even at fixed declared packet duration. CPU witness below confirms both.
3. **Observed:** Historical inference inputs include annotation-derived shape and annotation-derived initialization. `semkine/eval_track.py:177,193` reads `aux['betas']` and initializes every valid run with `pos51[a] + noise`; preprocessing obtains beta from annotation in `tools/prepare_hand_data.py:278-304,323-325`. This is a conditional tracking benchmark, not evidence of a complete event-only initializer/recovery system.
4. **Observed:** Validation, selection and reported test all use `zgz_global` and `zgz_local`: `semkine/dataset.py:475-492`, actual `data/hand_data51/splits_semkine.json`, `tools/select_checkpoint.py:75-98,112-120`. Subject leakage from training is checked, but repeated held-out selection bias remains. A new sealed final evaluation cannot be manufactured by renaming these two already inspected sequences.
5. **Observed:** The target time is `end` ms while observations include bin `end`, i.e. events up to strictly before `end+1` ms: `semkine/eval_track.py:61-77,242`, `semkine/dataset.py:335-339,364-366`. `docs/FAILURE_AND_CLEANUP_LEDGER.md:592-593` already acknowledges the difference. Preserve historical scores as historical; strict-causal experiments need an explicit protocol version and common re-evaluation of comparators.
6. **Observed:** The existing latency number cannot test a 7 ms end-to-end deadline. GPU staging is outside timing, legacy S37 timing uses each run's initial GT pose for every packet, output is 51D parameters rather than the final mesh, and aggregation is minimum of pass means scaled to an anchor. See `tools/make_s36_row.py:128-169,327-353`. The historical 50 ms non-overlapping collection interval also cannot deliver every event's mesh response within 7 ms if collection wait is charged.
7. **Observed/inferred:** The S37 loss defaults to `mse_51d` because its config has no `LOSS.TYPE` (`configs/semkine/s37_routed_s3407.yaml:93-98`; `model/model.py:67`). The CNN SO3/FK family uses `so3_trans_fk` (`configs/eventhands_track_render51_sem_rep2.yaml:54-68`). The archive must be checked for a same-budget S37 loss comparison before attributing the entire CNN–GNN gap to architecture. This audit found no such comparison in the S37 preregistration; that is a scoped search gap, not proof none exists.

These facts block a **complete-system success claim**, not read-only research or bounded Debug. They do not license deleting historical records, silently altering the baseline, or running full training to discover correctness bugs.

## 2. S37 actual dataflow and interface

**Observed:** Config `configs/semkine/s37_routed_s3407.yaml` specifies raw packed input, 30–300 ms training windows, 7-channel events, at most 2048 uniformly selected nodes, predecessor candidate window 32, K=8, hidden 128, three EdgeConv layers, global feature width 512, no previous-state rendering, and routed active heads. Training is mixed-noise teacher forcing, bf16, effective batch 1024 and 6000 updates. Retained metadata for both seeds lists 9 training subjects and 72 sequences. Exact source/checkpoint hashing is maintained by the parent audit.

Data path:

`raw x,y,t,p -> event_tokens(all events) -> stride selection -> state-independent causal event graph -> 3 EdgeConv layers -> [global mean,max] and per-event features -> previous MANO geometry -> front-nearest vertex / LBS routing -> 16 joint evidence rows -> root head + 15 local heads -> prev_mlp addition -> delta + prev -> 51D MANO parameters`.

Code locators:

- Packed event meanings and microsecond timestamps: `semkine/events.py:11-22,119-142`.
- Features are x/W, y/H, signed polarity, packet-normalized time, log inter-event gap and same/opposite-polarity ages: `semkine/encoder.py:11-19,153-170`.
- Graph predecessors and edge geometry: `semkine/event_gnn.py:148-176`. Index causality assumes sorted input; it does not validate timestamp order there. Equal timestamp ties are storage order. Time is a temporal coordinate, not physical depth.
- Pooling and hidden node features: `semkine/event_gnn.py:208-243`.
- Root reads global feature and all 16 evidence rows; finger head k reads row k+1 and its previous three angles: `model/model.py:1193-1228`.
- Routing geometry is detached intentionally: `model/model.py:1380-1388,1403-1411`.
- Independent previous-state MLP remains active: `model/model.py:946-954,1598-1610`. Therefore zeroing routed evidence does **not** remove all previous-state influence, and the graph being state-independent does **not** make the full model state-independent.
- Empty-packet delta gate occurs at `model/model.py:1600-1610`. The EGM branch explicitly keeps FP32 recurrent state; legacy S37 casts prev to output dtype. Existing routed empty-packet test (`tests/test_s37_routed_readout.py:189` onward) is FP32. **Observed CPU counterexample:** bf16 autocast changes empty-packet prev through rounding; see §7. GPU bf16 and real checkpoint coverage remain required.

### MANO quantities and units

**Observed:** State is `[translation_m (3), global axis-angle_rad (3), local axis-angle residuals_rad (45)]`. `model/mano_layer.py:120-142` builds 16 internal FK joints and chain transforms; `:151-169` skins 778 vertices, appends five fingertip vertices to the internal joints, reorders to 21 OpenPose joints, then adds camera translation. FK joints and mesh vertices are distinct objects. The route uses the 778×16 LBS weights; a weight is a kinematic deformation responsibility, not a measured probability that a visible event originated from that joint.

**Observed:** `model/model.py:1371-1378` projects with scaled calibrated intrinsics and returns both uv and z. z is clamped positive, so invalid behind-camera states need an explicit Debug condition: clamping is numerical stabilization, not a visibility proof. Intrinsics scaling is in `model/model.py:1114` onward. Geometry supplied by prev remains a model prior, even when used in a ray intersection.

### local/global and historical metric semantics

**Observed:** `tools/make_s36_row.py:55,83-90` partitions sequences by `_local` in the sequence name. These are motion categories, not local/global coordinate systems. `semkine/eval_track.py:256-274` decodes predictions and GT with the same GT sequence beta; MPJPE is mean Euclidean error across 21 joints after subtracting joint 0, multiplied by 1000 from metres to millimetres. Translation is removed; rotation and scale are not fitted. No Procrustes alignment occurs in this metric.

**Observed, important historical convention:** `root_align(x)` subtracts `x[..., :1, :]` (`semkine/eval_track.py:127-128`; identical legacy function `model/eval_track.py:83-84`). For MPVPE it therefore subtracts **vertex 0**, not the MANO wrist joint. Do not silently replace it and compare resulting values against old MPVPE values.

**Observed artifact:** `outputs/hand_data51/track_render51_dr_sem_rep2/eval_step1000/track_metrics_step50.json` records one checkpoint with local `15.104646682739258`, global `10.028024673461914`, overall `12.387967877774626`. Thus the remembered rounded local 15.1 is real but does not meet strict `<15.1`; 10.66 is the new goal threshold, not that file's matching global value. The historical file has 41 local valid runs and one global run, so repeated GT restarts materially differ across categories. This is artifact identification, not a new experiment or a strict best-model claim.

## 3. What the retained failure evidence can and cannot establish

### Source/artifact ledger

| ID | Evidence and locator | Status and boundary |
|---|---|---|
| A01 | `docs/S37_ROUTED_READOUT_PREREG.md` §§5–7; `outputs/semkine/probe_s37_route.json` | Observed artifact contains selected checkpoints, closed-loop, TF, evidence-zero and oracle-routing records. Historical mechanism interpretation is reported; no re-execution here. |
| A02 | `docs/FAILURE_AND_CLEANUP_LEDGER.md:502-539` | Reported FK/mesh alternatives, unstable root readout and deterministic-accumulation fixes. Distinct architectures, not a controlled visibility-only experiment. |
| A03 | `docs/FAILURE_AND_CLEANUP_LEDGER.md:541-564`; `docs/S38_MESH3D_PREREG.md` | Reported state-geometry injection and root-pooling variants; artifacts deleted, snapshot provenance retained. Do not treat current branches as those old trained systems. |
| A04 | `docs/S37_XYZ_EVENT_DEPTH_ANALYSIS_20260924.md:59-98`; `.experiments/xyz_graph_depth_20260924/s37_s340{7,8}.json` | Actual retained JSON inspected, especially A2 intervention and per-sequence keys. Diagnostic protocol is conditional tracking with GT-based offline analyses; not independent depth measurement or deployed improvement. |
| A05 | `docs/FAILURE_AND_CLEANUP_LEDGER.md:567-605`; `docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md` | Event-guided mesh implementation and no-observation propagation issues, then memory integration work. Engineering/debug evidence is not accuracy evidence. |
| A06 | `docs/S38_GEOROOT_PREREG.md:18-31,52-64,102-120`; `.experiments/s38_georoot_20260924/` | Result sections remain blank, but **complete G0a/G0b artifacts and terminal write logs were found**. A read-only application of the preregistered gate fails stage 0; details below. Do not rerun it or treat a blank document as missing experiment. |
| A07 | `docs/ASYNC_SPARSE_SOTA_MASTER_VERDICT_20260826.md:157-185` | Historical structured simulated review explicitly disclosed as such. It is not 60 real external experts or a universal theorem about all state-conditioned encoders. |

### Back-surface/cross-finger diagnosis

**Observed code:** `semkine/routed_readout.py:36-96` considers K nearest projected vertices and chooses minimum z among candidates whose distance is within 3 px of the nearest. It has no triangle surface coverage test. A front finger 2 px away may steal the event from the nearest surface. Fixed LBS then mixes adjacent joint responsibilities. Earlier EdgeConv has already mixed event features before this readout, so correcting the final routing cannot undo every previous cross-layer message.

**Observed retained artifact / reported interpretation:** A04 found nonzero disagreement between this point-based approximation and exact triangle/ray association, especially for local finger overlap. Replacing in-surface route weights by barycentric surface weights at inference gave mixed signs across the two checkpoints/categories. A04 also retained 3D-kNN shortcut and edge-instability diagnostics. These support the existence of geometric approximation error and the danger of current-pose 3D kNN. They do **not** establish which mechanism dominates the final error.

**Inferred:** An oracle-routing intervention changes the distribution of the geometry–event residual seen during training. Its negative result is compatible with either (a) route mismatch not being dominant, or (b) a model trained to exploit its biased route being disrupted by a better route. It cannot prove visibility is irrelevant. Likewise, GT-depth labeling of event edges is an offline diagnostic, never legal inference evidence.

**Proposed minimum factorial diagnosis:** Freeze input packets, predicted prev trajectory and S37 weights; distinguish (i) point-route vs triangle visibility at readout only, (ii) event message graph held fixed vs selective edge ablation, (iii) fixed prev vs an independently perturbed predicted prev, and (iv) pooled evidence before/after routing. A nuisance-matched edge control must remove the same number of edges with the same temporal/pixel-length distribution. A geometric control must retain the same node/evidence counts. Record output differences and later closed-loop effects separately. Use training-subject development clips for decisions; GT can score association errors offline, never enter forward state. Only a consistent readout or message effect justifies the respective minimal modification.

### Root, prior and observability diagnosis

**Reported, supported by retained diagnostic paths:** `docs/S38_GEOROOT_PREREG.md:18-28` describes weak event evidence for out-of-plane rotation/depth and stronger response from prev MLP. A04 shows depth inherited from predicted prev tracks predicted root-depth error and 3D-neighbor selection can create finger shortcuts. The tiny apparent depth motion of those particular 50 ms windows limits those measurements; it is not a universal proof that a long monocular event trajectory contains no useful depth information.

**Inferred:** Absolute depth can be weakly identified only through assumptions such as calibrated intrinsics, metric shape/scale prior, motion and temporal consistency. Rotational articulation has additional ambiguities. Any new uncertainty score must be calibrated against actual held-out residuals; event count or local LBS mass alone is not uncertainty. Zero events imply absence of new visual information, not proof that the hand stayed still.

**Counterexample to an overly strong old law:** State-conditioned encoding can have a controlled derivative if designed with appropriate contraction or measurement models. A few failed repository variants and empirical average gain estimates do not establish a universal lower bound on closed-loop gain. Retain the useful engineering default (cacheable state-free evidence) without presenting that default as a theorem.

### Latest S38 geometric-root stage 0: completed files, unfilled write-up

**Observed, no rerun:** `g0a_zgz_global.json` has 5544 samples / 1386 packets and `g0a_zgz_local.json` has 4816 / 1204, both GN=8 with the preregistered solver parameters. Their logs end with full sample counts and `wrote ...json`. `g0b_s3407.json` and `g0b_s3408.json` each contain both full sequences and pooled diagnostics; their logs also end with successful JSON writes. Read-only `ps` inspection found no matching Python `probe_s38_stage0.py` process at audit time. A shell exit code is not retained in these logs; the supported statement is that complete outputs are present and no matching task remains live.

The existing gate (`docs/S38_GEOROOT_PREREG.md:63-64`) requires gain≥0.3 and held-out R²≥0.3 on **both** sequences plus pooled cosine≥0.5 for **both** seeds, then admits stage 1 only if rot_x, rot_y or t_z qualifies. Applying it unchanged to retained JSON:

- rot_x: local R² `0.16849920495049397`; pooled cosines `0.14098838305658717`, `0.3234875324970634`. Fails.
- rot_y: local R² `0.20224616004578944`; pooled cosines `-0.13346511680583653`, `-0.07993932192847115`. Fails.
- t_z: global/local R² `0.263102594664263`, `0.07973536963548156`; pooled cosines exceed 0.5 but information gate fails.
- Among remaining degrees, t_y is admitted by the per-degree rule; t_x misses seed-3408 cosine (`0.4737375049957857`), and rot_z misses G0a/G0b.

**Inferred gate decision from observed files:** stage 0 does not authorize stage 1 because none of the three specified out-of-plane/depth degrees is admitted. This is a diagnosis of that particular residual/solver under its old protocol, not proof monocular event geometry is generally useless. Detailed read-only judgment is saved in `.research/s37_async_20260925/audit/georoot_existing_gate_audit.json`. The historical preregistration was not edited. Do not repeat this stage merely because §7/§8 were never filled; any successor must change a justified assumption and receive a fresh, non-zgz-tuned preregistration.

## 4. Six review perspectives and adversarial checks

| Requested perspective | Evidence-based concern | Strongest counterexample to an easy conclusion | Minimum discriminating experiment |
|---|---|---|---|
| ① S37 architecture | State-free event graph is complete-packet; previous state enters readout and MLP (`event_gnn.py:202-243`; `model.py:1569-1610`). | A causal edge list alone is not an asynchronous implementation; old tokens/selection change when future events append. | CPU prefix witness already executed; next implement one synchronous full-prefix reference and compare all output prefixes to a cache before training. |
| ② Historical failure evidence | A01–A07 describe heterogeneous losses, geometries, protocols and some deleted artifacts. | Good TF and bad closed-loop do not prove a unique feedback mechanism; oracle replacement also creates train/test distribution shift. | Immutable same-prediction evaluator parity, then one-variable intervention with matched nuisance control and both seeds; distinguish engineering pass from scientific pass. |
| ⑤ MANO kinematics | 16 internal joints, 21 evaluation joints, 778 skin vertices; global changes move hidden vertices (`mano_layer.py:120-169`). | Freezing every unseen vertex in camera coordinates while rotating the wrist contradicts a single valid MANO state; unchanged hidden latent evidence need not imply unchanged hidden coordinates. | Known root rotation and one finger articulation: verify wrist pivot, all vertex units, hierarchy response, Jacobian/finite differences and no-observation latent-state behavior separately. |
| ⑥ Visibility/association | Point front test is approximate and message mixing precedes it (`routed_readout.py:36-96`; A04). | Switching from projected uv,z to xyz/rays need not add information; xyz kNN may connect touching fingers. | Same predicted geometry and events: exact first-surface association only at readout; matched graph-edge suppression separate; no dense event raster required. |
| ⑱ Local/global and uncertainty | Root reads global+all joint evidence; each finger reads local evidence+prior; GT beta assists scale (`model.py:1193-1228`; `eval_track.py:177`). | Low root-relative error can coexist with large camera translation error; few events can mean either static hand or invisible moving hand. | Controlled illumination/event-rate/occlusion interventions, unperturbed-motion matched pairs, uncertainty-vs-error calibration and recovery tests from non-GT states. Keep absolute diagnostics in audit, not redefine goal MPJPE. |
| ⑳ Fair evaluation / adversarial novelty | zgz selected repeatedly; historical timing is scaled GPU mean; memory/async code presence is not performance (`dataset.py:475-477`; `make_s36_row.py:327-353`). | A two-seed average passing thresholds can hide failure of one checkpoint/category; P95 does not establish maximum latency; an efficient implementation alone may duplicate AEGNN/EventNet. | Single checkpoint must pass both raw thresholds; repetitions remain separate evidence. Predeclare untouched evaluation units, fixed train budget, load envelope and event-to-mesh timing with P50/P95/P99/max/exceedance. Search nearest work before novelty claim. |

## 5. At most three candidates, with a minimum repair control

These are **proposed**. No training decision should be inferred from this table; primary and fallback selection belongs to the parent synthesis after literature and resource evidence.

| Candidate | Core hypothesis / minimum change | Falsifier and current gate |
|---|---|---|
| C0: minimal S37 repair/control | Preserve encoder, heads and MANO interface; establish strict-causal endpoint, valid FP32 recurrence, fixed inputs and independent development selection. After artifact audit, compare existing SO3/FK objective to MSE as a single recipe change if not already fairly tested. | If protocol or loss change accounts for the apparent gain, do not credit a new async/geometry mechanism. Gate: mandatory enabling control; no promised accuracy gain, no architecture novelty claim. |
| C1: causal cached evidence, unchanged geometric readout | Arrival-final event features and predecessor graph make each event feature computable once; preserve S37 hidden widths, EdgeConv and active heads. Read last N cached event features, re-route against current predicted geometry at each mesh-update query. | Fail if exact full-prefix/cache agreement breaks, queue grows under declared load, complete worst measured latency exceeds 7 ms, or geometry refresh dominates cost. Scientific claim additionally fails if timestamp-destruction control retains the purported temporal improvement. Gate: best engineering-first direction, accuracy and novelty unproven. |
| C2: visibility-aware association correction | Only if factorial diagnosis demonstrates visibility-specific output/trajectory harm: replace front-point route by a bounded sparse first-surface association or uncertainty-aware ambiguous responsibility, keeping state-free evidence. | Fail if improvement disappears under equal node counts or train/test-matched routing, or extra surface work violates latency. Historical A04 mixed-sign intervention presently leaves this **hold**. Activate only on predeclared diagnostic evidence; do not also replace event graph by xyz kNN. |

Four independent generative lenses were used: (a) anomaly—TF/recursive disagreement; (b) cross-field mechanism—immutable causal DAG dynamic programming, mapping nodes/features/predecessors rather than a metaphor; (c) inversion—more geometrically exact routing can hurt a biased trained readout; (d) experiment first—same-prefix equivalence can invalidate the async claim before costly training. The user restricts the candidate count, so broad brainstorming was clustered into these three rather than inventing twenty model variants.

## 6. Precise cache semantics for C1

### What is cached

**Proposed definition:** Graph is the complete causal prefix, not an independently recomputed sliding-window graph. For an accepted event i, choose N_i from the W immediately preceding accepted events using fixed-scale spatial/time distance, without future-dependent sampling. With arrival-final token phi_i,

`h_i^0 = ReLU(embed(phi_i))`

`h_i^l = h_i^(l-1) + mean_{j in N_i} ReLU(W_l [h_j^(l-1) - h_i^(l-1); dp_ji])`, `l=1..L`.

Only h_i at each layer is new; old features never change because their inputs and incoming edges are immutable. Keep the most recent W node geometry and features at each required layer to calculate future nodes. A node leaving that working cache does **not** erase its already propagated influence in newer features. Keep last N final-layer features and event coordinates separately for the S37 readout. A full reference evaluates the entire accepted sequence prefix and pools its last N nodes; rebuilding only those N nodes from scratch is a **different model** and is not an exactness reference.

`dp_ji` uses timestamp differences calculated from int64 absolute microseconds before conversion/scaling, not subtraction of two large float32 absolute timestamps. Scaling is fixed before observing the packet; changing it invalidates the reference. Arrival-final features must eliminate current observed-span fallback and final-count uniform sampling. Any fixed clock epoch or timestamp channel must have explicit boundary tests and a time-origin nuisance test. Using a changing query time to redefine every old feature would invalidate the cache and require full recomputation. Existing S37 weights can initialize a compatible shape, but changed feature semantics are not bitwise checkpoint equivalence.

### What cannot safely be cached across geometry updates

At a state/shape/intrinsics update, every current readout event's visibility, nearest surface, LBS responsibility, weighted sum, hard maximum and coverage may change. Therefore recompute routing and per-joint pooling for **all last N readout nodes** at every 3D state update in the first correct version. Mean is subtractable when node weights stay fixed; a removed maximum needs extra bookkeeping, and geometry changes alter even retained weights. Do not silently reuse a stale max, stale route, or stale visibility flag. Full readout recomputation is the minimal trustworthy implementation until a separately proven selective update is justified.

Sequence start/gap, sensor/time-base reset, model-weight change, shape change, intrinsics change and sampling-policy change require explicit state/cache handling. Event evidence cache is unaffected by a pose-only update because it is state-independent, but geometric readout cache is invalidated. Changing beta/K does not change raw event features unless preprocessing/coordinates also change; invalidate only the dependency actually changed, while carrying provenance for both.

### Trigger frequencies and latency

Events are accepted in timestamp/storage order. Local features can be updated per event or in a causal microbatch; the latter must use earlier events inside the same microbatch exactly as the full-prefix reference does. A bounded microbatch timeout limits queue delay. 3D state and full mesh queries need a predeclared maximum spacing small enough that collection+queue+transfer+feature+all-node route+heads+MANO output fits 7 ms over the declared load. The old 50 ms state cadence is retained as an accuracy control, not relabeled a 7 ms system. Exact cadence and load envelope must be chosen from measured resource/throughput evidence, not arbitrary convenience.

Complexity per accepted event is O(W + LKC²) for this MLP implementation, bounded feature working state O(LWC), plus O(NC) readout features. Baseline geometric refresh remains O(NV) vertex-distance work (V=778), followed by O(NJC) weighted pooling and head/mesh costs; caching the encoder does not make this disappear. GPU kernel launch overhead and event transfer can dominate small arithmetic. Batching fifteen finger heads changes accumulation order and already has a reported recursive reproducibility concern (`model/model.py:1203-1207`), so it is a tested optimization, not an assumed identity.

## 7. Bounded CPU witnesses executed

Command:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/audit/check_prefix_contract.py
```

Script/output are confined to `.research/s37_async_20260925/audit/`; PyTorch 2.1.0 CPU, one thread. Wall time about 1.8 s. No checkpoint read, GPU allocation or optimizer.

- Expected: preserve first two tokens after appending a later event at a distinct pixel, if features were arrival-final. Actual: at fixed 10 ms duration both old SAE channels change from approximately 0.1 to 0.4; maximum token change 0.29999995. **Observed prefix invariance fails for current feature definition.** This is a characterization, not a regression introduced here.
- Expected for immutable node IDs: old accepted IDs unchanged after appending. Actual with `max_nodes=4`: count 5 selects `[0,1,2,3]`, count 8 selects `[0,2,4,6]`. **Observed final-count sampling is incompatible with immutable-prefix acceptance.**
- Constructed association witness: at uv=(0,0), nearest vertex depth 1 m and another vertex uv=(2,0), depth 0.5 m with distinct one-hot LBS. Actual route selects the second vertex. **Observed point-level risk**, not proof of real triangle occlusion or accuracy loss.

Additional CPU command, approximately 4 s, no optimizer:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/audit/check_empty_autocast.py
```

It reuses the existing tiny routed test configuration and two empty packets, with FP32 prev containing a translation `0.50123` m. FP32 output is bitwise equal. CPU bf16 autocast returns bf16, is not bitwise equal, and changes the largest state component by `0.001230001449584961` (metres for that translation entry), despite zero event delta. Raw output is `audit/empty_autocast.json`. This establishes a CPU mixed-precision contract failure; it does not measure trained accuracy or certify GPU behavior. The minimal production fix, if chosen by the parent, must preserve FP32 state addition and add a focused regression without overwriting historical checkpoints.

No numeric result from these unit witnesses is a trained model metric; do not produce a new main-row artifact for them or present them as precision/latency achievements.

## 8. Remaining required Debug and decisions

Before formal training: freeze strict time support and target contract; verify real monotonic timestamps/ties and no sample/subject leakage; FP32/bf16 empty and mixed empty batches; units/axis-angle mean/pivot/intrinsics; nonfinite and behind-camera states; gradient flow through trainable evidence/heads and intentional detached association; optimizer update; reset and chunk partition parity; long prefix drift; node acceptance/load truncation; data transfer/queue timing; small-sample fitting without evaluation-set selection; relevant legacy tests. Full-prefix parity should be checked at all outputs with fixed seeds and declared per-tensor tolerances, plus decoded geometry and long-run error. Any approximation must carry its own separate error budget.

Still unresolved: legal event-only initialization/recovery; legal test-time shape prior; untouched final cohort; comparable historical training provenance and loss/budget; a complete latency envelope; stale-cache/graph-update performance; trained robustness at short state intervals; actual temporal benefit beyond count/position statistics; independent nearest-work novelty. Those gaps cannot be closed by the existing oracle probes or code contracts alone.

Recommendation to synthesis: treat C0 as required comparison hygiene, advance C1 only through exactness/performance Debug, retain C2 solely with an explicit diagnostic trigger. Do not claim SOTA or threshold success; the retained S37 main row remains the current arm and old baselines remain untouched.
