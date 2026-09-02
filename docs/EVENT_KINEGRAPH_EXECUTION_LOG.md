# EventKineGraph Execution Log

> Historical execution receipt reconstructed from the verified E0A--E3 ledger after the original
> untracked working-tree copy was no longer present on 2026-09-02. Numeric results, hashes and Gate
> decisions below are copied from `docs/FAILURE_AND_CLEANUP_LEDGER.md` and the retained E0--E3
> receipts. This document does not reopen a failed stage or authorize later stages.

## 1. Scope and immutable boundaries

- Repository: `/data1/lyq/code/mesh/EventHands1`
- Research branch used for E0A--E3: `research/event-kinegraph-20260823`
- E0A--E3 base/remote HEAD: `53039b28fcb0b76af65f43cef163b5fd204869ee`
- Frozen B0 recursive RA: `19.25760436702419 mm`
- B2 selected checkpoints: seed3407/3408, both step3500
- B2 recursive val_core RA: `16.6375492202 / 17.7440119019 mm`
- B2 checkpoint SHA-256:
  - seed3407: `52228792acdd89e1c5bf75ded2615b23c2b91e7a8c93c061bdc7818f87070836`
  - seed3408: `fea201e9ad8a9eeb5f52ee5810d3bb918f7ec6d5811fd6a0baf6cc48654e864c`
- MANO SHA-256: `20cc6d7a555031767a42c5fd3738396acb1c8616a0170686e42ec5f59130c8e5`
- split SHA-256: `fa842f44df6d14a1123bb138ccc3e81cd4b0a4352b051c1dcf48c76aace356dd`
- `data` and `outputs/hand_data51` were external symlink boundaries and were never deletion targets.
- E0A--E3 produced no automatic commit or push.

Evidence labels used throughout:

| State | Meaning |
|---|---|
| `observed` | Produced by an executed command, real dataset, checkpoint or evaluator |
| `code-proven` | Follows directly from the retained implementation and recorded E3 function form |
| `reported` | Historical receipt exists, but the original artifact is absent |
| `inferred` | Consistent causal explanation without a selective mechanism test |
| `pending` | Requires a new explicitly authorized run |

## 2. Stage ledger

| Stage | Result | Retained result | Next-stage decision |
|---|---|---|---|
| E0A | PASS audit | cleanup boundary, evidence ledger and safe cleanup tool changes | E0B only |
| E0B | BLOCKED under original B1 contract | B0/B2 frozen; B1 remains `reported/PENDING` | user later explicitly allowed B2 as primary control |
| E1 | PASS | deterministic two-level ragged micro-packet contract | E2 only |
| E2 | PASS | bounded causal hash graph and one-layer message primitive | E3 only |
| E3 | NO-GO | failure receipt; E1/E2 general primitives retained | no E4A--E8/P |

## 3. E0A/E0B receipt

E0A audited tracked/ignored files, caches, outputs, logs, symlinks and external targets. E0B deleted
only the nine exact generated paths authorized by the audit: `.pytest_cache`, seven exact
`__pycache__` directories and `evcreader/build`. It did not delete source, configs, checkpoints,
metrics, figures, MANO assets, data, outputs or symlink targets.

B0 was rerun on 2,590 frames with the original config/checkpoint, val split, 50 ms window and seed0.
The result remained bit-equivalent apart from the checkpoint locator:

- overall RA-MPJPE: `19.25760436702419 mm`
- RA-MPVPE: `15.679647244633856 mm`
- absolute MPJPE: `62.214436154752164 mm`

B2 seed3407/3408 checkpoints fully loaded and retained their registered hashes and selected
val_core metrics. B1 remained only a historical `12.77 mm` report: the corresponding configs,
checkpoints and hashes were not found. B2 was never renamed to B1.

## 4. E1 deterministic micro-packet contract

E1 introduced two-level ragged ownership without `B x Nmax` padding:

```text
events(sumN,5) -> ptr(P+1) -> packets
packets(P) -> packet_to_interval(P), interval_ptr(I+1) -> supervision intervals
```

The default policy was fixed-count `256` with `dt_max=5 ms`; fixed-time `1/2/5 ms` was also
covered. Empty time bins were represented as legal packets, tail packets were retained, and only
the final packet of each supervision interval was registered as a query packet.

Observed audit:

- 12 sequences, 48 real 50 ms intervals, 691,580 events
- 6 local and 6 global sequences
- missing/duplicate/future/cross-run/cross-sequence events: all `0`
- non-monotonic/polarity/query-registration/packet-boundary errors: all `0`
- bitwise event round-trip mismatch: `0`
- 16 legal empty packets observed
- event-rate range: `876.25--35,318.75 events/interval`
- DataLoader p50 ratio versus legacy raw: `1.109890`, Gate `<=1.25`
- retained tensor payload ratio: `1.113467`, Gate `<=1.5`
- tracemalloc peak ratio: `0.612082`, Gate `<=1.5`

The first object-heavy implementation failed performance (`p50=10.0169x`, peak `1.6149x`). The
hot path was replaced by one preallocation plus owner-index gathering without changing packet
semantics. E1 then passed.

`GATE_VERDICT=PASS`, `REVERT_OR_KEEP=KEEP`, `NEXT_STAGE_AUTHORIZED=YES_E2_ONLY`.

## 5. E2 causal bounded hash graph

Default graph parameters:

```text
spatial cell = 4 x 4 px
time cell    = 1 ms
tau          = 5 ms
radius       = 8 px
K            = 16
```

Edges were ordered by `(timestamp_us, storage_index)`, restricted to a micro-packet and truncated
by stable normalized spatiotemporal distance. Edge attributes were:

```text
[dx/W, dy/H, log1p(dt/tau0), p_src, p_dst, p_src*p_dst]
```

The message primitive used a hidden-64 shared node/message MLP, normalized `index_add` aggregation
and a GRU update. There was no `torch.cdist(N,N)`, square mask, PyG or new dependency.

Observed results:

- N=1/2/8/32/128 with K=8/16 matched the brute-force neighbor set and ordering exactly
- future/cross-batch edges: `0`
- indegree `<=16`, edges `<=16N`
- finite and non-zero gradients
- 2N/N build p50/p95 ratio: `1.844609 / 2.318488`, Gate `<=2.8`
- real N=256 graph: 1,356 edges
- build p95: `3.977793 ms`
- CUDA message p95: `0.770099 ms`
- build+transfer+message p95: `4.956005 ms`, Gate `<=10 ms`
- maximum 50 ms window: 287,387 events, 1,127 packets, 1,684,552 edges, no OOM

Two implementation-only performance repairs replaced Python hash probing with a sorted integer-key
and vectorized `searchsorted` join while preserving exact neighbors.

`GATE_VERDICT=PASS`, `REVERT_OR_KEEP=KEEP`, `NEXT_STAGE_AUTHORIZED=YES_E3_ONLY`.

## 6. E3 RawGraph-Global control

### 6.1 System under test

The transient E3 tracker executed:

```text
raw 50 ms interval
  -> E1 fixed-count=256 / dt_max=5 ms micro-packets
  -> 7D raw event token
  -> exact-pixel previous-MANO [silhouette, inverse-depth]
  -> 9D node token
  -> E2 packet-local causal graph, one hidden=64 message layer
  -> interval last + mean + max (192D)
  -> MLP to 256D + previous-state MLP
  -> global 51D delta
  -> pred = prev + delta
  -> zero events only: pred = prev bitwise
```

The model contained 181,619 parameters versus B2's 11,209,429 (`0.01620x`). It did not construct a
ResNet/LNES inference trunk, joint router or MANO-tree GNN.

### 6.2 Correctness and overfit receipts

- fp32/bf16 forward and backward: PASS
- empty-packet identity: PASS
- same-prefix determinism maximum difference: `5.96e-8`
- GT mutation and two-stream isolation tests: PASS
- finite graph gradients and strict checkpoint load: PASS
- initial fixed-128 overfit loss drop: `87.34%`, FAIL `<90%`
- controlled repair 1 overfit loss drop: `95.329%`, PASS
- overfit RA: `5.4875 -> 2.6940 mm`
- return-previous RA: `2.7756 mm`
- predicted delta L2: `0.27988`

The overfit result proves that the computation graph was trainable, but it does not prove that the
event branch was useful: the trained result improved over returning the previous state by only
about `0.0816 mm`, and no event-shuffle control was run.

### 6.3 Recursive evaluation

B2 Gate ceilings were seed3407 `18.301304 mm` and seed3408 `19.518413 mm`.

| Arm | Seed | Step | RA-MPJPE mm | Absolute MPJPE mm | Low-event RA mm |
|---|---:|---:|---:|---:|---:|
| no-distill | 3407 | 500 | 116.7936793136 | 84,628,770.0865 | 112.6273589154 |
| no-distill | 3407 | 1000 | **93.2883257503** | 1,459.8503 | 104.9098880593 |
| no-distill | 3408 | 500 | 131.8621032560 | 16,746.5963 | 127.2343664052 |
| no-distill | 3408 | 1000 | 100.4025437931 | 317.7800 | 98.9365112259 |
| Lie/FK distill | 3407 | 500 | 113.8868057325 | 131,642,024.6204 | 105.2924813943 |
| Lie/FK distill | 3407 | 1000 | 122.3156870847 | 8,466,375.8401 | 120.3638422568 |
| Lie/FK distill | 3408 | 500 | 102.1956012024 | 12,296,020,728.3092 | 93.3870940655 |
| Lie/FK distill | 3408 | 1000 | 100.3591692334 | 34,942.3754 | **92.1482407590** |

Every row contains 8 sequences and 10,452 frames. All four arms were finite and completed without
OOM, but global/local/low-event accuracy all failed. The best E3 RA was still about 5.61 times its
corresponding B2 result.

The frozen teacher was seed-matched B2 step3500. Student checkpoints contained 34 student keys and
zero teacher keys and strict-loaded successfully. Distillation therefore did not leak the teacher
into inference; it also did not provide a repeatable rescue.

### 6.4 Latency and memory

| Window | Events | Graph-build p95 | Prebuilt-forward p95 | Peak allocated |
|---|---:|---:|---:|---:|
| low | 1,261 | 12.355 ms | 8.892 ms | 17,591,808 B |
| median | 44,972 | 433.269 ms | 32.958 ms | 300,621,824 B |
| high | 287,387 | 2,786.525 ms | 177.456 ms | 1,766,498,816 B |

The long tail explains why synchronous full-window evaluation was slow, but it does not explain
the 93--122 mm tracking error because there was no OOM or timeout truncation.

### 6.5 Artifact cleanup

Before cleanup, no-distill config SHA-256 was
`63537ee37fc8ab5f6904cea1b1fd82e121600d96b276b68e43dbdb576a40e43e`; distill config SHA-256 was
`66a447a91940ffec12eef0563e7b5971249445bb01bd35975343ee03ae4a28f2`.

After the NO-GO decision, the four exact E3 output directories were deleted: 32 files and
45,324,338 bytes. The transient tracker, two configs, E3 test and train/eval entry changes were
also removed. General E1 packet and E2 graph primitives were retained. Cleanup was followed by
88 passed/1 skipped tests, E1/E2 smoke checks and bit-equivalent B0 recursive parity.

`GATE_VERDICT=NO-GO`, `REVERT_OR_KEEP=REVERT_E3_KEEP_E1_E2`,
`NEXT_STAGE_AUTHORIZED=NO`.

## 7. Detailed E3 failure analysis

### 7.1 Primary structural defect: cross-packet order aliasing (`code-proven`)

`semkine/packetizer.py` subtracts each packet's relative start from event time. The 7D event token
uses only packet-local normalized time and inter-event/SAE time. E2 disallows every cross-packet
edge. The recorded E3 interval readout was `last+mean+max`, and it did not feed `packet_id`, outer
`t_start_us`, query age or trailing-silence duration into the event token.

Let packet `i` produce a packet-local node multiset `H_i`. E3 formed:

```text
z = concat(last(H_m), mean(union_i H_i), max(union_i H_i))
```

For any permutation of `H_1...H_(m-1)` that leaves the final packet unchanged, `mean`, `max` and
`last` are unchanged. Consequently, two 50 ms trajectories that contain the same packet-local
event sets in different orders can map to the same E3 representation even when their motion
directions differ. This is an information-identifiability failure, not merely something the
optimizer failed to learn.

The finding applies to this packet-local E3 control. It does not rule out raw graphs that encode
packet-to-query age, maintain causal state across packets or add legal cross-packet temporal edges.

### 7.2 Single-step teacher forcing versus recursive rollout (`code-proven + inferred dynamics`)

Training approximately optimized:

```text
s_hat(t+1) = s_GT(t) + epsilon + Delta(E_t, s_GT(t)+epsilon)
```

Recursive evaluation executed:

```text
s_hat(t+1) = s_hat(t) + Delta(E_t, s_hat(t))
```

Near a ground-truth trajectory, the error evolves approximately as:

```text
e(t+1) ~= [I + dDelta/ds] e(t) + bias
```

The one-step loss did not constrain the transition gain, delta bias, student-generated state
distribution or update strength for nonzero-but-uninformative windows. The zero-event gate only
sets delta to zero when `N=0`; one background or hot event still enables a full global 51D update.

This mismatch is directly established by the data/evaluator paths. A transition gain above one is
still `inferred`, because the E3 cleanup removed checkpoints before a horizon ladder or Jacobian
probe was retained. The extreme absolute-MPJPE tail is consistent with translation/root delta
accumulation but is not by itself a Jacobian measurement.

### 7.3 Exact previous-geometry query is sparse (`observed`)

The retained renderer projects and rounds 778 MANO vertices and splats those points into the
silhouette/depth images; it does not fill mesh triangles. E3 queried those images at exact integer
event pixels with nearest-neighbor lookup.

A read-only diagnostic used B2 seed3407 step3500's identical renderer and 32 val_core raw windows:
8 sequences, four windows per sequence at the 20/40/60/80% index positions. It covered 672,270
events with median 10,049.5 events/window. The silhouette contained a mean 627.125 support pixels
(median 641, range 467--696).

| Support radius | Mean hit | Median | p25 | p75 | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 px | 0.29179433 | 0.24325322 | 0.20009322 | 0.36612096 | 0.18060434 | 0.53668225 |
| 1 px | 0.76745026 | 0.74391159 | 0.66654487 | 0.85756008 | 0.61984670 | 0.94836450 |
| 2 px | 0.87979717 | 0.87713474 | 0.83202584 | 0.93506218 | 0.75402814 | 0.99386317 |
| 4 px | 0.91832720 | 0.92204800 | 0.88252750 | 0.97113350 | 0.80312920 | 0.99601102 |
| 8 px | 0.93952662 | 0.94735205 | 0.90654232 | 0.98049957 | 0.84070104 | 0.99693155 |

At radius0, `70.820567%` of events received zero silhouette and inverse-depth support. Expanding the
same mask by one pixel increased coverage by `47.565593` percentage points. Exact point lookup is
therefore highly sensitive to small projection errors.

This diagnostic used GT previous states, not failed E3 rollout states. The proposed positive
feedback—pose drift reduces geometry hits, which weakens corrections and increases drift—remains
`inferred` until geometry hit rate and pose error are recorded together across a recursive run.

### 7.4 Previous-state shortcut (`inferred`)

The previous-state MLP received a strong 50 ms pose prior. On the fixed overfit set, E3's final
`2.6940 mm` RA improved over return-previous `2.7756 mm` by only `0.0816 mm`. A nonzero delta proves
that the model changes state; it does not prove that causal event content determines the change.

Required negative controls are event shuffle with matched previous state/count/rate, event-content
replacement with preserved packet layout, and previous-state shuffle with fixed events.

### 7.5 Global event pooling is sensitive to nuisance multiplicity (`inferred`)

- `mean` weights packets in proportion to event count;
- `max` is sensitive to one outlier or hot event;
- `last` contains one terminal event, not the terminal packet distribution or trailing silence;
- sparse geometry was concatenated, not used to reject background events;
- one global 51D head could not establish which joint an event supports.

Domrand keep-rate/hot-pixel changes event multiplicity. LNES instead saturates repeat activity at a
pixel to its latest timestamp and lets a CNN spatially mix the result. Duplicate/thinning controls
were not run, so nuisance sensitivity remains an explanation rather than an observed causal fact.

### 7.6 Training exposure is a material confound (`observed`)

| Quantity | E3 scout | B2 selected control |
|---|---:|---:|
| Batch x accumulation | `8 x 8` | `1024 x 1` |
| Effective samples/optimizer step | 64 | 1024 |
| Evaluated/selected step | 1000 | 3500 |
| Approximate sample exposures | 64,000 | 3,584,000 |
| Train dataset size | 2,491,120 | 2,491,120 |
| Approximate dataset passes | 0.0257 | 1.4387 |
| LR / warmup | `1e-3 / 500` | `4e-3 / 500` |

E3 saw about 1/56 as many samples as B2, and half its optimizer steps were warmup. Its no-distill
RA was still improving from step500 to step1000: `116.79 -> 93.29 mm` and
`131.86 -> 100.40 mm`. Therefore the experiment proves that E3 failed its preregistered scout; it
does not prove that every sufficiently trained raw causal graph architecture must fail.

This limitation does not reverse the NO-GO. The fixed budget was intended to prevent an
unsuccessful branch from absorbing unlimited compute. A broader architecture-level negative claim
would require matched sample exposure or a preregistered convergence criterion.

### 7.7 Why distillation did not rescue E3

The B2 teacher observed a 50 ms LNES recency surface. E3 had already discarded non-final packet
order, so different teacher inputs/outputs could collapse to the same student observation. The
student can only learn a conditional average when its representation is insufficient.

Distillation also remained teacher-forced on GT/noised-GT previous state. Lie/FK output-space
supervision cannot recover missing input order or guarantee stability on student-generated states.
Observed seed behavior was inconsistent: seed3407 became worse at step1000, while seed3408 was
nearly unchanged.

## 8. Ruled-out and unresolved explanations

| Hypothesis | Status | Evidence |
|---|---|---|
| raw microsecond events were lost/duplicated | refuted | E1 bitwise round-trip; all violation counts zero |
| graph included future/cross-batch edges | refuted | E2 brute-force parity and causality tests |
| NaN/OOM produced corrupted checkpoints | refuted | all arms finite and completed |
| teacher leaked into inference | refuted | 34 student keys, zero teacher keys, strict load |
| evaluator protocol changed | refuted | same val_core/step50/seed0/noise/trust; B0 parity retained |
| too many parameters caused overfit | unlikely | 181,619 parameters, 1.62% of B2 |
| too few layers/parameters caused underfit | plausible/pending | one hidden-64 layer plus 56x exposure confound |
| missing joint router explains everything | refuted as sole cause | root/global/local/low-event all failed before routing |
| additive axis-angle is the primary difference | unlikely | B2 used the same 51D family and stayed stable |
| graph runtime caused accuracy failure | refuted | slow long tail, but no truncation/OOM |

## 9. Decisive validation queue if E3 is explicitly reopened

These tests are not currently authorized; they define the cheapest future evidence path.

| ID | Test | What is preserved/destroyed | Decision rule |
|---|---|---|---|
| K1 | Permute non-final packet order with fixed weights and final packet | preserves every event/local graph; destroys outer order | numerically identical predictions with different target motion confirms aliasing |
| K2 | Shuffle matched-count/rate events while keeping previous state | preserves nuisance/layout; destroys event semantics | retained effect kills the claim that event content drives delta |
| K3 | Evaluate GT-prev one-step and 2/4/8/16-step predicted-prev | same checkpoint/observation; changes feedback horizon | sharp monotonic growth isolates transition instability |
| K4 | Finite-difference previous translation/root/local coordinates | fixes events; measures transition state gain | gain above one correlated with drift supports the Jacobian mechanism |
| K5 | Compare exact/1px/2px/distance geometry against equal-density shifted masks | preserves support density in negative control; destroys alignment | only aligned support may count as geometry benefit |
| K6 | Duplicate or thin events while preserving support/timing distribution | changes multiplicity, not pose | large delta change confirms pooling nuisance sensitivity |
| K7 | Match cumulative exposure or train to a preregistered plateau | removes undertraining confound | only meaningful after K1/K2 representation checks pass |
| K8 | One-step, 4-step GT-prev, scheduled and full-predicted training | changes state distribution, keeps frontend | full-predicted gain supports teacher-forcing mismatch |

Priority is K1, K2, K3. Increasing capacity or training duration before those tests would spend
compute without resolving whether the representation contains the needed information.

## 10. Claim boundary and final verdict

Allowed statement:

> E3 RawGraph-Global failed the registered two-seed, 1000-step B2-domrand scout and two controlled
> repairs. Recursive RA and absolute stability remained far outside the Gate, so the E3-specific
> implementation was removed while E1/E2 general primitives were retained. The postmortem found
> a formal non-final-packet order invariance, a teacher-forced/on-policy mismatch and only 29.18%
> exact previous-geometry event support. Training exposure was 56x below B2, limiting the negative
> conclusion to this E3 control and budget.

Disallowed statements:

- “event graphs are unsuitable for hand tracking”;
- “raw events are inherently worse than LNES”;
- “the router or MANO-tree GNN failed” when E4A/E5 were not run;
- “damping proves event evidence is useful” without the K2 mechanism control;
- any SOTA, asynchronous acceleration or kinematic-graph contribution based on E3.

Final state:

```text
GATE_VERDICT=NO-GO
REVERT_OR_KEEP=REVERT_E3_KEEP_E1_E2
NEXT_STAGE_AUTHORIZED=NO
```
