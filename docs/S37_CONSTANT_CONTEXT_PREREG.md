# CC0 — constant pre-pool calibration versus the saved local message adapter

2026-09-26. Frozen before new numerical Debug, fitting, or suffix scoring. This is a separate C0r engineering discriminant motivated by SC0's shared L/F improvement, not a rescue of SC0 F. Original SC0 gates and stop remain. Admission report: `research_state/debug/CC0_constant_context_admission_20260926.md`.

## Question and the exact simpler model

SC0 did not show remote-specific advantage over the local adapter. Both adapters improved the same source on the reused development prefix. Is an extra input-dependent event-message block necessary to obtain comparable engineering improvement, or can a constant feature calibration suffice?

A is unchanged S37 seed3407 step2500. L3407/L3408 are the **already completed, unchanged SC0 L step500 checkpoints**; no retraining or selection. C3407/C3408 fit only a zero-initialized 128-vector b after the original third event layer and before original pooling:

`h'_i = h_i + q_i b`, where q_i=1 iff node i is live and has at least one live predecessor under the same SC0 L source mask; otherwise q_i=0.

C is the SC0 residual MLP submodel obtained by W2=0 and last bias=b. Implementation may broadcast b across the same valid message slots and use the same mean arithmetic to preserve finite-precision nesting. No new event/position/previous-state values enter this vector. Original graph, token normalization, 2048 cap, pooling, routing, root/finger heads, prev_mlp and H feedback remain. Zero source/empty packet contracts remain.

C is **not a constant pose/output update**: the old features, route assignments and nonlinear decoder remain input/state dependent. For a joint's weighted mean, its change is b times the routed mass of eligible nodes divided by total routed mass; the max can change its winner. It can alter decoder thresholds and feedback gain without adding information. Matching C with L cannot prove that the trained L internally only used its own b2. If L beats C, it only rejects this constant submodel, not every calibration or per-node nonlinear re-encoding explanation, and does not establish neighbor-interaction necessity or novelty.

P1 scaled old output deltas without training; CI1 subtracted same-state zero-evidence head outputs; R0 fit new finger heads on pooled frozen histories. These do not fit this nested shared feature-bias under the original heads and each model's own feedback. Known bias/adapter calibration is not presented as a novel method.

## Fixed fitting and provenance

Only two new fits, seeds3407/3408, each500 optimizer updates, batch16, FP32, Adam0.001/wd0, no scheduler or accumulation, fixed final500. Only b128 updates; all original parameters/buffers exact frozen. Same source S37 as SC0, weights-only load/fresh Adam. Both b initializations are zero; the seeds vary data order and pure(seed,index) augmentation, not source backbone.

Use exactly the SC0 64-sequence fit manifest, original dataset index ordering, window/noise/augmentation recipe and independent seeded permutation. Assert index and full sampler-order identities against the corresponding saved SC0 L run; first two actual batch hashes also match. No ylf adapter fitting, no zgz access. Save each100 committed steps through existing CompleteCheckpoint; bind SC0 reused source and all new code/config/asset/data identities, with actual optimizer/RNG/sampler/loops. Checkpoints are recovery artifacts, not permission to extend a failed run.

Mandatory Debug: zero C=A in FP32/BF16; finite-precision nesting against SC0 W2=0,b2=b with exact valid-degree mask; only128 trainable parameters and finite/nonzero gradients/updates; original tensor identity; empty/one-node and padding contracts; same real16 samples fit16 updates ending below initial loss; actual4-update continuous versus2+2 complete recovery and loss-trace equality. CPU library threads4, no analysis sharding. All checks before fitting; model Debug changes require retest.

## New locked development suffix, own history

Use ylf_global then ylf_local. For **A, saved L3407/L3408, and new C3407/C3408**, replay each sequence from its original legal H initial state over the first256 eligible 50ms frames. Preserve original valid-run boundaries and original H final1ms bin. Carry each model's own predictions throughout; never reset at frame129, use A's state, or initialize the scored suffix with GT. Same init RNG0, GT betas and original MANO/add-mean convention for all arms.

Primary score is only eligible frames129–256 (1-based) in each sequence, all128 frames with no support/output-dependent filtering. A/L first128 predictions must reproduce their sealed SC0 prefix exactly in the same FP32 execution; that prefix is a continuity regression, not a score to select or combine with the suffix. C also runs through it before scoring, with no tuning on its prefix results. Freeze actual endpoints/valid-run starts and provenance before scoring. Verify original full256 evaluator scores by saved-prediction FK, then score the fixed suffix with the same per-frame metric formula.

This suffix is selected by position before inspecting its scores, not by SC0's first128 errors. It is **not claimed never used elsewhere**, independent temporal data, or independent generalization: it is adjacent to a previously used prefix, same sequences, source backbone saw ylf, and earlier project development reused ylf. No global/local or seed-specific model selection; all five streams complete/finite and identities sealed before joint gates.

## Fixed decisions, no post-hoc adoption

Define utility U(X), X=C or L: two-seed mean local improvement over A ≥1.1mm, each seed's local improvement strictly positive, and global mean regression versus A ≤0.3mm. This describes a fixed suffix engineering observation, not final targets.

C sufficiency gate: U(C), plus C relative to its paired L is no worse by >0.5mm local or >0.3mm global for **both individual seeds and both two-seed means**. These are engineering tolerances, not statistical equivalence claims. If all pass, record `CONSTANT_CALIBRATION_SUFFICIENT_ON_FIXED_SUFFIX`: comparable utility here does not require the extra message block. Only the simpler calibration merits a separately specified cost/replication decision; no automatic adoption or main training.

If C does not pass but U(L) passes, record `CONSTANT_SUBMODEL_NOT_SUFFICIENT_NO_MESSAGE_NECESSITY_PROOF`. This only excludes this fixed constant comparator; no automatic unfreezing, new pointwise adapter, extra message layer, or main training.

If neither U(C) nor U(L) passes, record `STOP_SHARED_GAIN_NOT_TRANSFERRED_TO_FIXED_SUFFIX`. If U(C) passes but misses the L tolerance, record `CONSTANT_UTILITY_WITHOUT_MATCHING_L`; no post-hoc winner, precision aggregation or adoption. All statuses close this fixed question. Any incomplete/nonfinite/identity-failed stream gives INCOMPLETE/DEBUG_FAIL, not a utility verdict. No parameter/window/LR/seed/prefix/checkpoint/budget retries to rescue a failed scientific gate.

## Resources and reporting

Total allocated wall including cleanup ≤640 GPU-seconds: Debug120+10=130, each of two screen fits150+10=320 total, one evaluation180+10=190. GPU jobs only through budget_run.py, idle authorized GPU1–7, neverGPU0; stage caps remain separate. Four DataLoader workers per real fit, four library threads; no extra CPU compute workers for analysis. Core/runtime/old SC0 source/checkpoints/official splits remain unchanged. Reuse scratch interfaces; diagnostic bias path may still compute the inherited local edges, so no deployment timing or ≤7ms claim.

Keep generated baseline+S37 main table. No CC0 main row, no zgz evaluation, and no scientific innovation claim. Original SC0 F remains closed regardless of CC0 outcome.
