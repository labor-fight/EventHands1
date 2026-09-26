# SC0 — frozen S37 pre-pool context diagnostic

2026-09-26. **Pre-registered before implementation Debug, any fitting, or new dev scoring.**
Admission: `research_state/debug/SC0_spatial_context_admission_20260926.md` (independent G21/G22 audit). C0r diagnostic only; not a fourth complete candidate, not admission to main training. S37 remains adopted.

## Question and limits

Do trainable interactions between an original S37 node descriptor and descriptors beyond its existing explicit message ancestry help more than an equal-capacity local adapter? Original three W32 layers, seven event tokens, cap 2048, routing, pooled readout, heads, history, loss, H50ms feedback remain. Token SAE statistics and packet pooling already access broader evidence: this is **not** a claim that S37 lacks all remote information, that remote indices mean spatially distant fingers, or that a new independent observation has been introduced. Existing NG1 fixed-weight neighbor substitution, X1c absolute hierarchy, mesh/EGM and R0 pooled-head diagnostics do not isolate this interface. Their failed gates remain closed. This familiar graph-context operation is not claimed as scientific novelty or asynchronous execution.

## Fixed arms

A is current S37 seed3407 step2500 (weights only, same source for all arms). L and F add one residual message MLP after original layer3 and before all original pooling, with input `[h_j-h_i, p_j-p_i]`, 131→32→128, ReLU only between linear layers. Last weight and bias zero. First-layer initialization uses seed3407 or3408, identically paired for L/F. Old parameters and buffers remain unchanged; only four adapter tensors train. Old graph edge-summary g is retained. Same original normalized x/W,y/H,packet time times t_scale; t is not z.

L selects min(8,i) distinct earlier sources from the preceding min(32,i) nodes. For source count n and degree k, bucket midpoint index is `floor((2*s+1)*n/(2*k))`, s=0..k−1. F uses the identical rule over indices `[0,i−129]` when i≥136 (at least eight sources); otherwise it uses the L sources. No valid source yields zero residual. Sources and queries must be live, sources strictly earlier. The original3 layers can reach i−96; adding L reaches i−128, so F's newly accessed third-layer descriptor is outside that explicit four-layer ancestry. This is an index/computational-path statement, not a physical-radius/information theorem.

Both arms have identical source count per query, shape, initial function, optimization and old source. No new head, attention, geometry conditioning, memory, altered original kNN or learned selection. Wrapper may repeat original global projection; all such overhead is counted for diagnostic timing, with no production latency claim.

## Fitting protocol and identity

Two seeds 3407/3408, each L/F **500 optimizer updates, batch16, single GPU, FP32, Adam lr0.001, wd0, no scheduler, no accumulation**, final step500 only. These are explicitly new small adapter fits, not original S37 batch1024/LR0.004 continuation. Seed controls adapter and dataset augmentation; an independent seeded generator fixes the sampled index permutation, paired exactly between L/F. All original training window/noise/augmentation rules and original MSE51D then log10 objective remain. The filtered diagnostic manifest renumbers dataset indices, so individual noise/augmentation instances differ from the old nine-subject trajectory; L/F use exactly the same new index and perturbation for each paired seed, with no exact-continuation claim. Original weights are strictly loaded before attaching the adapter; source Adam/scheduler/RNG are not resumed.

Fit uses all64 original sequence variants for subjects ch,lfz,lpc,lr,ly,lyh,lyq,ycy. All ylf variants excluded from adapter fitting. Diagnostic manifest is a separate scratch file; original split file untouched. Source backbone did see ylf, and ylf has been reused in earlier development; this is NOT independent held-out generalization. No zgz loading or selection. Explicit source/checkpoint/config/split/asset/data identities and source hashes are frozen in scratch contract and per-run complete recovery contracts, including scratch entry and adapter code.

Every100 committed updates save model, Adam, step, loops, per-rank RNG and committed sampler via existing CompleteCheckpoint. Use strict deterministic mode. Scratch code hashes are asserted at startup and bound into normalized config. A complete checkpoint is a recovery artifact, not permission to extend timeout or select a better step. No silent retry, seed replacement, checkpoint selection, width/window/LR/length scan, unfreezing or fallback.

## Mandatory Debug before screen

Synthetic tests: index bounds, causal/live/empty masks, unique matching degrees, remote support at boundary, initial A/L/F equality FP32 and BF16, gradients through residual, only adapter updates. Real source-checkpoint regression uses existing eight training packets, including original sampled-node cap. New real fitting Debug uses first16 samples of the seed3407 fixed fit permutation; compare initialization with A, finite predictions/loss/gradients, unchanged old tensors, Adam changes only adapter, first-layer gradient after second update, same-batch16-update final loss strictly below initial, empty packet exact prev. Check sample/sequence isolation and counts; no label-selected packets. Measure actual FP32 batch step wall/memory to ensure the planned short fit is feasible. Check deterministic interrupted/resumed next-update equality with actual optimizer/RNG/sampler contract; existing CompleteCheckpoint tests remain required. All required checks must pass to start screen; implementation errors may be repaired with preserved failed receipts inside fixed total budget, not relaxed gates.

## Development evaluation

A plus all four final adapters separately use **their own closed-loop state**, same original H init noise RNG0, no damping, 50ms window/step including the original final1ms bin, original GT betas, unchanged original evaluator. Fixed sequence order ylf_global then ylf_local, first128 eligible H frames of each, valid-run resets retained. The exact endpoint/run prefix is frozen before scoring, and all frames remain, including empty/no-route/no-remote rows. This is a diagnostic prefix, not the full original zgz main-row evaluation. Predictions, states, targets, ends and parameter/checkpoint identities must be preserved for independent rescoring. Evaluation uses FP32 for all arms. No output-dependent frame filtering.

No dev metric may influence fitting or stopping. Seal all final predictions/identities and finite checks before aggregate gate calculation. Any nonfinite frame in any arm fails completion/utility, with failed data preserved, not discarded. Original evaluator used to score; independent saved-prediction FK confirms identity of scoring implementation.

## Frozen continuation gate

F must improve local mean across its two seeds by at least **1.1mm relative to A and relative to paired L**. Local improvement direction must be strictly positive for each seed versus both A and its paired L. F global mean regression versus A and L must each be ≤0.3mm. All prediction streams complete and finite, initialization/paired-data/frozen-old-weight identities pass. These thresholds follow existing training scatter discipline, not final hard accuracy targets.

Passing only keeps a pre-pool remote-context interface eligible for bounded cost/replication audit, not adoption, final accuracy, novelty, or ≤7ms. F tied/worse than L closes this fixed form; common L/F gains do not establish remote benefit. Failure does not prove all jointly retrained wider graphs impossible, and does not authorize rescue by unfreezing. Insufficient support or timeout is recorded separately as STOP_SUPPORT/INCOMPLETE.

## Fixed resources and stop

Total allocated wall including initialization and cleanup across this SC0 diagnostic ≤1800 GPU-seconds, with no transfer of stage budgets. Planned maxima: Debug230+10 cleanup=240; four screen fits each300+10=1240; single final evaluation300+10=310; total1790 leaving10 unallocated. All GPU work through research_state/tools/budget_run.py, idle authorized GPUs1–7 only, neverGPU0. Debug additional attempts reduce the remaining diagnostic budget; unused caps do not authorize longer fitting. Parent must verify measured throughput before launch. Real training workers4/job, torch/BLAS4 threads; no multiprocessing CPU analysis. Independent fits may use idle cards concurrently after Debug.

No new main row for this diagnostic. User-facing reporting retains generated baseline+S37 table; detailed diagnostic receipts and failures live here and in FAILURE_AND_CLEANUP_LEDGER.md.
