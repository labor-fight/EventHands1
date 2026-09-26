# Final independent CPU contract review

Date: 2026-09-25. Reviewer: delegated agent `/root/audit_s37`. Scope: read-only inspection of `semkine/streaming.py`, `semkine/streaming_tracker.py`, both `tests/test_streaming*.py`, `.research/s37_async_20260925/streaming_tests_v2.log`, `tiny_fit.py` and `tiny_fit.json`. The reviewer originally implemented the encoder but did not implement the tracker or its formal tests; this is an independent pass over those integration/tests, not an external independent replication claim.

No production file was changed during this review. One bounded CPU counterexample was executed, four threads, no optimizer/training/GPU. Pending GPU or accuracy work is not classified as a software bug.

## Finding R1 — full-model parameter updates do not invalidate tracker state

**Status: observed counterexample; mandatory contract remains failed until repaired and regressed.**

`CausalEventEncoder._version` records `self.encoder.named_parameters()` (`semkine/streaming.py:91-93`), and tracker query validates that graph state through encoder readout. `StreamTrackingState` currently carries no signature of the rest of the model. Consequently, updating a root/finger head or the previous-state MLP while leaving the event encoder unchanged permits a historical pose produced by old weights to continue with new weights. This violates the frozen `DECISION.md` requirement that parameter updates require explicit stream reset.

This is not the documented unsupported `.data` mutation escape. The counterexample used ordinary tracked `with torch.no_grad(): root_head.bias.add_(0.001)`. PyTorch increased that parameter's `_version` from 1 to 2, but the unchanged encoder signature still matched and query accepted the old state. Evaluating the same queued state/events changed the predicted pose by approximately 0.001. A head-only fine-tuning update is a practical trigger; standard all-parameter training happens to update the encoder too and therefore did not expose this gap in the existing test.

Command and evidence:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python \
  .research/s37_async_20260925/audit/review_final_counterexample.py
```

Raw evidence: `.research/s37_async_20260925/audit/review_final_counterexample.json`; reported script wall time approximately 1.7 s. No training or optimizer operation took place; the temporary untrained tiny model existed only inside this process.

**Proposed minimal repair:** put a complete model parameter identity/version/device/dtype signature in `StreamTrackingState` at initialization, and validate it on both append and query, including the no-new-events path. Keep encoder's narrower signature for its own standalone cache. Add targeted tests for separately updating root head, a finger head and previous-state MLP, plus fresh reset succeeding. No architectural change is required.

## Existing evidence that was actually read

- **Observed log:** `streaming_tests_v2.log` records 35 passed tests in 4.64 s, with dependency deprecation warnings. This supports its tested contracts, not R1: the parameter-update test uses SGD over the whole encoder and cannot catch a head-only tracker update.
- **Observed code coverage:** tests compare different chunkings after readout eviction, every cached layer against a complete-prefix reference, gradients between incremental/reference execution, timestamp origin shifts, same-time polarity ages, input-buffer reuse, long-gap zero messages, invalid inputs, empty state, query watermark/cadence, no-new-event pose preservation, stream isolation, explicit K/beta and eight-step mesh recurrence.
- **Observed artifact and script:** `tiny_fit.json` records 64 bounded CPU optimization steps using two training-subject fixtures, final/initial loss ratio approximately 0.041, changed encoder weights, no checkpoint write and no GT inference inputs. Reading `tiny_fit.py` confirms GT poses are used as loss targets; the tracker initializes neutral pose and mean shape, uses causal two-by-4-ms event packets and resets state on every optimization iteration. This is evidence of this small supervised optimization path working, not generalization, robust initialization, latency or target accuracy.

## Static findings without additional execution

1. **Long-history graph semantics are coherent:** append saves h^0 through h^L for the last W predecessors, and new layer-l nodes use old/same-microbatch layer-(l−1) features. Older dependencies are already represented in those features. The independent reference builds edges over the complete prefix, then trims only readout. No sliding-window/full-prefix confusion was found in this pass.
2. **Temporal ties and integer differencing are coherent:** ordinal-based SAE search separates previously unseen from seen-at-the-same-timestamp; int64 subtraction precedes floating conversion. Fixed H normalization removes dependence on future packet span. The tested time-origin control supports this definition.
3. **Cache ownership is now explicit:** retained coordinate/time suffixes and hidden-feature suffixes are cloned. The prior input-buffer alias defect is covered by the new buffer-reuse test. Frozen dataclasses do not make tensors immutable; callers are explicitly forbidden to mutate state tensor contents.
4. **No-new-events state behavior is correct in the inspected path:** `pending_events` is reset after each query and gates delta integration, while final MANO decoding still runs. Old evidence is not repeatedly integrated in event-free intervals. R1 is still relevant because signature validation must occur even on such intervals.
5. **Gradient boundaries are deliberate:** sparse ordinal/index decisions and old `_route_nodes` geometry are not differentiable, while cached event features and head/state additions retain autograd. No new accidental detach was found. Long training histories can retain autograd graphs despite bounded tensor caches; this is explicitly documented and requires a bounded unroll/reset schedule, not silent detachment.
6. **Half-open query capacity is explicitly narrower than an arbitrary historical query:** the encoder keeps last N accepted nodes before applying `[now-H,now)`. At `now==last_timestamp`, same-time tail events consume cache capacity but are omitted. This is documented and shared by the reference. The tracker prohibits `last_event_us>=query`, so its normal queries avoid this boundary. It is not classified as a defect against the frozen current contract.
7. **GT/state separation is preserved:** tracker API accepts events, calibration and its explicit state; neutral initialization and fixed zero beta are distinct from old GT initialization/GT sequence beta. Meaningful real cold-start/recovery performance remains unknown, without invalidating this API separation.

## Remaining validation boundaries

No additional bug was established by this pass. This does not prove absence of bugs. Specifically, this review did not rerun the formal suite, tiny fitting, long GPU/bf16 streams, actual load profiling or accuracy evaluation. Those are pending evidence scopes, not grounds to call this encoder implementation incorrect.

The appropriate engineering status at this review point is **CPU Debug incomplete because R1 is confirmed**, despite the valid existing test/tiny-fit successes. After the minimal version-signature repair, the affected contracts must be rerun and their new evidence linked; the finding should remain in this review as provenance, with a separate resolution note. No claim of method effectiveness, novelty or goal completion follows from closing R1.

## R1 resolution follow-up — read-only verification

The parent implemented the repair after receiving the counterexample. The original finding and its JSON are preserved above; this follow-up did not run another counterexample or repeat training.

**Observed code:** `StreamTrackingState.model_version` is now stored at initialization (`semkine/streaming_tracker.py:38,108-110`). `_model_version` enumerates the full model's parameter identities, PyTorch versions, devices and dtypes (`:86-94`). Both append and query invoke validation before their empty/no-new-event paths (`:112-126`). This covers the head-only change that escaped the encoder signature.

**Observed test additions:** `test_head_weight_change_requires_explicit_tracking_reset` covers root head, joint heads and previous-state MLP, each with empty and nonempty historical state. It requires rejection on both query and empty append after a regular tracked in-place parameter change, then verifies a fresh initialization works.

**Observed new log:** `.research/s37_async_20260925/streaming_tests_v3.log` ends with `41 passed, 18 warnings in 5.38s`. The six added cases address the identified scope rather than merely repeating the former all-encoder update test.

**Updated judgment:** R1 is resolved for normal PyTorch-tracked parameter changes, supported by inspected code and targeted passing regressions. No further CPU implementation defect was established in this bounded review. Existing limitations—unsupported `.data` mutation, caller mutation of state tensor contents, GPU performance/precision, real initialization/recovery and final accuracy—remain explicitly distinct from this resolved bug.
