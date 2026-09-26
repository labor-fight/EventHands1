# C1 minimal encoder implementation hand-off

2026-09-25; follows frozen `DECISION.md` §4–6. The only production file added by this agent is `semkine/streaming.py`; no original model, config, evaluator, history or tests were edited. No GPU or formal training was used.

## API

```python
runner = CausalEventEncoder(event_gnn, horizon_us=50000,
                            max_events_per_append=8192)
state = runner.append(xyp, timestamps, state=None, stream_id=0)
feat, h, px, py, mask = runner.readout(state, now_us)
reference = runner.full_reference(all_xyp, all_timestamps, now_us)
```

The runner is a plain Python object referencing the caller's existing EventGNN, with no new parameters or duplicate module registration. Inputs are floating `(M,3)` integer pixel coordinates and 0/1 polarity, matching the encoder parameter dtype/device; timestamps are same-device int64 microseconds. `now_us` is a Python integer, must be at least the newest accepted timestamp, and output support is half-open `[now-H,now)`. Times may have any int64 origin as long as required differences fit int64.

Tokens exactly follow the frozen arrival-final seven-channel specification. Sparse SAE is an ordered tensor map (`sae_keys`, `sae_timestamps`), not an image: history has ordinal zero, new events use arrival ordinals; sorting and `searchsorted` find earlier same/opposite polarity including zero-age ties. Old entries are replaced by each key's latest timestamp. No per-event Python token loop or implicit sorting of input events occurs.

`append` constructs predecessor edges for all new events in a microbatch and updates one entire layer at a time. It reads previous-layer features for cached and earlier same-batch nodes. State includes each layer's last W features, final-layer last N readout nodes, their coordinates/times, sparse SAE, last timestamp, accepted count, stream identity, parameter identity/version/device/dtype, runner configuration signature, input dtype/device and feature compute dtype. It is a frozen dataclass; callers must not mutate its tensor contents.

`full_reference` never calls append or its graph builder. It constructs all edges over the complete supplied prefix and invokes the original EdgeConv layers synchronously. Only its readout is limited to N. This distinguishes exact prefix execution from truncating and rebuilding a sliding window.

## Contracts and boundary choice

- New events are never sampled or silently dropped; append above the explicit limit raises.
- State=None is the explicit reset. Wrong stream, changed weights/config/device/dtype, changed autocast feature dtype, nonfinite/invalid pixels or polarity, and backward timestamps raise.
- Empty append with existing state returns the same object. First empty append returns valid empty state. Empty/expired readout is zero, including projection biases.
- Parameter updates are detected through PyTorch parameter `_version`, identity, dtype and device. Unsupported `.data` mutation can bypass PyTorch version tracking and must not be used.
- The precise cache-capacity convention is **last N accepted events, then apply time support**. Thus a query equal to the newest timestamp excludes those equal-time events, even though they occupy cache capacity. The full reference uses the identical convention. Recovering N eligible earlier events at that boundary requires a different, larger cache and should be an explicit contract change.
- State and cache tensor operations preserve autograd. The caller chooses no-grad/inference mode; optimizer updates require reset. Truncated backpropagation is not silently inserted.

## CPU smoke performed

`audit/streaming_smoke.json` records an untrained small encoder, 25 events with tied large absolute timestamps, and chunks `[1,2,7,1,14]`: incremental/reference feature disagreement stayed below the frozen 1e-5 tolerance; all twelve parameter tensors received finite gradients; gradient agreement and stale-version rejection passed. Additional bounded checks covered opposite-polarity same-time zero ages, empty identity, expiration and large common timestamp shift. These checks are implementation smoke; independent formal tests are assigned to the parent.

## Remaining risks / no performance or accuracy claim

The implementation changes old S37 token and acceptance semantics; loading its weights is initialization, not equivalence to its old predictions. Sparse SAE merge/sort, `topk`, dynamic shapes, input-validation synchronization, and all-node geometric readout/mesh work still need real GPU profiling. O(MW + MLKC²) graph arithmetic and bounded feature tensor caches do not imply a 7 ms end-to-end bound. Autograd may retain older computation graphs through cached features; its memory use must be bounded by an explicit training schedule. GPU bf16, long drift, real load, recurrent S37 adapter, initializer and legal shape prior remain outside this encoder-only smoke.
