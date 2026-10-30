# Existing hand_data51 adapter

`data.py` opens the unique existing `splits_semkine.json`, event arrays,
microsecond remainders, offsets, auxiliary calibration and 51D pose files in
read-only mode. It never writes to the data root. No EvRealHands stream is read.

The default small sample comprises `ch`, `lfz`, `lpc`, `lr`, `ycy`, `ylf`, and
`zgz`, each in local/global: at most eight windows per recording. The first four
subjects are a subset of FIT7, `ycy`/`ylf` are SELECT, and `zgz` remains
development/report-only, never an independent sealed test set. Main windows
are 50 ms long, at least 100 ms apart; sampling uses only valid-run boundaries
and a fixed evenly spaced index rule. Each current window starts at least
150 ms after the valid run starts.

The default prior state comes from **two frozen S43 W50 predictions using only
the preceding 100 ms of events**. The first prediction starts from the public
constant 51D pose `[0,0,.6,0,...,0]`. The two predictions are returned so that
the same causal velocity proposal can be shared by all representation arms.
This deliberately avoids using old `evalx` traces as deployable inputs: those
traces were initialized with ground truth at each valid-run start. A
`public_constant` mode provides a weaker no-learning control; it does not
represent a diagnostic upper bound.

Only the returned `target` uses pose labels. It is for training/scoring and must
not be used to construct a candidate, crop, texture or representation. The
sequence beta and K remain external conditions, matching S43's original
protocol; event-only beta estimation is not claimed.

The returned native events have columns `[relative_us,x,y,stored_polarity]`.
The half-open interval is `[start_us,cutoff_us)`. Real same-time multiple
triggers retain distinct source IDs (their absolute event-row indices).
Stored 0/1 polarity does **not** assert a verified physical ON/OFF mapping.
The target at `end_ms` is 1 ms earlier than the complete event cutoff
`(end_ms+1)*1000`; this is explicitly a fixed-lag retrospective target, not
zero-lookahead output at the target time.

Native events occupy 240×180 pixels. The auxiliary K retains the original
camera scale. Multiply its first two rows by `.375` to obtain the event-grid
intrinsics, then scale x/y to any common reduced resolution. Reduced intrinsics
use `(cx+.5)*scale-.5`, and source event cells use
`floor((x+.5)*scale)`, matching `geometry.map_pixels`. The fixed spatial
reduction uses no label-dependent crop.

These are existing hand_data51 recordings with the documented v2e provenance
limitations. This prototype does **not** call them a separate real sensor test.

Metadata-only command (does not evaluate a model or target):

```bash
CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python \
  -m tools.s43_br.data --metadata-only
```

Use `/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python` when loading the
frozen S43 model; the default Python interpreter in this workspace does not
have torchvision. Small sample generation is invoked by the preregistered
prototype runner, not automatically at import time.
