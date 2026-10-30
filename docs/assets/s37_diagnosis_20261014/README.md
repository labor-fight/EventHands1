# S37 diagnosis: saved-output CPU recalculation

This directory adds a reproducible analysis of existing XDL predictions. It does
not contain a new training arm, model inference, FK evaluation, or modified raw
artifact. Run from the `EventHands1` repository root:

```bash
python docs/assets/s37_diagnosis_20261014/recompute_clean_bias.py
```

The script writes `clean_bias_recalculation.json` beside itself. All input paths
are repository-relative. The JSON records the script and each input SHA256:
four `bank.pt.gz` files, `packet_bank.json`, and both zgz `.meta` / `_aux.npz`
pairs. Aux files are hashed for provenance only; the computation reads GT from
`.meta` and saved 51-state predictions from the bank. It imports no model code.
No timestamp or random sampling enters the output. Exact versions are recorded;
the same inputs and versions produce identical JSON bytes.

All 24 clean packets in each sequence/run are retained, with per-packet values.
Initial states are asserted equal to GT at `previous_end_ms`; errors use GT at
the actual `end_ms`. Thus hold includes normal motion. All 144 pure translation
probes per scale and all 240 existing matched controls per mode are retained.
The repeated probes share packets and are not independent sequence replicates.

For fixed beta, wrist is `J0(beta) + transl`, so its error vector equals the
translation error vector algebraically. Subtraction is performed in float64 on
saved float32 states and then converted from meters to millimeters. In the JSON,
`b` is clean-output error, `r` is perturbed-output minus clean-output, and the
actual perturbed target error is `e = b + r`. This is a Euclidean translation
identity, not an assertion about additivity of axis-angle rotations. Positive
`improvement_over_hold_mm` means lower target error than holding the input.

The main new observation is that local depth error appears in the first update
from a correct previous state: Q1 mean signed z error is -64.66 mm, Q0 -50.93 mm,
N1 -52.40 mm, versus old C0 +8.09 mm. Their local mean wrist-equivalent errors
are 65.79, 53.75, 57.96, and 13.35 mm; the common hold error is 3.04 mm. Q1, Q0,
and N1 each improve over hold on 0/24 clean packets. N1 receives no observation
features, so event association is not necessary for this particular failure.
Q0 has no self training, so self training is also not necessary. These controls
do not isolate a unique architectural or training cause: old C0 and XDL differ
in several ways, and N1 is a separately trained model. The fixed packet bank is
not a new full-sequence score or evidence of cross-subject generalization.
