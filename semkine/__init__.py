"""SemKine: event-driven kinematic active state estimation.

The stage logs that used to live in `docs/semkine/` were retired on 2026-09-29; the conclusions that still bind
are in `docs/EXPERIMENT_RECORDS.md#ledger-dense`. Everything here is additive: the legacy EventHands modules
under `model/` keep working unchanged, and every SemKine feature is off unless a config asks for it.

The raw-event encoder family (S2 / S16 / S36-S38 heads), the S37 routed / mesh-query / FK-graph / mesh-graph
readouts, the XYZ-C1 association / residual / AURA / AGG778 / MRF variants and the SO(3) Lie helpers were removed on
2026-11-01 (core cleanup); their studies are closed and recorded in `docs/EXPERIMENT_RECORDS.md`.

Module map (the modules `semkine/s45*.py` and `c1sf_zrel.py` are the opt-in hooks of the S45-line studies):

  dataset    S1  fixed splits_semkine.json; nine training subjects, zgz evaluation
  domrand    S1  camera-consistent geometric + event-statistics randomisation
  events     S1  raw-event packet contract (ragged packing, no N_max padding)
  buckets    S5  motion-activity labelling and evaluation buckets
  metrics    S1  bucketed metrics, paired bootstrap
  encoder    S2  the seven-scalar event token (and its SAE / inter-event helpers)
  event_gnn  S36 event-graph (EdgeConv) encoder used by the S45-TR T0 / T1a hooks
  anchored   S13 anchored / filtered trackers and the deployed count-gain filter (S39 line)
  xyz_mesh   S37 XYZ-C1: prev mesh in camera XYZ, ray association, MeshGNN, common readout (S45)
  xag        XAG source-label routing (part / face route) of the XYZ association
  xag_sources XAG pseudo-source cache reader
  eval_track S1  extended recursive evaluator
  (train     S1  training entry point, `python -m semkine.train`; not imported with the package)
"""

__all__ = [
    "anchored",
    "buckets",
    "dataset",
    "domrand",
    "encoder",
    "eval_track",
    "event_gnn",
    "events",
    "metrics",
    "xag",
    "xag_sources",
    "xyz_mesh",
]
