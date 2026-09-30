"""SemKine: event-driven kinematic active state estimation.

Stage-by-stage modules. The stage logs that used to live in `docs/semkine/` were retired
on 2026-09-29; the conclusions that still bind are in `docs/FAILURE_AND_CLEANUP_LEDGER.md`.
Everything here is additive: the
legacy EventHands modules under `model/` keep working unchanged, and every SemKine feature is
off unless a config asks for it.

Module map:

  protocol   S1  subject-disjoint splits, sequence statistics, split manifests
  domrand    S1  camera-consistent geometric + event-statistics randomisation
  events     S1  raw-event packet contract (ragged packing, no N_max padding)
  dataset    S1  dataset exposing `legacy_lnes` and `raw_packed` modes
  buckets    S5  motion-activity labelling and evaluation buckets
  metrics    S1  bucketed metrics, paired bootstrap
  lie        S3  SO(3)/SE(3) exp/log and the 51D <-> Lie state adapter
  encoder    S2  raw-event recurrent encoder and sparse-cell fallback
  kssf       S6  kinematic skinning semantic field, queried at event coordinates
  jacobian   S8  projected MANO Jacobian, Fisher information, Schur complement
  router     S9  root-disentangled observability router
  estimator  S10 unique-pathway per-joint head + routing mask
  gnn        S11 kinematic-tree GNN battle arm
  filter     S12 continuous-time Lie filter with heteroscedastic measurements
  anchor     S13 drift-triggered absolute anchor and tangent-space fusion
  gn         S14 active-set Gauss-Newton / LM refinement
  packetizer S15 information-driven adaptive event packetization
  frontends  S16 frontend battle (scan / cell / AEGNN-lite)
  mainline   S9+S12+S13 combined estimator
"""

__all__ = [
    "protocol",
    "domrand",
    "events",
    "dataset",
    "buckets",
    "metrics",
    "lie",
    "encoder",
    "kssf",
    "jacobian",
    "router",
    "estimator",
    "gnn",
    "filter",
    "anchor",
    "gn",
    "packetizer",
    "frontends",
    "mainline",
]
