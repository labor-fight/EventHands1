# XDE fast diagnostic artifacts

The complete findings are in [S37统一诊断：XDE快实验结果](../../EXPERIMENT_RECORDS.md#xde-results).
Only physical GPUs 2–7 were authorized and used for this study. All six study
devices were free after the workers completed. Production model code, historical
checkpoints, the canonical data split and S39 were unchanged.

## Scope and outputs

- E1: [protocol](../../../outputs/xde/e1/protocol.json),
  [audit](../../../outputs/xde/e1/final_audit.json),
  [commands and details](../../../outputs/xde/e1/README.md).
- E2: [protocol](../../../outputs/xde/e2/protocol.json),
  [report](../../../outputs/xde/e2/REPORT.md),
  [independent audit](../../../outputs/xde/e2/independent_review_xa.json),
  [rollout AUC, motion, roughness and denominators](../../../outputs/xde/e2/supplemental_audit.json).
- E3: [frozen geometry protocol](../../../outputs/xde/e3_preflight/protocol.json),
  [decision](../../../outputs/xde/e3_preflight/result.json),
  [corrected wrist-depth coverage](../../../outputs/xde/e3_preflight/actual_wrist_depth_coverage.json).
  The registered candidate gate failed; no full model training was launched.
- Software pipeline: [protocol](../../../outputs/xde/pipeline/protocol.json),
  [timing](../../../outputs/xde/pipeline/result.json),
  [timestamp-derived event age](../../../outputs/xde/pipeline/event_age.json).
- [Execution summary and measured phase costs](../../../outputs/xde/execution_summary.json).
- [PNG](fast_evidence.png), [SVG](fast_evidence.svg),
  [figure provenance](figure_manifest.json). Figure axes and captions separate
  whole-sequence clean counterfactuals from small-head conditional probes.

The E2 heads are diagnostic artifacts, not accepted replacement models. The
training-check split was used to select each head epoch. All head seeds, widths,
wrong-content controls and failed rollouts are retained. None of these is a new
official full-sequence main row or a cross-subject SOTA claim.

Recorded worker-phase cost totals **0.24993 GPU hours**. A separate, unmeasured
0.15 GPU-hour startup/shutdown budget reserve brings conservative accounting to
0.39993 GPU hours. This is not utilization-integrated billing; phase timers do not
all begin at process launch. The reserve is explicitly a provision, not an
observed cost. No full-model run consumed the originally proposed E3 budget.

## Execution commands

Run from the repository root with the existing environment
`/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python` (PyTorch 2.1.0).
These are the original task entry points; preparation commands freeze protocols
and intentionally refuse to overwrite an incompatible existing protocol.
Raw outputs and JSON files include source and checkpoint hashes.

```bash
CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e1_diagnostics.py prepare
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e1_diagnostics.py worker --group q1
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e1_diagnostics.py worker --group other
CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e1_diagnostics.py cpu

CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py prepare
CUDA_VISIBLE_DEVICES=4 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py extract --shard 0
CUDA_VISIBLE_DEVICES=5 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py extract --shard 1
CUDA_VISIBLE_DEVICES=4 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py fit --seed 1701
CUDA_VISIBLE_DEVICES=5 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py fit --seed 1702
CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py summarize
CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py prepare-rollout
CUDA_VISIBLE_DEVICES=4 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py rollout --sequence zgz_local
CUDA_VISIBLE_DEVICES=5 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py rollout --sequence zgz_global
CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e2_probes.py supplemental

CUDA_VISIBLE_DEVICES=6 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e3_geometry.py --freeze
CUDA_VISIBLE_DEVICES=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/e3_geometry.py --run --device cuda

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/profile_pipeline.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/pipeline_event_age.py
CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xde/plot_fast_results.py
CUDA_VISIBLE_DEVICES='' python tools/xde/finalize_fast.py
```

The E2 report records the additional frozen rollout protocol, actual rollout
commands and CPU supplemental analysis. E1 final auditing and E3 corrected
coverage scripts are preserved beside their outputs, without changing their
original inference or geometry contracts.

Pipeline numbers include synchronized software stages and host-ready mesh output.
They exclude sensor acquisition, real-time queueing, display, sequence opening,
segment initialization and extra audit copies. Event age is computed from
timestamps under zero transport/queue delay, not measured hardware latency.
