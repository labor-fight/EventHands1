#!/usr/bin/env python3
"""CPU-only completeness/provenance receipt for the completed XDE fast phase."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/xde'
INPUTS = {}


def read(path):
    p = OUT / path
    INPUTS[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def main():
    e1 = read('e1/final_audit.json')
    assert all(x['n'] == 1204 and x['all_predictions_finite'] and x['failures'] == 0
               and x['old_24_clean_replay_max_abs_state_difference'] == 0 for x in e1['clean'].values())
    assert len(e1['clean']) == 4
    assert all(x['pairs'] == 480 and x['all_predictions_finite']
               and x['factual_replay_max_abs_state_difference'] == 0 for x in e1['own'].values())
    e2 = read('e2/supplemental_audit.json')
    assert e2['audit']['all_finite'] and e2['audit']['failures'] == 0 and e2['audit']['calls'] == 13440
    cache = read('e2/cache_audit.json')
    assert cache['rows'] == 7680 and cache['all_finite']
    for seed in (1701, 1702):
        fitted = read(f'e2/fit_s{seed}.json')
        assert len(fitted['results']) == 10
    gates = read('e2/verdict.json')
    assert len(gates['gates']) == 4
    e3 = read('e3_preflight/result.json')
    assert e3['status'] == 'FAIL' and e3['projection_gate'] == 'PASS' and e3['candidate_gate'] == 'FAIL'
    assert not e3['training_launched']
    read('e3_preflight/actual_wrist_depth_coverage.json')
    pipeline = read('pipeline/result.json')
    smoke = read('pipeline_smoke/result.json')
    assert pipeline['actual_queries'] == pipeline['expected_queries'] == 2590
    assert pipeline['full_denominator']
    read('pipeline/event_age.json')
    costs = {'E1_workers': e1['budget']['reported_worker_gpu_hours'],
             'E2_extract_fit_rollout_workers': e2['budget']['allocated_gpu_hours'],
             'E3_geometry_preflight': e3['total_seconds']/3600,
             'pipeline_full': pipeline['gpu_hours_allocated'],
             'pipeline_smoke': smoke['gpu_hours_allocated']}
    # Timing clocks differ slightly in their start point; do not pretend this is
    # exact process-launch-to-exit billing. Reserve is explicitly not measured.
    reserve = .15
    receipt = {'study': 'XDE-FAST-20261014', 'status': 'completed_fast_phase',
        'physical_gpus_used': [2, 3, 4, 5, 6, 7],
        'full_encoder_training_runs': 0, 'diagnostic_readout_heads_fit': 20,
        'E1_full_local_clean_forwards': 4*1204, 'E1_own_paired_queries_per_sequence': 480,
        'E2_extraction_forwards': 15360, 'E2_rollout_forwards': 13440,
        'pipeline_full_queries': 2590,
        'E3_disposition': 'not launched: registered geometry candidate gate failed',
        'production_or_default_model_changed': False,
        'measured_phase_gpu_hours': costs, 'measured_phase_sum_gpu_hours': sum(costs.values()),
        'unmeasured_startup_shutdown_reserve_gpu_hours': reserve,
        'sum_with_budget_reserve_gpu_hours': sum(costs.values())+reserve,
        'cost_limit': 'phase wall time times assigned cards; startup reserve is a budget provision, not a measured charge; not GPU utilization integration',
        'input_sha256': INPUTS,
        'current_tool_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((ROOT/'tools/xde').glob('*.py'))},
        'interpretation': 'No deployment adoption. E1 bias established; E2 conditional information survives bank compression but recovery fails; E3 depth hypothesis remains untested by matched retraining.'}
    (OUT/'execution_summary.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': receipt['status'], 'measured_phase_gpu_hours': sum(costs.values()),
                      'sum_with_budget_reserve_gpu_hours': sum(costs.values())+reserve}, ensure_ascii=False))


if __name__ == '__main__':
    main()
