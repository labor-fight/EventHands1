"""Explicit new-protocol release gate; no existing single readiness can open it.

The release and acceptance bundle must be produced after all three current
endpoints have complete actual evaluation. This module never creates a release.
Every evidence file is hash-bound; the new source/config manifest is separately
required. The authority is the root execution review, not a training callback.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SEEDS = (3407, 3408, 3409)
REQUIRED_CHECKS = (
    'original_readiness_all_three', 'actual112_all_three', 'full2590_all_three',
    'clean_and_signed_recovery_all_three', 'global_high_motion_protected_all_three',
    'paired_training_audit_all_six', 'train_motion_evidence_reviewed',
    'joint_implementation_verified',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(entry):
    path = Path(entry['path']).resolve()
    require(path.is_file() and sha(path) == entry['sha256'], 'release evidence missing or changed: ' + str(path))
    return path, json.loads(path.read_text())


def identity(value, endpoints, seed):
    require(value.get('seed') == seed and value.get('global_step') == 2600,
            'evidence has wrong seed/global endpoint')
    require(value.get('checkpoint_sha256') == endpoints[str(seed)]['sha256']
            and Path(value['checkpoint']).resolve() == Path(endpoints[str(seed)]['path']).resolve(),
            'evidence is not from the accepted current parent')


def assessed(value, kind):
    require(value.get('kind') == kind and value.get('status') == 'ASSESSED'
            and not value.get('validation_errors') and not value.get('missing'),
            'missing, invalid or wrong typed independent evidence: ' + kind)


def actual_assessment(value, endpoints):
    assessed(value, 'INDEPENDENT_FIXED_I_CONTINUATION_ASSESSMENT')
    require(set(value.get('arms', {})) == {'parent', 'raw', 'aug'}, 'actual112 needs all nine actual runs')
    for arm in ('parent', 'raw', 'aug'):
        require(set(value['arms'][arm]) == {str(s) for s in SEEDS}, 'actual112 has missing seeds')
    for seed in SEEDS:
        row = value['arms']['raw'][str(seed)]
        identity(row['identity'], endpoints, seed)
        require(row.get('full_raw_geometry_and_gate_replay') is True,
                'actual112 independent raw replay missing')
    return value['arms']


def validate_evidence(check, entries, endpoints):
    """Check actual producer schemas, not acceptance booleans or an empty JSON."""
    require(isinstance(entries, list) and bool(entries), 'empty typed evidence: ' + check)
    root = Path(__file__).resolve().parents[2]
    bound = {}
    def retain(path, digest):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        require(path.is_file() and sha(path) == digest, 'typed evidence dependency changed: ' + str(path))
        bound[str(path)] = digest
    def retained_reference(entry):
        path, value = reference(entry)
        retain(path, entry['sha256'])
        for field in ('source_sha256', 'input_sha256'):
            for name, digest in value.get(field, {}).items():
                retain(name, digest)
        return value
    values = [retained_reference(entry) for entry in entries]
    if check == 'original_readiness_all_three':
        require(len(values) == 3, 'need three original readiness artifacts')
        seen = set()
        for value in values:
            require(value.get('kind') == 'XKF_IONLY_ORIGINAL_READINESS_ACTUAL_FUSION', 'wrong original-readiness evidence kind')
            seed = value['identity']['seed']; require(seed in SEEDS and seed not in seen, 'duplicate/wrong readiness seed')
            seen.add(seed); identity(value['identity'], endpoints, seed)
            readiness = value.get('readiness', {})
            require(readiness.get('status') == 'PASS' and readiness.get('ready_for_full_training') is True
                    and bool(readiness.get('checks')) and all(v is True for v in readiness['checks'].values()),
                    'an original readiness check failed')
            current = retained_reference(value['artifacts']['current.json'])
            initial = retained_reference(value['artifacts']['initial.json'])
            require(current.get('readiness') == readiness and current.get('identity') == value.get('identity')
                    and initial.get('identity', {}).get('global_step') == 0,
                    'original-readiness current/initial evidence differs from summary')
            from tools.xkf.debug_eval import readiness as original_readiness
            require(original_readiness(current, initial) == readiness,
                    'original readiness does not reproduce with the complete frozen formulas')
    elif check in ('actual112_all_three', 'clean_and_signed_recovery_all_three'):
        require(len(values) == 1, 'one complete all-nine actual112 assessment required')
        arms = actual_assessment(values[0], endpoints)
        if check == 'clean_and_signed_recovery_all_three':
            for seed in SEEDS:
                row = arms['raw'][str(seed)]
                require(row.get('clean_pass') is True and row.get('directional_pass') is True,
                        'clean/signed recovery failed')
                loops = row.get('original_candidate_loop_checks', {})
                require(set(loops) == {'zgz_local', 'zgz_global'} and all(
                    v.get('mean_abs_pass') is True and v.get('endpoint_wrist_pass') is True for v in loops.values()),
                    'original combined loop check failed')
    elif check == 'full2590_all_three':
        require(len(values) == 3, 'need three actual full2590 rows')
        seen = set()
        for value in values:
            require(value.get('numeric_status') == 'PASS' and value.get('finite_subset_reduction') is False,
                    'invalid or filtered full row')
            protocol = value.get('protocol', {})
            require(protocol.get('exact_optimizer_step') == 2600 and protocol.get('stability_mode') == 'per_sample_model',
                    'fullrow is not the actual stable current endpoint')
            identity(value['provenance'], endpoints, value['provenance']['seed'])
            seed = value['provenance']['seed']; require(seed in SEEDS and seed not in seen, 'duplicate fullrow seed'); seen.add(seed)
            row = value.get('per_seed', {}).get(str(seed), {})
            require(row.get('n_frames') == 2590 and set(row.get('sequences', [])) == {'zgz_local', 'zgz_global'},
                    'full2590 denominator/sequence evidence missing')
    elif check == 'paired_training_audit_all_six':
        require(len(values) == 1, 'one complete six-endpoint training audit required')
        value = values[0]
        require(value.get('kind') == 'INDEPENDENT_IONLY_PAIRED200_TRAINING_AUDIT'
                and value.get('status') == 'PASS' and not value.get('errors') and not value.get('missing'),
                'complete paired training audit missing')
        require(set(value.get('seeds', {})) == {str(s) for s in SEEDS}, 'paired training seeds incomplete')
        for seed in SEEDS:
            row = value['seeds'][str(seed)]
            require(set(row) == {'raw', 'aug', 'pair'} and all(v.get('status') == 'PASS' for v in row.values())
                    and row['pair'].get('paired_rows') == 204800 and row['pair'].get('independently_reconstructed_rows') == 204800,
                    'complete paired sample audit missing')
            require(row['raw']['checkpoint_sha256'] == endpoints[str(seed)]['sha256'], 'paired audit current endpoint differs')
    elif check == 'train_motion_evidence_reviewed':
        require(len(values) == 1, 'one complete paired training-motion assessment required')
        value = values[0]; assessed(value, 'INDEPENDENT_CANONICAL_I_CONTINUATION_MOTION_COMPARISON')
        expected = {f'{arm}/{seed}' for arm in ('parent', 'raw', 'aug') for seed in SEEDS}
        require(set(value.get('verification', {})) == expected and bool(value.get('paired_comparisons')),
                'training motion must include all nine runs and actual comparisons')
        for seed in SEEDS:
            identity(value['identities'][f'raw/{seed}'], endpoints, seed)
        require(all(v.get('all2304x10_complete') is True and v.get('main_eligible') == 2016 and v.get('cold') == 288
                    and v.get('D_calls') == 0 and v.get('I_calls') == 23040 for v in value['verification'].values()),
                'training motion conditions/domain incomplete')
    elif check == 'global_high_motion_protected_all_three':
        # The old prereg is descriptive here. Require an explicit new review,
        # never silently manufacture a numeric threshold from its results.
        require(len(values) == 1, 'one explicit high-motion review required')
        value = values[0]
        require(value.get('kind') == 'XKF_I_CONTINUATION_HIGH_MOTION_REVIEW_V1'
                and value.get('status') == 'PASS' and value.get('seeds') == list(SEEDS)
                and isinstance(value.get('criteria_and_rationale'), str) and len(value['criteria_and_rationale'].strip()) > 20,
                'new explicit high-motion review absent')
        actual = retained_reference(value['actual_assessment']); arms = actual_assessment(actual, endpoints)
        require(set(value.get('cells', {})) == {str(s) for s in SEEDS}, 'high-motion review seeds incomplete')
        for seed in SEEDS:
            cell = value['cells'][str(seed)]
            require(cell.get('accepted') is True, 'high-motion review rejects a seed')
            for arm in ('parent', 'raw', 'aug'):
                require(cell.get(arm) == arms[arm][str(seed)]['global_high_motion'], 'high-motion review differs from actual data')
    elif check == 'joint_implementation_verified':
        require(len(values) == 1, 'one independent joint implementation CPU audit required')
        value = values[0]
        require(value.get('kind') == 'INDEPENDENT_XKF_JOINT_IMPLEMENTATION_AUDIT_V1'
                and value.get('status') == 'PASS' and not value.get('errors') and not value.get('missing')
                and value.get('independent_source_review') is True, 'joint implementation review missing')
        cpu = retained_reference(value['cpu_validation'])
        require(cpu.get('kind') == 'XKF_JOINT_CPU_IMPLEMENTATION_VALIDATION_V1'
                and cpu.get('status') == 'PASS' and cpu.get('device') == 'cpu'
                and set(cpu.get('parents', {})) == {str(s) for s in SEEDS}, 'real CPU validation missing')
        for seed in SEEDS:
            row = cpu['parents'][str(seed)]
            require(row.get('checkpoint_sha256') == endpoints[str(seed)]['sha256']
                    and row.get('global_step') == 2600 and row.get('i_adam_step') == row.get('d_adam_step') == 2400
                    and row.get('named_Adam_slots') == 96 and row.get('strict_model_restore') is True,
                    'CPU joint parent/96 Adam restore differs')
        step = cpu.get('actual_step', {})
        require(step.get('seed') in SEEDS and step.get('precision') == 'bf16'
                and step.get('all_96_gradients_finite') is True and step.get('both_branches_nonzero_gradients') is True
                and step.get('all_96_Adam_steps_before') == 2400 and step.get('all_96_Adam_steps_after') == 2401
                and step.get('all_65_named_buffers_bitwise_unchanged') is True
                and step.get('direct_call_grad_enabled') == [False, False, False, True]
                and step.get('actual_self_rows') == 1 and step.get('actual_burn_rows') == 3
                and step.get('loss_finite') is True and step.get('no_deployment_adapter_in_training') is True,
                'actual joint CPU step/Adam/buffers/original burn validation incomplete')
        import xml.etree.ElementTree as ET
        test_ref = value['pytest_xml']; retain(test_ref['path'], test_ref['sha256'])
        suites = ET.parse(test_ref['path']).getroot().findall('testsuite')
        require(bool(suites) and sum(int(s.get('tests', '0')) for s in suites) >= 14
                and all(int(s.get(k, '0')) == 0 for s in suites for k in ('errors', 'failures', 'skipped')),
                'complete joint CPU test suite did not pass')
    else:
        raise ValueError('unknown release check')
    return bound


def validate_release(release_path, manifest_path, root, config, parent, seed):
    require(release_path is not None and manifest_path is not None,
            'full6200 is blocked: a new three-seed acceptance release and manifest are required')
    root = Path(root).resolve()
    release_path, manifest_path = Path(release_path).resolve(), Path(manifest_path).resolve()
    release = json.loads(release_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    require(release.get('kind') == 'XKF_JOINT_GLOBAL6200_EXECUTION_RELEASE_V1'
            and release.get('status') == 'RELEASED'
            and release.get('seeds') == list(SEEDS), 'not the new complete three-seed joint release')
    require(release.get('origin_global_step') == 2600 and release.get('origin_I_D_steps') == [2400, 2400]
            and release.get('target_global_step') == 6200 and release.get('target_I_D_steps') == [6000, 6000]
            and release.get('original_global6000_protocol') is False, 'release must declare actual mixed-history clocks')
    require(release.get('manifest_sha256') == sha(manifest_path), 'new manifest differs from release')
    require(manifest.get('kind') == 'XKF_JOINT_6200_FROZEN_MANIFEST_V1', 'an old manifest cannot release joint training')
    sources = manifest.get('source_sha256', {})
    for name in ('tools/xkf_joint/model.py', 'tools/xkf_joint/train.py', 'tools/xkf_joint/release.py'):
        require(name in sources, 'new entry missing from release manifest: ' + name)
    require(bool(sources), 'empty release manifest')
    for name, digest in sources.items():
        path = Path(name) if Path(name).is_absolute() else root / name
        require(path.is_file() and sha(path) == digest, 'released source changed: ' + name)
    require(set(manifest.get('configs', {})) == {str(s) for s in SEEDS}, 'new manifest must bind three full configs')
    for entry in manifest['configs'].values():
        require(Path(entry['path']).is_file() and sha(entry['path']) == entry['sha256'], 'released config changed')
    config_ref = manifest.get('configs', {}).get(str(seed))
    require(config_ref and Path(config_ref['path']).resolve() == Path(config).resolve()
            and config_ref['sha256'] == sha(config), 'current full config is not in new manifest')
    acceptance_path, acceptance = reference(release['acceptance'])
    require(acceptance.get('kind') == 'XKF_JOINT_THREE_SEED_COMPLETE_ACCEPTANCE_V1'
            and acceptance.get('status') == 'PASS' and acceptance.get('seeds') == list(SEEDS),
            'missing complete three-seed actual acceptance bundle')
    checks = acceptance.get('checks', {})
    require(set(checks) == set(REQUIRED_CHECKS) and all(checks.get(key) is True for key in REQUIRED_CHECKS),
            'a single readiness or incomplete debug evidence cannot release joint full training')
    endpoints = acceptance.get('endpoints', {})
    require(set(endpoints) == {str(s) for s in SEEDS}, 'acceptance must bind all three exact parents')
    for key, entry in endpoints.items():
        path = Path(entry['path']).resolve()
        require(path.is_file() and sha(path) == entry['sha256'], 'accepted parent changed: ' + key)
        require(entry.get('global_step') == 2600 and entry.get('i_adam_step') == entry.get('d_adam_step') == 2400,
                'accepted parent clocks differ: ' + key)
    require(Path(endpoints[str(seed)]['path']).resolve() == Path(parent).resolve()
            and endpoints[str(seed)]['sha256'] == sha(parent), 'current parent was not accepted')
    evidence = acceptance.get('evidence', {})
    require(set(evidence) == set(REQUIRED_CHECKS), 'every acceptance check requires actual evidence')
    hashes = {str(acceptance_path): release['acceptance']['sha256']}
    for key, entries in evidence.items():
        hashes.update(validate_evidence(key, entries, endpoints))
        for entry in entries:
            path, value = reference(entry)
            hashes[str(path)] = sha(path)
    return dict(release=str(release_path), release_sha256=sha(release_path),
                manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                manifest_source_sha256=sources,
                acceptance_sha256=release['acceptance']['sha256'], evidence_sha256=hashes,
                scope='New continuation protocol; global6200/I6000/D6000, never original global6000.')
