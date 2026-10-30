#!/usr/bin/env python3
"""Predeclare and snapshot paired XKF repair continuation before optimizer updates."""
import json
import hashlib
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/xkf/repair_20261015_r2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original = ROOT / 'outputs/xkf/debug_20261015_r1/frozen_manifest.json'
    previous = json.loads(original.read_text())
    changed = [p for p, h in previous['source_sha256'].items() if sha(ROOT / p) != h]
    if changed:
        raise RuntimeError(f'old frozen sources changed: {changed}')
    sources = set(previous['source_sha256'])
    sources.update(['tools/xkf/recovery_probe.py', 'tools/xkf/direct_probe.py',
                    'tools/xkf/diagnostics.py', 'tools/xdl/diagnostics.py',
                    'tools/xdl/make_row.py', 'semkine/eval_track.py',
                    'tools/make_s36_row.py', 'tools/report_table.py'])
    for pattern in ('semkine/xkf_repair*.py', 'tools/xkf_repair/*.py',
                    'tests/test_xkf_repair*.py', 'configs/xkf_repair/*.yaml'):
        sources.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
    hashes = {p: sha(ROOT / p) for p in sorted(sources)}
    lineage = {}
    for seed in (3407, 3408, 3409):
        run = ROOT / f'outputs/xkf/debug_20261015_r1/xkf_s{seed}'
        lineage[str(seed)] = {label: {'path': str(path), 'sha256': sha(path)} for label, path in (
            ('checkpoint', run / f'xkf_s{seed}-step=2000.ckpt'),
            ('actual_random_initial', run / 'initial_state.ckpt'))}
    protocol = {
        'study': 'XKF_REPAIR_R2_PAIRED_DEBUG', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User asked to repair now; priorGPU2-7/debug-before-full authorization continues',
        'formal': False, 'automatic_full_training': False,
        'lineage': lineage, 'allocation': {'3407': [2, 3], '3408': [4, 5], '3409': [6, 7]},
        'paired_arms': {
            'control': 'Original objective/inference; all repair flags false and auxiliaries zero',
            'repair': 'FP32 state addition; true augmented GT endpoint motion target independent of incoming error; '
                      'two training-only DGP auxiliaries with GT geometry/2D separately, each0.5*LAMBDA_T'},
        'common_continuation': 'Full resume each ownseed2000 weights, Adam states, scheduler/global/data clocks and gain moments; '
                               'explicitpostrestore LR0.0004 for BOTH arms; constant schedule; bf16;2ranksx512;noaccumulation',
        'sampling': 'Canonical9identities72sequences and named counter streams unchanged; motionGT onlytrainingdict, '
                    'all main/prefix eventpacket bytes unchanged; same depth augmentation/window law',
        'first_endpoint': 2200, 'checkpoint_selection': 'Allthree seeds at same endpoint, no bestseed or perseed checkpoint selection',
        'continuation_rule': 'At2200 evaluate both pairedarms; decide fromallseed evidence whether another matchedshortstage is warranted. '
                             'No implicit6000 launch; retain every stage even if effects fail.',
        'evaluation': ['Unchanged originalreadiness formulas/thresholds with actualsavedrandominitial reevaluated under matchingconfig',
                       'Fixed112clip9condition6mode allobserved andfullhorizon diagnosis, clean andsignedperturbations',
                       'Full2590 Dmeasurement decomposition plus128/seq trainingidentitycontrols',
                       'Full2590 standardrecursive row withdistinctnonformalname andexactstep'],
        'effect_assessment': 'Compare repair to SAME-LR control and both toold2000. '
                             'Softwarecorrectness alone doesnot establishlearningeffect or fullreadiness.',
        'additional_full_readiness_safeguards': {
            'allthree_original_debug_readiness': True,
            'eachseed_localbank_clean_fused_abs_not_worse_than_hold': True,
            'eachseed_localcold_fullhorizon_clean_fused_abs_not_worse_than_hold': True,
            'eachseed_globalbank_clean_fused_abs_le_control_times': 1.1},
        'known_limits': ['Approximate depthaugmentation geometricresidual remains, not changed simultaneously',
                         'Sparse actuallocal evidence is underrepresented bycurrenteventaugmentation; no claim this round fixesit',
                         'OriginalStage0F1 failure andformalG3 shortclipunassessability retained',
                         'Bundledrepair tests intended contract restoration, not separablecausal attribution amongthreechanges'],
        'source_sha256': hashes}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    manifest = {'protocol_sha256': sha(path), 'source_sha256': hashes}
    (OUT / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for source in sources:
        dst = OUT / 'source_snapshot' / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, dst)
    print(json.dumps({'output': str(path), 'frozen_files': len(sources)}))


if __name__ == '__main__':
    main()
