"""Freeze the next, limited attribution study before reading new probe scores."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/s43_revisit_20261028'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'stage1_protocol.json'
    if target.exists():
        raise FileExistsError('Attribution preregistration is immutable; use a named amendment.')
    dependencies = [
        'tools/s43_rel/core.py', 'tools/s43_rel/geometry.py',
        'tools/s43_rel/model.py', 'tools/s43_rel/joint_reader.py',
        'tools/s43_rel/training.py', 'tools/s43_rel/data.py',
        'tools/s43_rel/mechanisms.py', 'tools/s43_rel/readouts_revision.py',
        'tools/s43_integration/model.py', 'model/model.py',
        'configs/s43_rel/r1/full.yaml',
        'outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz',
        'outputs/s43_rel_20261027/final_decision.json',
        'docs/S43_REL_实现训练与流式结果_20261027.md',
    ]
    protocol = dict(
        schema='s43_revisit_attribution_v1',
        registered_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User 2026-10-28: continue REL attribution, redesign, decisive experiments and implementation in stages.',
        prior_results_disclosed='REL four FT6000 arms and 54 streams complete; no stable directed-Z benefit, beta0 full RA11.095 vs paired S43 11.028; original pretrained S43 RA10.770. All known before this preregistration.',
        scope='Stage1 attribution only. Later representation choices and training require a new pre-score protocol, not another user approval.',
        immutable_old_run='outputs/s43_rel_20261027; read only. No change to frozen defaults, data, K, targets or historical verdict.',
        partition=dict(cache='outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz',
                       FIT=224, SELECT=64, SELECT_subjects=['ycy', 'ylf'],
                       note='Existing 50ms reset snippets, 18 recordings; not continuous deployment. Canonical network already trained on these subjects; SELECT is held out only from these diagnostic heads.',
                       zgz='No new reads or tuning in Stage1; historical results remain disclosed.'),
        P1=dict(name='algebraic information-chain audit',
                checks=['angular moment collision with positive support', 'm-to-f nullspace collision',
                        'per-slot Gram coupling collision under equal aggregate C and diagonal G',
                        'recoverability positive control', 'zero-evidence controls', 'parameter/storage accounting'],
                interpretation='An algebraic collision proves non-injectivity, not task information loss or a natural-data bottleneck.'),
        P2=dict(name='same-cache reader ladder',
                arms=['context_only','lnes','enhanced','a_only','production_joint_gram','slot_gram','point_factor','geometry_only'],
                features=dict(production_joint_gram='A288+f1224+C2601+diagG1224',
                              slot_gram='A288+f1224+24 symmetric Gram upper triangles (31824)',
                              point_factor='A288+f1224+a6528+lambda3072',
                              geometry_only='a6528; isolate geometry/prior leakage'),
                controlled='Same raw cache samples, previous states, K, beta, target FK, partitions and context; no geometry recomputation. Production C is full, unlike prior offline R1 diagonal-only probe.',
                normalization='Signed log or log1p as appropriate; uncentered FIT RMS floor1, clip8, semantic blocks equal norm budget; old context normalization.',
                readers=dict(rbf=dict(ridge=.01, bandwidth='FIT median distance, no SELECT search'),
                             mlp=dict(JL=512, hidden=[128,128], steps=300, lr=.001, batch=32, seed=3407)),
                metrics=['FIT/SELECT absolute and J0-aligned joint error mm', 'MSE mm2', 'each of four SELECT recordings', 'parameters and feature size'],
                evidence_rule='Slot Gram or point factor beats production in both readers on SELECT by >=max(.25mm,2%), >=3/4 recordings improve and none worsens >.5mm: evidence of task cost at this compression and budget. Otherwise unresolved, not proof of sufficiency. Higher-dimensional RBF and JL-limited MLP are explicitly not information-theoretic equivalence tests.'),
        P3=dict(name='interface, optimization and cost attribution',
                source='Frozen weights/curves/configs plus same-prefix FIT-only engineering forwards.',
                tests=['actual Jacobian/gradient reachability of sensor and typed branches at initialization and trained weights',
                       'active-mask and zero-event-gate path audit with identical observations and previous state',
                       'old curve/BN/BF16/LR/sample budget audit; no new task loss used to select settings',
                       'source-level complexity and existing measured service cost separation'],
                interpretation='A hard-zero interface derivative is an obstruction; nonzero derivatives do not prove learning or sufficient optimization. Existing training-domain fitting with held-out degradation cannot uniquely identify optimization or generalization. Any new intervention training will be separately preregistered.'),
        hardware=dict(CPU='Only Stage1; CUDA_VISIBLE_DEVICES empty', GPU='No Stage1 training'),
        dependencies={p: sha(ROOT/p) for p in dependencies},
    )
    target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(path=str(target.relative_to(ROOT)), sha256=sha(target),
                          registered_at_utc=protocol['registered_at_utc']), ensure_ascii=False))


if __name__ == '__main__':
    main()
