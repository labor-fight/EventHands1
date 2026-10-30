#!/usr/bin/env python3
"""Freeze the XKF pre-registration (outputs/xkf/protocol.json) and the training manifest before any training."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'outputs/xkf'
TRAIN = ('tools/xdl/data.py', 'tools/xdl/train.py', 'semkine/xyz_observation.py', 'semkine/xyz_recovery_study.py',
         'model/model.py', 'semkine/xyz_mesh.py', 'semkine/event_gnn.py', 'semkine/dataset.py', 'semkine/events.py',
         'semkine/domrand.py', 'semkine/encoder.py', 'semkine/train.py', 'model/mano_layer.py', 'model/config.py',
         'model/backbones.py', 'semkine/lie.py', 'semkine/xkf.py', 'semkine/xkf_direct.py', 'tools/xkf/data.py',
         'tools/xkf/train.py', 'tools/xkf/make_configs.py', 'configs/xkf/xkf_s3407.yaml', 'configs/xkf/xkf_s3408.yaml',
         'data/hand_data51/splits_semkine.json', 'tests/test_xkf.py')


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


if __name__ == '__main__':
    path = OUT / 'protocol.json'
    if path.exists():
        raise SystemExit('protocol already frozen')
    stage0 = json.loads((OUT / 'stage0/result.json').read_text())
    p = {
        'study': 'XKF-direct-measurement-plus-XYZ-motion-fusion-20261014', 'created_unix': time.time(),
        'frozen_before_any_training': True,
        'authorization': 'user 2026-10-14: GPU4-7 four-card training, all CPU; agent sandbox has no GPU, the user launches',
        'design': {'direct': 'prev-free ResNet18-l3 on long-window LNES + camera planes + log W; root 6D, 45 finger AA, '
                             '21 keypoint heatmaps (1/4 res soft-argmax); translation by HandDGP closed-form solve',
                   'motion': 'unchanged S37 XYZ path (stride_sample, associate_rays, XYZMeshNet, bank) with state96; '
                             'trained with small prev noise (5 mm / 0.05 rad gaussian), XDL K4 self burn',
                   'fusion': 'detached inputs; per-group gains K=(V_I-C)/(V_I+V_D-2C) from EMA(0.99) residual moments '
                             'on self rows (all rows before the self curriculum), all-reduced; root on SO(3) with a 60 deg '
                             'innovation clip; empty W50 packet holds prev',
                   'loss': 'log10(L_I) + log10(L_D)', 'papers': ['E-3DPSM CVPR 2026 arXiv 2604.08543',
                                                                'HandDGP ECCV 2024 arXiv 2407.15844', 'CLIFF ECCV 2022']},
        'arms': {'xkf_s3407': {'gpus': [4, 5], 'master_port': 29731}, 'xkf_s3408': {'gpus': [6, 7], 'master_port': 29732}},
        'training': '2 ranks x 512, bf16, Adam 4e-3 constant after 500 warmup, exact step 6000, fresh init, no selection; '
                    'long window log-uniform 30-300 ms clipped to run start; DOMRAND SCALE_MODE depth',
        'evaluation': 'tools/xkf/make_row.py (val_core zgz, rawW50 packets, rng0 init noise, direct window W200 cut at run '
                      'start) for kmode fused / i / d / d_zero; tools/xkf/diagnostics.py on CPU for the gates',
        'stage0': {'F1_dgp': 'zgz_local depth RMSE 10.8 / 30.1 / 52.3 mm at 1 / 2 / 3 px keypoint noise (target <= 20 mm at '
                             '2 px: FAIL at local, PASS at global and training-identity depths)',
                   'F2_motion_vs_hold': 'existing XYZ branches from GT prev on check packets are worse than holding: wrist '
                                        '9.0-9.3 vs 5.5 mm, root 4.0-4.9 vs 2.3 deg, fingers equal',
                   'raw': 'outputs/xkf/stage0/result.json', 'raw_sha256': hashlib.sha256((OUT / 'stage0/result.json').read_bytes()).hexdigest()},
        'pre_registered_expectation': 'if the retrained motion branch stays worse than holding, the fused tracker reduces to '
                                      'the direct measurement plus a zero-motion smoother and G5 fails; the claim then rests '
                                      'on G1-G4 (an independent event measurement that removes prior-driven updates)',
        'gates': {
            'G0': 'direct output bit-identical under any prev and zero gradient to prev; empty packet returns prev exactly; '
                  'XDL default path bit-identical (tests/test_xkf.py)',
            'G1': 'direct only (kmode d), GT prev, all zgz packets: OLS slope of D wrist depth on GT wrist depth in '
                  '[0.8, 1.2]; zgz_local |mean signed z| <= 15 mm; zgz_local mean root geodesic <= 10 deg; with the W50 '
                  'events and long-window LNES of a packet >= 1 s away in the same sequence, mean wrist error >= 2x',
            'G2': 'fused, zgz_local, one step from GT prev with prev z +/-50 mm and 0: OLS slope of output depth on input '
                  'depth <= 0.6; prev root rotated +/-20 deg about camera x and y: median ratio of one-step root error '
                  'to the injected 20 deg <= 0.5',
            'G3': 'fused, start of every zgz_local run (41) and 24 fixed zgz_global clip starts, initial state GT with '
                  'z +/-50 mm or root +/-20 deg about camera x / y (6 perturbations): recovered = within 10 steps, '
                  'wrist error <= 2x + 1 mm and root error <= 2x + 1 deg of the same clip started from GT, for 3 '
                  'consecutive steps; pass if >= 90% of trials recover',
            'G4': 'fused, GT prev, zgz_local: mean one-step root geodesic <= 4 deg; main-row zgz_local '
                  'jitter_all_mm_per_step <= 2 x gt_move_mm_per_step',
            'G5': 'main-row overall RA of fused better than each of kmode i, d, d_zero by >= 2 mm',
            'G6': 'main table with Q1, q1r, q1dz, S39; the problem counts as solved only if G1-G4 pass for both seeds'},
        'not_claimed': 'no S39 replacement; two seeds are not a confidence interval; zgz is one subject; no '
                       'checkpoint or gain selection on zgz',
        'source_sha256': {f: sha(f) for f in TRAIN}}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, indent=1, ensure_ascii=False) + '\n')
    manifest = {'status': 'frozen_for_xkf_two_seed_training', 'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_sha256': {f: sha(f) for f in TRAIN}}
    (OUT / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'protocol': str(path), 'manifest': str(OUT / 'frozen_manifest.json'), 'files': len(TRAIN)}))
