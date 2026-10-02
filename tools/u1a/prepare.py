#!/usr/bin/env python3
"""Prepare matched frozen-S38 U1a initial states and configs before results."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import torch
import yaml

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R / 'model')]
from model import MNISTModel
from semkine.u1a_readout import copy_shared_to_untied
from semkine.dataset import sequences_for_split

O = R / 'outputs/u1a'
C = R / 'configs/u1a'
O.mkdir(parents=True, exist_ok=True)
C.mkdir(parents=True, exist_ok=True)
base = yaml.safe_load((R / 'configs/s38/s38_spmeas.yaml').read_text())
manifest = Path(base['DATA']['SPLITS_MANIFEST'])
assert manifest.is_file() and manifest.name == 'splits_semkine.json'
seqs = sequences_for_split(Path(base['DATA']['ROOT']), 'train', manifest)
subjects = sorted({s.split('_')[0] for s, _ in seqs})
assert len(seqs) == 72 and subjects == ['ch','lfz','lpc','lr','ly','lyh','lyq','ycy','ylf']
records = {'split_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'subjects': subjects, 'train_sequences': len(seqs), 'pairs': {}}
for seed in (3407, 3408):
    anchor = R / f'outputs/semkine/s38_spmeas_s{seed}/last.ckpt'
    cfg = copy.deepcopy(base)
    cfg['SEED'] = seed
    cfg['MODEL'].update(U1A_MODE='shared', U1A_HIDDEN=64, U1A_FREEZE_ENCODER=True)
    cfg['TRAIN']['TRAINABLE_PREFIXES'] = ['u1a_readout.']
    torch.manual_seed(seed)
    shared = MNISTModel(copy.deepcopy(cfg))
    loaded = shared.init_from_abs_checkpoint(str(anchor))
    cfg['MODEL']['U1A_MODE'] = 'untied'
    torch.manual_seed(seed)
    untied = MNISTModel(copy.deepcopy(cfg))
    untied.init_from_abs_checkpoint(str(anchor))
    copy_shared_to_untied(shared.u1a_readout, untied.u1a_readout)
    for k, v in shared.event_encoder.state_dict().items():
        assert torch.equal(v, untied.event_encoder.state_dict()[k]), k
    pair = {'anchor': str(anchor), 'anchor_sha256': hashlib.sha256(anchor.read_bytes()).hexdigest(),
            'loaded_common_tensors': loaded, 'arms': {}}
    for mode, model in (('shared', shared), ('untied', untied)):
        init = O / 'init' / f'{mode}_s{seed}.ckpt'
        init.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': model.state_dict()}, init)
        pair['arms'][mode] = {'init': str(init), 'params': sum(p.numel() for p in model.parameters()),
                              'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        for budget in (20, 2000, 6000):
            conf = copy.deepcopy(cfg)
            conf['MODEL'].update(U1A_MODE=mode, INIT_FROM=str(init))
            tag = 'debug' if budget == 20 else ('2k' if budget == 2000 else '6k')
            run = f'u1a_{mode}_{tag}_s{seed}'
            conf['TRAIN'].update(MAX_STEPS=budget, DEVICES=2, BATCH_SIZE_PER_GPU=512,
                                  ACCUMULATE_GRAD_BATCHES=1, NUM_WORKERS=14,
                                  VAL_CHECK_INTERVAL=2000 if budget > 20 else 20,
                                  SAVE_EVERY_N_STEPS=500 if budget > 20 else 20,
                                  RUN_NAME=run, OUTPUT_DIR=str(R/'outputs/semkine'/run))
            conf['EVAL']['OUTPUT_DIR'] = str(R/'outputs/semkine'/run/'eval')
            if budget == 20:
                conf['TRAIN'].update(BATCH_SIZE_PER_GPU=16, NUM_WORKERS=2, WARMUP_STEPS=0, LIMIT_VAL_BATCHES=1)
            (C/f'{run}.yaml').write_text(yaml.safe_dump(conf, sort_keys=False))
    records['pairs'][str(seed)] = pair
(O/'initialization.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
