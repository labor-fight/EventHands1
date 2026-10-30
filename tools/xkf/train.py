#!/usr/bin/env python3
"""XKF trainer: `tools/xdl/train.py` (provenance, frozen manifest, 2 ranks x 512, exact step 6000, GPU2-7) with the
long-window dataset and `semkine.xkf.XKFModel`. Nothing in the XDL trainer is edited; its module-level names are
rebound here before `main()` runs."""
from __future__ import annotations

from pathlib import Path
import json
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'model'))

import tools.xdl.train as T  # noqa: E402
import semkine.xyz_recovery_study as S  # noqa: E402
from semkine.xkf import XKFModel  # noqa: E402
from tools.xkf.data import build_train_dataset  # noqa: E402
import torch  # noqa: E402


class XKFProvenance(T.RecoveryProvenance):
    """Keep the canonical exposure audit and add branch learning diagnostics."""

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        if trainer.global_rank == 0:
            path = self.output_dir / 'training_metadata.json'
            metadata = json.loads(path.read_text())
            metadata.update(study='XKF-20261015-debug-before-full',
                            loss_contract='log10(L_I) + log10(L_D); independent branches, 2 ranks x 512',
                            fusion_contract='constant correlated-residual gains; no fusion gradient')
            if int(metadata['start_step']) == 0:
                torch.save({'state_dict': {k: v.detach().cpu() for k, v in pl_module.state_dict().items()},
                            'global_step': 0, 'pytorch-lightning_version': T.canonical.pl.__version__,
                            'hyper_parameters': dict(pl_module.hparams)}, self.output_dir / 'initial_state.ckpt')
            initial = self.output_dir / 'initial_state.ckpt'
            if initial.exists():
                metadata['initial_checkpoint_sha256'] = T.sha(initial)
            T.atomic_json(path, metadata)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.global_rank != 0:
            return
        step = int(trainer.global_step)
        if step != 1 and step % 20:
            return
        metrics = {k: float(v.detach().float().cpu()) for k, v in trainer.callback_metrics.items()
                   if k.startswith('train_') and torch.is_tensor(v) and v.numel() == 1}
        record = {'global_step': step, 'metrics': metrics,
                  'gains': pl_module.gains().detach().cpu().tolist(),
                  'gain_updates': int(pl_module.xkf_updates), 'preflight_not_formal': self.preflight}
        with (self.output_dir / 'learning.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        T.atomic_json(self.output_dir / 'learning_latest.json', record)

T.build_train_dataset = build_train_dataset
T.RecoveryProvenance = XKFProvenance
T.SOURCE_FILES = tuple(T.SOURCE_FILES) + ('semkine/xkf.py', 'semkine/xkf_direct.py', 'tools/xkf/data.py',
                                          'tools/xkf/train.py', 'model/backbones.py', 'semkine/lie.py')
# `T.main` imports the model class by name at call time
S.RecoveryXYZModel = XKFModel

if __name__ == '__main__':
    T.main()
