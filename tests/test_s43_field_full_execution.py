"""Full execution bridge preserves science and dispatch domain, fixture-only."""
import json,threading
from pathlib import Path
import pytest
import torch
from tools.s43_revisit import s43_field_training_R1 as P
from tools.s43_revisit import s43_field_full_training as T
from tools.s43_revisit import s43_field_full_queue as Q


def test_frozen_full_budget_model_and_normalization_contract_preserved():
 assert T.STAGES==P.STAGES and T.ARMS==P.ARMS and T.FULL_SUBJECTS==P.FULL_SUBJECTS
 assert T.PILOT_FREEZE_SHA==P.sha(T.PILOT_FREEZE)
 for arm in T.ARMS:
  old=P.read(P.OUT/'configs/full'/f'{arm}.json');new=T.read(T.OUT/'configs/full_fastreadout'/f'{arm}.json')
  assert new.pop('execution_variant')==T.EXECUTION_VARIANT
  new['schema']=old['schema'];assert new['runtime_contract'].pop('readout_backend')=='native_ring_bins_v1'
  assert new==old
  a=T.arm_contract(arm);assert a.pop('readout_backend')=='native_ring_bins_v1';assert a==P.arm_contract(arm)


def test_full_loader_rejects_prior_weights_before_factory(tmp_path):
 file=tmp_path/'wrong.ckpt';torch.save(dict(schema='s43_field_training_R1_v1',STATE_FP32=True,RENDER_FP32=True,LOSS_FP32=True),file)
 with pytest.raises(ValueError):T.load_weights(file)
 torch.save(dict(schema='s43_field_full_training_v1',STATE_FP32=True,RENDER_FP32=True,LOSS_FP32=True,execution_variant='other'),file)
 with pytest.raises(ValueError):T.load_weights(file)


def test_five_full_jobs_use_new_loader_all_four_authorized_gpus_and_fresh_init(tmp_path,monkeypatch):
 monkeypatch.setattr(T,'OUT',tmp_path);calls=[];lock=threading.Lock()
 class Process:
  def __init__(self,command,**kwargs):
   with lock:calls.append((command,kwargs));self.pid=100+len(calls)
  def wait(self):return 0
  def terminate(self):raise AssertionError('healthy fixture must not terminate')
 monkeypatch.setattr(Q.subprocess,'Popen',Process)
 Q.run_stage('full',tmp_path/'freeze.json')
 assert len(calls)==5
 arms=[]
 for cmd,kw in calls:
  assert cmd[1:4]==['-m','tools.s43_revisit.s43_field_full_training','train']
  assert '--resume' not in cmd
  config=Path(cmd[cmd.index('--config')+1]);assert config.parent==tmp_path/'configs/full_fastreadout'
  arms.append(config.stem);env=kw['env'];assert env['CUDA_VISIBLE_DEVICES'] in ('4','5','6','7')
  assert env['OPENBLAS_NUM_THREADS']=='1' and env['OMP_NUM_THREADS']=='8'
 assert sorted(arms)==sorted(T.ARMS)
 assert json.loads((tmp_path/'full_training_jobs_complete.json').read_text())['all5_fixed_budgets_complete']
