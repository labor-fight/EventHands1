"""Real parent Adam/gradient/freeze and paired producer contracts for I-only200."""
from pathlib import Path
import copy,sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import numpy as np,torch,pytest
from config import load_config
from mano_layer import ManoLayer
from semkine.events import collate_packets
from tools.xkf_fusion.motion_data import MotionDataset,arm_config,collate_incremental
from tools.xkf_fusion.finetune_motion import IncrementalFinetuneModel,validate_parent,state_hashes,optimizer_named_state
from tools.xkf_repair.data import build_train_dataset
from tools.xdl.data import packet_hash,counter_rng
BASE=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune'

def config(arm='raw',seed=3407):return arm_config(load_config(str(ROOT/f'configs/xkf_repair/repair_s{seed}.yaml')),arm)
def parent(seed=3407):return torch.load(BASE/f'raw_s{seed}/xkf_donly_raw_s{seed}-step=2400.ckpt',map_location='cpu',weights_only=False)

@pytest.mark.parametrize('seed',[3407,3408,3409])
def test_exact_parent_adam_migration_and_independent_clocks(seed):
 torch.set_num_threads(4);model=IncrementalFinetuneModel(config(seed=seed));saved=parent(seed);audit=validate_parent(model,saved)
 assert audit['parameter_count']==96 and audit['direct_parameter_count']==60
 assert audit['parent_I_adam_step']==2200 and audit['parent_D_adam_step']==2400
 assert sum(p.requires_grad for p in model.parameters())==36
 bad=copy.deepcopy(saved);bad['global_step']=2200
 with pytest.raises(ValueError):validate_parent(model,bad)


def test_source_GT_pairing_and_unchanged_original_packets():
 datasets={a:MotionDataset(config(a),start_step=2400,max_steps=2401) for a in ('raw','aug')}
 originals={a:build_train_dataset(config(a),start_step=2400,max_steps=2401) for a in ('raw','aug')}
 for idx in (0,17,511):
  pair={a:datasets[a][idx] for a in datasets}
  assert pair['raw']['sampling_identity']==pair['aug']['sampling_identity']
  for arm,item in pair.items():
   reference=originals[arm][idx]
   for name in ('main',):assert packet_hash(collate_packets([item[name]]))==packet_hash(collate_packets([reference[name]]))
   assert all(packet_hash(collate_packets([a]))==packet_hash(collate_packets([b])) for a,b in zip(item['prefix'],reference['prefix']))
   assert all(np.array_equal(a.lnes,b.lnes) and a.lnes_window_ms==b.lnes_window_ms for a,b in zip([item['main'],*item['prefix']],[reference['main'],*reference['prefix']]))
   assert np.array_equal(item['motion_delta'],item['main'].target-item['previous_gt'])
   assert len(item['prefix'])==3 and all(v.lnes is not None for v in item['prefix'])


def test_actual_bf16_motion_backward_retains_D_BN_EMA_and_burn():
 torch.set_num_threads(4);cfg=config();model=IncrementalFinetuneModel(cfg);saved=parent();validate_parent(model,saved)
 for module in model.modules():
  if isinstance(module,ManoLayer):module.fast_chain=True
 ds=MotionDataset(cfg,start_step=2400,max_steps=2401)
 idx=next(i for i in range(20) if counter_rng(3407,2400,i,'self').random()<.5)
 batch=collate_incremental([ds[idx]])
 optimizer=model.configure_optimizers()['optimizer'];optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
 before=state_hashes(model,frozen_only=True);named=optimizer_named_state(model,optimizer)
 calls=[];hook=model.direct.register_forward_pre_hook(lambda *a:calls.append(torch.is_grad_enabled()))
 model._trainer=SimpleNamespace(global_step=2400);model.log=lambda *a,**kw:None;model.train()
 with torch.autocast('cpu',dtype=torch.bfloat16):loss=model.training_step(batch,0)
 loss.backward();hook.remove()
 assert torch.isfinite(loss) and model.xdl_self_rows==1 and model.xdl_burn_rows==3
 assert calls==[False,False,False]
 for name,p in model.named_parameters():
  if name.startswith('direct.'):assert p.grad is None
  else:assert p.grad is not None and torch.isfinite(p.grad).all()
 optimizer.step();assert state_hashes(model,frozen_only=True)==before
 after=optimizer_named_state(model,optimizer)
 for name,p in model.named_parameters():
  assert int(optimizer.state[p]['step'])==(2400 if name.startswith('direct.') else 2201)
  if name.startswith('direct.'):assert after[name]==named[name]
 marker={};model.on_save_checkpoint(marker)
 assert 'xkf_direct_continuation' not in marker
 assert marker['xkf_incremental_continuation']['parent_direct_continuation']==saved['xkf_direct_continuation']
 model._trainer=None


def test_burn_incoming_does_not_change_motion_label():
 cfg=config();model=IncrementalFinetuneModel(cfg);ds=MotionDataset(cfg,start_step=2400,max_steps=2401);batch=collate_incremental([ds[0]])
 main=batch['main'];target=batch['motion_delta'].clone();prediction=main.prev_state+target
 a,_=model.incremental_loss(prediction,main,batch)
 main.prev_state=main.prev_state+.1;prediction=main.prev_state+target
 b,_=model.incremental_loss(prediction,main,batch)
 assert a<1e-12 and b<1e-12 and torch.equal(batch['motion_delta'],target)
