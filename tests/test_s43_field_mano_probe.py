"""Stage4 reader arithmetic/shape fixtures, no scientific targets or fitting."""
import json
import numpy as np
import pytest
import torch
from tools.s43_revisit import newfield_mano_probe as p


def tensors():
    torch.set_num_threads(2)
    return torch.randn(2,40,180,240),torch.randn(2,8,90,5,7,7),torch.randn(2,70)


def test_same_hidden_initialization_and_both_final_heads_exact_zero():
    models=[]
    for reader in p.READERS:
        torch.manual_seed(3407);models.append(p.IdentityFeatureReader('static',reader))
    a,b=models
    for k,v in a.state_dict().items():
        if k.startswith('head.4.'):continue
        torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    g,l,c=tensors()
    assert torch.count_nonzero(a(g,l,c))==torch.count_nonzero(b(g,l,c))==0
    assert b.accounting()['parameters']-a.accounting()['parameters']==-12*129


@pytest.mark.parametrize('arm',['context_only','global_only'])
def test_disabled_measurement_branches_cannot_leak_features_or_biases(arm):
    torch.manual_seed(3407);m=p.IdentityFeatureReader(arm,'joint63')
    with torch.no_grad():m.head[-1].weight.fill_(.01)
    g,l,c=tensors();a=m(g,l,c)
    b=m(g,torch.randn_like(l)*100,c)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    if arm=='context_only':torch.testing.assert_close(a,m(torch.randn_like(g)*100,l,c),rtol=0,atol=0)
    assert all(not x.requires_grad for x in m.local_encoder.parameters())


def test_fixed10mm_loss_and_joint63_identity_without_mean_std():
    target=torch.full((2,21,3),.01);z=torch.zeros_like(target)
    assert p.joint_mse(z,target).item()==pytest.approx(1)
    raw=torch.zeros(2,63,requires_grad=True);identity=torch.randn(2,21,3)
    out,state=p.decode_joints(raw,torch.zeros(2,51),torch.zeros(2,10),'joint63',None,identity)
    torch.testing.assert_close(out,identity,rtol=0,atol=0);assert state is None
    p.joint_mse(out,identity+.001).backward();assert raw.grad.abs().sum()>0


def test_zero_head_first_gradient_role_and_later_encoder_connectivity():
    torch.manual_seed(3407);m=p.IdentityFeatureReader('static','joint63');g,l,c=tensors()
    m(g,l,c).sub(1).square().mean().backward()
    assert m.head[-1].weight.grad.abs().sum()>0
    assert m.local_encoder[0].weight.grad.abs().sum()==0
    m.zero_grad(set_to_none=True)
    with torch.no_grad():m.head[-1].weight.fill_(.001)
    m(g,l,c).sub(1).square().mean().backward()
    assert m.local_encoder[0].weight.grad.abs().sum()>0


def test_target_is_continuous_joint_tensor_with_distinct_existing_endpoint():
    rows=[dict(case_id='toy',cutoff_ms=2)]
    label={'toy':dict(target=np.ones((21,3)).tolist(),target_joints=np.arange(3*21*3).reshape(3,21,3).tolist())}
    np.testing.assert_array_equal(p.target_joints(rows,label,'existing'),np.ones((1,21,3)))
    np.testing.assert_array_equal(p.target_joints(rows,label,'synthetic'),np.arange(3*21*3).reshape(3,21,3)[2:3])


def test_input_scales_fit_only_and_exact_stage3_context_rule():
    g=torch.zeros(3,40,2,2);g[1]=2;g[2]=1e6
    l=torch.ones(3,8,90,5,2,2);l[2]=1e6
    context=np.zeros((3,70),np.float32);context[1]=1e-7;context[2]=1e6
    mask=np.array([True,True,False]);index=np.arange(3)
    a=p.input_normalization(g,l,index,index,context,mask)
    g[2]=-1e9;l[2]=-1e9;context[2]=-1e9
    b=p.input_normalization(g,l,index,index,context,mask)
    for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
    np.testing.assert_array_equal(a[3],np.ones(70));assert np.all(a[0]>=1)


def test_prepare_reuses_static_coordinates_collapse_and_never_decodes_poison_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'OUT',tmp_path);monkeypatch.setattr(p,'require_gate',lambda x:True)
    source=tmp_path/'field_atomic_v1/features/synthetic';source.mkdir(parents=True)
    rows=[]
    for i in range(80):
        for arm in set(p.SOURCE_ARM.values()):
            for condition in p.P.CONDITIONS:
                rows.append(dict(arm=arm,case_id=str(i),condition=condition,coordinate_file=arm,transform='none'))
    (source/'rows.json').write_text(json.dumps(rows));(source/'labels.json').write_bytes(b'\xffPOISON')
    freeze=tmp_path/'gate.json';freeze.write_text('{}');dest=tmp_path/'new';order=[]
    original=p.shutil.copyfile
    def copy_after_inputs(source,destination):
        assert (dest/'observations_complete.json').exists() and not (dest/'labels.json').exists()
        order.append('labels_copied_after_observations')
        return original(source,destination)
    monkeypatch.setattr(p.shutil,'copyfile',copy_after_inputs)
    p.prepare('synthetic',dest,freeze)
    prepared=json.loads((dest/'rows.json').read_text());assert len(prepared)==2800
    assert all(r['coordinate_file']=='static' and r['transform']=='collapse' for r in prepared if r['arm']=='static_temporal_collapse')
    assert (dest/'labels.json').read_bytes()==b'\xffPOISON'
    assert order==['labels_copied_after_observations']


def test_nominal_guard_uses_nominal_fit_for_training_and_statistics():
    rows=[dict(fold=-1,partition=partition,training_view=condition in ('none','wrong_x','wrong_yaw'),condition=condition)
          for partition in ('FIT7','SELECT') for condition in p.P.CONDITIONS]
    fold,a=p.training_masks(rows,'existing',-1,'all3')
    _,b=p.training_masks(rows,'existing',-1,'nominal_guard')
    assert fold.sum()==5 and a.sum()==3 and b.sum()==1
    assert rows[np.flatnonzero(b)[0]]['condition']=='none'


def test_all_four_guards_must_exist_before_any_score_or_poison_label_decode(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'require_gate',lambda x:True);calls=[]
    def check(folder,feature,domain,reader,freeze,regime):
        calls.append(regime)
        if regime=='nominal_guard':raise RuntimeError('guard not complete')
    monkeypatch.setattr(p,'check_predictions',check)
    for domain in ('synthetic','existing'):
        d=tmp_path/'features'/domain;d.mkdir(parents=True)
        (d/'rows.json').write_text('[]');(d/'labels.json').write_bytes(b'\xffPOISON')
        (d/'complete.json').write_text(json.dumps(dict(rows_sha256=p.sha(d/'rows.json'),labels_sha256=p.sha(d/'labels.json'))))
    with pytest.raises(RuntimeError,match='guard not complete'):
        p.score(tmp_path/'features',tmp_path/'readers',tmp_path/'report',tmp_path/'freeze.json')
    assert calls.count('all3')==28 and calls[-1]=='nominal_guard'
    assert not (tmp_path/'report').exists()


def test_json_receipts_choose_utf8_explicitly_for_gpu_process_locale(tmp_path,monkeypatch):
    from pathlib import Path
    original=Path.write_text;seen=[]
    def write(self,data,**kwargs):
        seen.append(kwargs.get('encoding'));return original(self,data,**kwargs)
    monkeypatch.setattr(Path,'write_text',write)
    p.dump(tmp_path/'receipt.json',dict(meaning='std<1e-6→1'))
    assert seen==['utf-8']
    assert json.loads((tmp_path/'receipt.json').read_text(encoding='utf-8'))['meaning'].endswith('→1')
