"""Exact endpoint and whole-cohort scoring contracts for the XDL producer."""
from pathlib import Path
import importlib
import importlib.util
import pytest
import torch


def producer():
    assert importlib.util.find_spec('tools.xdl.make_row') is not None, 'missing exact-last row producer'
    return importlib.import_module('tools.xdl.make_row')


def test_checkpoint_rejects_earlier_fallback_and_ambiguous_6000(tmp_path):
    p=producer()
    torch.save({'global_step':5999},tmp_path/'run-step=5999.ckpt')
    with pytest.raises(RuntimeError):p.exact_checkpoint(tmp_path)
    expected=tmp_path/'run-step=6000.ckpt';torch.save({'global_step':6000},expected)
    assert p.exact_checkpoint(tmp_path)==expected
    torch.save({'global_step':6000},tmp_path/'run-step=6000-v1.ckpt')
    with pytest.raises(RuntimeError):p.exact_checkpoint(tmp_path)


def test_filename_cannot_override_actual_optimizer_step(tmp_path):
    p=producer()
    torch.save({'global_step':42},tmp_path/'run-step=6000.ckpt')
    with pytest.raises(RuntimeError):p.exact_checkpoint(tmp_path)


def test_scoring_requires_both_sequences_and_keeps_nonfinite_failure():
    p=producer()
    rows=[dict(seq='zgz_local',n_frames=1204,**{k:30. for k in p.METRICS}),
          dict(seq='zgz_global',n_frames=1386,**{k:10. for k in p.METRICS})]
    agg=p.accuracy_aggregate(rows)
    assert agg['overall']['mpjpe_ra_mm']==pytest.approx(49980/2590)
    assert agg['local']['mpjpe_ra_mm']==30 and agg['global']['mpjpe_ra_mm']==10
    with pytest.raises(RuntimeError):p.accuracy_aggregate(rows[:1])
    rows[0]['mpjpe_ra_mm']=float('nan')
    with pytest.raises(FloatingPointError):p.accuracy_aggregate(rows)
    invalid=[];raw=p.sanitize_json({'rows':rows},invalid=invalid)
    assert raw['rows'][0]['mpjpe_ra_mm'] is None and raw['rows'][1]['mpjpe_ra_mm']==10
    assert invalid==['$.rows[0].mpjpe_ra_mm']
