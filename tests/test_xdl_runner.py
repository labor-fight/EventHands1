"""Resource and freeze guards for the long, exact-endpoint experiment."""
import importlib.util
from pathlib import Path
import pytest

P = Path(__file__).resolve().parents[1] / 'tools/xdl/run_study.py'


def module():
    spec = importlib.util.spec_from_file_location('xdl_runner', P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_endpoint_is_not_inferred_from_successful_process(tmp_path):
    m = module()
    (tmp_path / 'train_finish.json').write_text('{"status":"completed_exact_step6000","global_step":5999}')
    with pytest.raises(RuntimeError, match='6000'):
        m.verify_endpoint(tmp_path)


def endpoint_fixture(m, tmp_path, step=6000):
    import json
    import torch
    checkpoint = tmp_path / 'xdl_q1-step=6000.ckpt'
    torch.save({'global_step':step}, checkpoint)
    (tmp_path/'train_finish.json').write_text(json.dumps({
        'status':'completed_exact_step6000','global_step':6000,
        'preflight_not_formal':False,'checkpoint':str(checkpoint),
        'checkpoint_sha256':m.sha(checkpoint)}))
    return checkpoint


def test_duplicate_versioned_endpoint_is_rejected(tmp_path):
    m = module()
    checkpoint = endpoint_fixture(m, tmp_path)
    (tmp_path/'xdl_q1-step=6000-v1.ckpt').write_bytes(checkpoint.read_bytes())
    with pytest.raises(RuntimeError, match='unique'):
        m.verify_endpoint(tmp_path)


def test_saved_optimizer_step_overrides_filename_and_finish_claim(tmp_path):
    m = module()
    endpoint_fixture(m, tmp_path, step=5999)
    with pytest.raises(RuntimeError, match='saved.*6000'):
        m.verify_endpoint(tmp_path)


def test_budget_charges_all_active_cards_and_stops_before_cap():
    m = module()
    running = [{'phase':'training','cards':[2,3],'started_mono':0},
               {'phase':'training','cards':[4,5],'started_mono':0},
               {'phase':'evaluation','cards':[6],'started_mono':0}]
    spent = m.live_cost({'training':58.0,'evaluation':0.0}, running, 1800)
    assert spent == {'training':60.0,'evaluation':0.5}
    assert m.exhausted(spent) == 'training'


def test_restart_budget_carryover_is_bound_to_original_journal(tmp_path):
    import json
    m = module()
    journal = tmp_path/'journal.json'
    journal.write_text(json.dumps({'status':'stopped_incomplete',
                                  'gpu_hours_allocated':{'training':0.5,'evaluation':0.0}}))
    carry = tmp_path/'carry.json'
    carry.write_text(json.dumps({'source_journal':str(journal),
                                'source_journal_sha256':m.sha(journal),
                                'gpu_hours_allocated':{'training':0.5,'evaluation':0.0}}))
    assert m.load_prior_cost(carry) == {'training':0.5,'evaluation':0.0}
    journal.write_text('{}')
    with pytest.raises(RuntimeError, match='carryover'):
        m.load_prior_cost(carry)


def test_evaluation_cannot_precede_verified_training_endpoint():
    m = module()
    waiting = ['f1','q1','n1','f0','q0']
    ready = ['n1']
    assert m.next_job([2], waiting, ready) == ('evaluation','n1',[2])
    assert m.next_job([2,3], waiting, ready) == ('training','f1',[2,3])
    assert m.next_job([], waiting, ready) is None
