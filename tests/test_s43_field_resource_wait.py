"""Resource scheduling simulation, no GPU work or process termination."""
import json
from tools.s43_revisit.s43_field_collect import wait_resource_idle


def test_foreign_occupancy_really_waits_on_fixed_device_and_logs_transition(tmp_path):
    states=iter(([444],[444],[]));queried=[];slept=[]
    def query(gpu):
        queried.append(gpu);return dict(physical_gpu=gpu,gpu_uuid='GPU-fixed4',pids=next(states))
    result=wait_resource_idle(4,tmp_path,'fixed_case',query=query,sleep=slept.append,interval_seconds=5)
    assert queried==[4,4,4] and slept==[5,5] and result['status']=='IDLE' and result['checks']==3
    records=[json.loads(x) for x in (tmp_path/'resource_wait/fixed_case.jsonl').read_text().splitlines()]
    assert [x['status'] for x in records]==['WAITING','WAITING','IDLE']
    assert all(x['read_only'] for x in records)
