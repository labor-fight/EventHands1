"""Real mixed-sequence batch/online parity, without reading target values."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from .data import load_clips
from .memory import SequenceBank,batch_from_windows
from .model import load_integrated
from .train import windows


def main():
    torch.set_num_threads(2)
    rows=load_clips();chosen=[]
    for name in ['ch_local','lfz_global']:
        chosen.append(next(c for c in rows if c['recording']==name))
    previous=np.stack([c['history'] for c in chosen]).astype(np.float32)
    entries=windows(chosen,0,previous)
    offline=batch_from_windows(entries,mode='br')
    bank=SequenceBank('br',max_sequences=2)
    for w in entries:bank.reset(w['seq_id'],w['start_us'],w['history'],w['betas'],w['K_cam'])
    for b in range(50):
        for w in reversed(entries):
            t0=w['start_us']+b*1000;a=np.asarray(w['events']);keep=(a[:,0]>=t0)&(a[:,0]<t0+1000)
            bank.ingest(w['seq_id'],a[keep],np.asarray(w['source_ids'])[keep],t0+1000)
    online=[bank.readout(w['seq_id'],w['cutoff_us']) for w in entries]
    diffs={key:max(float(np.max(np.abs(a[key]-b[key]))) for a,b in zip(offline,online))
           for key in ['lnes','enhanced','R','templates']}
    assert max(diffs.values())==0
    model,_=load_integrated('configs/s43_integration/a3_br.yaml',
        'outputs/s43_integration_20261027/train/a3_br/last.ckpt');model.eval()
    p=torch.tensor(previous);beta=torch.tensor(np.stack([c['betas'] for c in chosen]).astype(np.float32))
    camera=torch.tensor(np.stack([c['K_cam'] for c in chosen]).astype(np.float32))
    with torch.inference_mode():
        batched=model(offline,p,beta,camera)
        separate=torch.cat([model([online[i]],p[i:i+1],beta[i:i+1],camera[i:i+1]) for i in range(2)])
    prediction_diff=float((batched-separate).abs().max())
    assert prediction_diff<1e-5
    other=bank.digest(entries[1]['seq_id'])
    w=entries[0];bank.reset(w['seq_id'],w['start_us'],w['history'],w['betas'],w['K_cam'],segment_id='new')
    assert bank.digest(entries[1]['seq_id'])==other
    for w in entries:bank.release(w['seq_id'])
    out={'real_checkpoint':True,'real_recordings':[c['recording'] for c in chosen],
         'interleaved_native_ms_packets':100,'representation_max_abs_diff':diffs,
         'batch_vs_individual_prediction_max_abs_diff':prediction_diff,
         'reset_one_sequence_keeps_other_exact':True,'released_sequences':len(bank.states),
         'GT_accessed_for_constructor_or_assertions':False,
         'source_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=Path('outputs/s43_integration_20261027/mixed_sequence_smoke.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(out,indent=2));print(out)


if __name__=='__main__':main()
