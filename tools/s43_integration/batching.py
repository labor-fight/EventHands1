"""Padding-safe supervised batches using the same streaming constructor.

The fixed two-window debug dataset has no padding. This reusable collator also
supports variable-length client batches without passing padding targets/state
into the original S43 loss. GT remains a separate target argument.
"""
import numpy as np
import torch
from .memory import batch_from_windows


def collate_windows(windows,targets,*,mode='br',provider=None):
    if len(windows)!=len(targets):raise ValueError('window/target row mismatch')
    reps=batch_from_windows(windows,mode=mode,provider=provider)
    valid=np.array([bool(w.get('valid',True)) for w in windows],dtype=bool)
    histories=[];betas=[];cameras=[];labels=[]
    for w,y,keep in zip(windows,targets,valid):
        if not keep:
            # Do not inspect arbitrary padded payloads, including NaN labels.
            histories.append(np.zeros(51,np.float32));betas.append(np.zeros(10,np.float32))
            cameras.append(np.eye(3,dtype=np.float32));labels.append(np.zeros(51,np.float32));continue
        def array(v):return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
        histories.append(np.array(array(w.get('history',w.get('prev_state'))),np.float32).reshape(51))
        betas.append(np.array(array(w['betas']),np.float32).reshape(10))
        cameras.append(np.array(array(w.get('K_cam',w.get('camera_K'))),np.float32).reshape(3,3))
        labels.append(np.array(array(y),np.float32).reshape(51))
    return {'representations':reps,'previous':torch.tensor(np.stack(histories)),
            'betas':torch.tensor(np.stack(betas)),'camera_K':torch.tensor(np.stack(cameras)),
            'target':torch.tensor(np.stack(labels)),'valid':torch.tensor(valid)}


def masked_original_loss(model,prediction,batch):
    keep=batch['valid'].to(prediction.device)
    if not bool(keep.any()):
        return prediction.sum()*0.,{'valid_rows':0,'padding_rows':int(len(keep))}
    result,parts=model.loss(prediction[keep],batch['target'].to(prediction.device)[keep],
                            batch['betas'].to(prediction.device)[keep])
    return result,{**parts,'valid_rows':int(keep.sum()),'padding_rows':int((~keep).sum())}
