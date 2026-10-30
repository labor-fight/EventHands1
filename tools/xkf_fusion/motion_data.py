"""I-only paired data: original K4/self-burn packets and true motion labels."""
import hashlib
import numpy as np
from torch.utils.data import Dataset
from tools.xkf_repair.data import build_train_dataset,collate_motion
from tools.xkf_fusion.direct_data import arm_config,sampling_identity

class MotionDataset(Dataset):
    def __init__(self,cfg,*,start_step=2400,max_steps=2600):
        self.base=build_train_dataset(cfg,start_step=start_step,max_steps=max_steps)
        if not self.base.prefix_lnes:raise ValueError('original complete prefix LNES required')
        self.sequences,self.coverage=self.base.sequences,self.base.coverage
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        item=self.base[index];main=item['main'];h=self.base._handle(main.sequence_id)
        original=np.stack([h.pos51[int(main.meta['incoming_ms'])],h.pos51[int(main.meta['end_idx'])]]).astype(np.float32)
        item['sampling_identity']={**sampling_identity(item),'original_GT_endpoints_sha256':hashlib.sha256(original.tobytes()).hexdigest()}
        if not np.array_equal(item['motion_delta'],main.target-item['previous_gt']):raise ValueError('motion label changed')
        return item

def collate_incremental(items):
    result=collate_motion(items);result['sampling_identity']=[v['sampling_identity'] for v in items]
    return result
