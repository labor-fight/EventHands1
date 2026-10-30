#!/usr/bin/env python3
"""Same fresh short wrapper plus explicit full-validation/return receipts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.xyz_c1_repro import train as W

class ValidationAudit(W.ORIGINAL.ThroughputCallback):
    def on_validation_start(self,trainer,model):
        self.validation_batches=0;self.validation_samples=0
        self.validation_before=W.normalized(model.state_dict())
    def on_validation_batch_end(self,trainer,model,outputs,batch,batch_idx,dataloader_idx):
        self.validation_batches+=1;self.validation_samples+=batch.batch_size
    def on_validation_end(self,trainer,model):
        if self.validation_batches!=126 or self.validation_samples!=64094:
            raise RuntimeError('full original validation denominator changed')
        same=self.validation_before==W.normalized(model.state_dict())
        if not same:raise RuntimeError('validation modified model state')
        W.save(Path(model.cfg['TRAIN']['OUTPUT_DIR'])/f'validation_rank{trainer.global_rank}_step{trainer.global_step:06}.json',{
            'status':'PASS','step':int(trainer.global_step),'batches':self.validation_batches,
            'samples_with_original_distributed_padding':self.validation_samples,
            'rank':trainer.global_rank,'world_size':trainer.world_size,'model_state_unchanged':same,
            'math_flags':W.flags(),'validation_harness_sha256':W.file_sha(__file__),
            'scope':'Original validation loader full126 batches per rank; no optimizer step during validation. Return to training is checked against baseline fifty-step trace.'})

if __name__=='__main__':
    W.ORIGINAL.ThroughputCallback=ValidationAudit
    W.main()
