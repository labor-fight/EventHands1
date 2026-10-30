"""Optional shared Torch readout from the frozen integer count core.

This is a separately validated engineering adapter, not part of the frozen
Stage3/4 task implementations. Both training and serving can call this same
function. It transfers binned integer counts instead of a repeated grid40.
"""
from dataclasses import dataclass
import numpy as np
import torch

from .field_core import TimeFieldSnapshot


@dataclass(frozen=True)
class TensorField:
    global40: torch.Tensor       # [B,40,180,240], log-density/coverage/log-count
    cell3: torch.Tensor          # [B,8,3,H/r,W/r], two log-density + exposure
    observed_support: torch.Tensor  # [B,1,180,240], fine/coarse support, no coverage
    count_payload_bytes: int
    reference_grid_bytes: int


def tensorize(snapshots, device="cpu"):
    """Read only snapshots; geometry, poses, targets and learned weights absent.

    Aggregation remains the exact frozen snapshot.rasterize() on CPU. GPU/CPU
    Torch uses the same integer coarse sums and FP32 analytic transforms.
    Torch versus NumPy log1p is a numeric-tolerance comparison, not bitwise.
    This function does not claim an end-to-end latency improvement by itself.
    """
    snapshots=list(snapshots)
    if not snapshots or not all(isinstance(s,TimeFieldSnapshot) for s in snapshots):
        raise ValueError("a nonempty batch of TimeFieldSnapshot required")
    config=snapshots[0].config
    if any(s.config!=config for s in snapshots):raise ValueError("same count configuration required")
    if (config.width,config.height,config.time_bins,config.horizon_ms)!=(240,180,8,50):
        raise ValueError("prototype fixes 240x180 and eight bins in50 native-ms")
    rasters=[s.rasterize() for s in snapshots]
    fine=np.stack([r["fine_counts"] for r in rasters])
    # uint32 counts over at most50 native-ms rows and16px coarse cells fit
    # int64. Explicitly reject invalid externally fabricated snapshots.
    if fine.max(initial=0)>np.iinfo(np.int64).max//16:
        raise OverflowError("count sums cannot safely use signed64")
    exp=np.stack([r["bin_exposure_ms"] for r in rasters]).astype(np.int64)
    widths=np.stack([r["bin_width_ms"] for r in rasters]).astype(np.int64)
    if np.any(widths<=0) or np.any(exp>widths):raise ValueError("invalid native exposure")
    integer=fine.astype(np.int64)
    payload=int(integer.nbytes+exp.nbytes+widths.nbytes)
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        counts=torch.as_tensor(integer,device=device)
        exposure=torch.as_tensor(exp,device=device).float()/torch.as_tensor(widths,device=device).float()
        B,T,H,W,P=counts.shape
        step=4//config.resolution
        coarse=counts.reshape(B,T,45,step,60,step,P).sum((3,5))
        density=counts.float().permute(0,1,4,2,3)/(config.resolution**2)
        logfine=torch.log1p(density)
        cells=torch.cat((logfine,exposure[:,:,None,None,None].expand(B,T,1,H,W)),2)
        f=logfine.repeat_interleave(config.resolution,-2).repeat_interleave(config.resolution,-1)
        q=torch.log1p(coarse.float()).permute(0,1,4,2,3).repeat_interleave(4,-2).repeat_interleave(4,-1)
        c=exposure[:,:,None,None].expand(B,T,180,240)
        grid=torch.cat((f.flatten(1,2),c,q.flatten(1,2)),1)
        fine_support=(counts.sum((1,4))>0)[:,None].repeat_interleave(config.resolution,-2).repeat_interleave(config.resolution,-1)
        coarse_support=(coarse.sum((1,4))>0)[:,None].repeat_interleave(4,-2).repeat_interleave(4,-1)
        support=fine_support|coarse_support
    return TensorField(grid,cells,support,payload,int(grid.numel()*grid.element_size()))
