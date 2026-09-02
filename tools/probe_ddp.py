#!/usr/bin/env python3
"""Minimal two-rank NCCL probe: does a collective on GPU 0-1 complete at all?

Separates "the model deadlocks under DDP" from "this box's NCCL hangs on its first collective".
Nothing here touches a model or the dataset.

On this machine the answer is the latter. The process group comes up in ~1.3 s and then a bare
1024-element `all_reduce` never returns -- measured at 118 s before a timeout killed it. With
`NCCL_P2P_DISABLE=1` the same probe finishes in under 5 s, so PCIe peer-to-peer between GPU 0 and 1
is advertised but does not work. `tools/run_s36.sh` sets that variable for exactly this reason.

    python tools/probe_ddp.py                     # expect a hang, ~120 s with timeout
    NCCL_P2P_DISABLE=1 python tools/probe_ddp.py  # expect PROBE_OK
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    t0 = time.time()
    dist.init_process_group("nccl", rank=rank, world_size=world)
    print(f"[rank {rank}] process group up in {time.time() - t0:.2f}s", flush=True)

    torch.cuda.set_device(rank)
    t1 = time.time()
    x = torch.ones(1024, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    print(f"[rank {rank}] all_reduce in {time.time() - t1:.2f}s -> {float(x[0])} "
          f"(p2p_disabled={os.environ.get('NCCL_P2P_DISABLE', '0')})", flush=True)

    t2 = time.time()
    net = torch.nn.Linear(64, 64).to(rank)
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[rank], find_unused_parameters=False)
    ddp(torch.randn(8, 64, device=rank)).sum().backward()
    torch.cuda.synchronize()
    print(f"[rank {rank}] ddp backward in {time.time() - t2:.2f}s", flush=True)
    dist.destroy_process_group()
    if rank == 0:
        print("PROBE_OK")


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
