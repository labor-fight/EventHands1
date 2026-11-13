"""semkine/rfs.py: the route-first sampler (CPU; CUDA_VISIBLE_DEVICES="")."""
import torch

from semkine import rfs as RF
from semkine import xyz_mesh as XM


def _ptr(counts):
    return torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.long)


def test_all_eligible_equals_stride_sample():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        counts = torch.randint(0, 6000, (7,), generator=g).tolist()
        ptr = _ptr(counts)
        flags = torch.ones(int(ptr[-1]), dtype=torch.bool)
        for cap in (2048, 100, 1):
            s1, m1 = XM.stride_sample(ptr, cap)
            s2, m2 = RF.stride_over(flags, ptr, cap)
            assert torch.equal(m1, m2)
            assert torch.equal(s1[m1], s2[m2])


def test_only_flagged_events_and_per_packet():
    g = torch.Generator().manual_seed(1)
    counts = [0, 300, 5000, 17, 2048, 4096]
    ptr = _ptr(counts)
    flags = torch.rand(int(ptr[-1]), generator=g) < 0.3
    src, mask = RF.stride_over(flags, ptr, 2048)
    for b in range(len(counts)):
        s = src[b][mask[b]]
        assert bool(flags[s].all())
        assert bool(((s >= ptr[b]) & (s < ptr[b + 1])).all())
        n_flag = int(flags[ptr[b]:ptr[b + 1]].sum())
        assert int(mask[b].sum()) == min(n_flag, 2048)
        assert bool((s[1:] > s[:-1]).all()) if s.numel() > 1 else True
