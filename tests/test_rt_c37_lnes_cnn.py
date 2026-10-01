"""Root tracking C37: the LNES-CNN frontend builds the evaluator's LNES from raw events, hands S37's
routed readout node features of S37's shapes, and leaves S37 untouched."""
from pathlib import Path

import numpy as np
import pytest
import torch

from semkine import eval_track as ET
from semkine.lnes_cnn import LnesCNN

DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")


def _packets_from_zgz(ends, window=50):
    events, offsets, aux, pos51 = ET.load_sequence(DATA, "val", "zgz_global")
    tsub = np.load(DATA / "val" / "zgz_global_tsub.npy", mmap_mode="r")
    evs, ptr, ref = [], [0], []
    for b, end in enumerate(ends):
        e = ET._window_events(events, offsets, tsub, int(end), window)
        e[:, 0] = b
        evs.append(e)
        ptr.append(ptr[-1] + len(e))
        ref.append(ET.build_lnes(events, offsets, int(end), window))
    return (torch.from_numpy(np.concatenate(evs)), torch.tensor(ptr),
            torch.full((len(ends),), window * 1e-3), np.stack(ref))


@pytest.mark.skipif(not DATA.exists(), reason="needs the dataset")
def test_lnes_matches_the_evaluator():
    ev, ptr, dt, ref = _packets_from_zgz([5049, 20049, 40049, 60049])
    img = LnesCNN().lnes(ev, ptr, dt).numpy()                      # (B, 2, H, W)
    np.testing.assert_allclose(img.transpose(0, 2, 3, 1), ref, atol=1e-6)


def test_shapes_empty_packet_and_gradients():
    torch.manual_seed(0)
    enc = LnesCNN()
    g = torch.Generator().manual_seed(0)
    n = [400, 0, 90]
    ev = torch.zeros(sum(n), 5)
    ev[:, 0] = torch.repeat_interleave(torch.arange(3), torch.tensor(n))
    ev[:, 1] = torch.randint(0, 240, (sum(n),), generator=g).float()
    ev[:, 2] = torch.randint(0, 180, (sum(n),), generator=g).float()
    ev[:, 3] = torch.rand(sum(n), generator=g) * 0.05
    ev[:, 4] = (torch.rand(sum(n), generator=g) > 0.5).float()
    ptr = torch.tensor([0, 400, 400, 490])
    out, h, gg, px, py, mask = enc(ev, ptr, torch.full((3,), 0.05), return_nodes=True)
    assert out.shape == (3, 512) and h.shape == (3, 23 * 30, 128) and mask.shape == (3, 690)
    assert torch.equal(out[1], torch.zeros(512)) and not bool(mask[1].any())
    assert float(px.max()) < 240 and float(py.max()) < 180 and bool(mask[0].any())
    (out.pow(2).sum() + h.pow(2).sum()).backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_hidden_must_be_layer2_width():
    with pytest.raises(ValueError):
        LnesCNN(hidden=96)
