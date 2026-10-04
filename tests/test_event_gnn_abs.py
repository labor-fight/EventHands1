"""absolute EventGNN control: the absolute GNN arm never reads prev and builds the non-linear head."""
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine.events import EventPacketBatch  # noqa: E402

CFG = ROOT / "configs/semkine_recipes/event_gnn_abs.yaml"


def _batch(prev):
    n = 500
    ev = torch.zeros(n, 5)
    ev[:, 1] = torch.rand(n) * 239
    ev[:, 2] = torch.rand(n) * 179
    ev[:, 3] = torch.linspace(0, 0.05, n)
    ev[:, 4] = (torch.rand(n) > 0.5).float()
    K = torch.tensor([[603.45, 0, 325.09], [0, 602.96, 242.10], [0, 0, 1.0]]).view(1, 3, 3)
    return EventPacketBatch(events=ev, ptr=torch.tensor([0, n]), sequence_id=torch.zeros(1, dtype=torch.long),
                            t_start_us=torch.zeros(1, dtype=torch.long), t_end_us=torch.tensor([50000]),
                            delta_t_s=torch.tensor([0.05]), is_sequence_start=torch.zeros(1, dtype=torch.bool),
                            is_sequence_end=torch.zeros(1, dtype=torch.bool), target=prev.clone(),
                            prev_state=prev, betas=torch.zeros(1, 10), camera_K=K, lnes=None)


def test_e7_reads_no_prev_and_head_is_mlp():
    cfg = load_config(CFG)
    m = MNISTModel(cfg).eval()
    assert isinstance(m.pose_head, torch.nn.Sequential) and m.pose_head[0].out_features == 512
    assert not hasattr(m, "prev_mlp") or not m.prevpos_embed
    p1 = torch.zeros(1, 51)
    p1[0, 2] = 0.45
    p2 = p1 + 0.3
    b1 = _batch(p1)
    b2 = copy.copy(b1)
    b2.prev_state = p2
    with torch.no_grad():
        o1, o2 = m.forward_packet(b1), m.forward_packet(b2)
    assert torch.equal(o1, o2), "absolute arm must not depend on prev"
