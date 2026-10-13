#!/usr/bin/env python3
"""trk6 / S42 candidate C debug gate (CPU): the batched EnsembleTeacher equals the evaluated ensemble rule
(`tools/dt/infer_probe.py combine`) row by row, its members are full-frame S39 networks even for a GLIMPSE student, and
one distillation term on a real batch is finite."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "semkine"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
import train as T                                                     # noqa: E402
from dt.infer_probe import combine                                    # noqa: E402

torch.set_grad_enabled(False)
cfg = load_config(str(REPO / "configs/trk6/g112kd_2k.yaml"))
ck = T.resolve_distill_ckpt(cfg)
teacher = T.load_distill_teacher(ck, cfg)
assert isinstance(teacher, T.EnsembleTeacher) and len(teacher.members) == 3
assert all(m.glimpse == 0 for m in teacher.members), "teachers must be full-frame S39 networks"
student = MNISTModel(cfg)
assert student.glimpse == 112
root = Path(cfg["DATA"]["ROOT"])
seq, d = sequences_for_split(root, "val_core")[1]
events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
ev_ch = EV.event_channels(cfg)
ends = [1999, 2049, 5049, 9049, 12049, 20049]
ends = [e for e in ends if np.isfinite(pos51[e]).all()]
x = torch.from_numpy(np.stack([ET.build_lnes(events, offsets, e, 200, ev_ch) for e in ends]))
prev = torch.from_numpy(np.stack([np.asarray(pos51[e - 50], np.float32) for e in ends]))
betas = torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1).expand(len(ends), -1)
K = torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3).expand(len(ends), -1, -1)
out = teacher(x, prev, betas=betas, camera_K=K)
worst = 0.0
for i in range(len(ends)):
    outs = [m(x[i:i + 1], prev[i:i + 1], betas=betas[i:i + 1], camera_K=K[i:i + 1]) for m in teacher.members]
    ref, _ = combine(outs)
    worst = max(worst, float((ref - out[i:i + 1]).abs().max()))
print("rows", len(ends), "max |batched - combine|", worst)
assert worst < 1e-5
st = student(x, prev, betas=betas, camera_K=K)
term = student._compute_loss(st, out, betas)[0]
print("distillation term on an untrained student", float(term))
assert torch.isfinite(term)
print("GATE PASS")
