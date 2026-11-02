"""T1A-UB engineering amendment 1 (tools/t1aub/devfix.py): S45-SM's router cache per device. CPU only; `meta` stands in
for the second device (the failure on GPU 4 was CUDA after a CPU MAC count)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45sm as SM  # noqa: E402
from tools.t1aub import devfix  # noqa: E402


def _devices(net):
    return {p.device.type for p in net.parameters()}


def _fresh(monkeypatch):
    router = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    monkeypatch.setitem(SM._STATE, "net", router)
    monkeypatch.setitem(SM._STATE, "on", {})
    return router


def test_native_cache_moves_the_one_router(monkeypatch):
    devfix.uninstall()
    _fresh(monkeypatch)
    a = SM._net(torch.device("cpu"))
    SM._net(torch.device("meta"))
    assert SM._net(torch.device("cpu")) is a and _devices(a) == {"meta"}       # the failure: the cpu entry left cpu


def test_fixed_cache_keeps_one_copy_per_device(monkeypatch):
    router = _fresh(monkeypatch)
    devfix.install()
    try:
        x = torch.randn(5, 4)
        a = SM._net(torch.device("cpu"))
        b = SM._net(torch.device("meta"))
        assert a is not b and _devices(a) == {"cpu"} and _devices(b) == {"meta"} and _devices(router) == {"cpu"}
        assert SM._net(torch.device("cpu")) is a and torch.equal(a(x), router(x))
        for p, q in zip(a.parameters(), router.parameters()):
            assert torch.equal(p, q) and p is not q
    finally:
        devfix.uninstall()
    assert SM._net.__module__ == "semkine.s45sm"
