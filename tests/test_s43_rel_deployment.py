"""Output physical units, translation and mean-hand-pose compatibility."""
from types import SimpleNamespace
import numpy as np
import torch
import pytest

from tools.s43_integration.model import load_legacy, DEFAULT_CKPT, DEFAULT_CFG
from tools.s43_rel.deployment import decode_surface


def test_terminal_surface_uses_original_fk_and_camera_meter_translation():
    torch.set_num_threads(1)
    base,_=load_legacy(DEFAULT_CKPT,DEFAULT_CFG,"cpu")
    model=SimpleNamespace(base=base)
    state=np.array([0.,0.,.6]+[0.]*48,np.float32)
    beta=np.linspace(-.1,.1,10,dtype=np.float32)
    surface,cost=decode_surface(model,state,beta)
    with torch.inference_mode():
        v,j=base.mano(torch.from_numpy(beta[None]),torch.zeros(1,3),
                      base.mano.hands_mean[None],torch.from_numpy(state[None,:3]))
    np.testing.assert_array_equal(surface["vertices_m"],v[0].numpy())
    np.testing.assert_array_equal(surface["joints_m"],j[0].numpy())
    shifted=state.copy();shifted[:3]+=[.005,-.015,.01]
    moved,_=decode_surface(model,shifted,beta)
    for key in ("vertices_m","joints_m"):
        np.testing.assert_allclose(moved[key]-surface[key],np.broadcast_to(shifted[:3]-state[:3],surface[key].shape),rtol=1e-5,atol=1e-7)
    assert all(value>=0 for value in cost.values())
    assert sum(value for key,value in cost.items() if key!="output_mesh_total_seconds")==pytest.approx(cost["output_mesh_total_seconds"])
    assert not any(parameter.grad is not None for parameter in base.parameters())


def test_terminal_surface_rejects_nonfinite_pose_before_output():
    base,_=load_legacy(DEFAULT_CKPT,DEFAULT_CFG,"cpu")
    with pytest.raises(ValueError,match="finite"):
        decode_surface(SimpleNamespace(base=base),np.full(51,np.nan,np.float32),np.zeros(10,np.float32))
