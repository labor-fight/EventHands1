"""Differentiable MANO51 right-local action decoder, independent of readers.

Training uses actual rotation matrices through full MANO LBS/FK, no Log.
The detached residual-axis-angle chart is only a commit/save representation.
No targets, camera, learned prior/loss, clamps or extra mechanism are included.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'assets/mano_right.npz'


def rotation_exp(v):
    """SO(3) Exp with finite first derivatives at exactly zero; no0/0branch."""
    theta=torch.linalg.vector_norm(v,dim=-1)
    a=torch.sinc(theta/torch.pi)
    b=.5*torch.sinc(theta/(2*torch.pi)).square()
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(v.shape[:-1]+(3,3))
    I=torch.eye(3,dtype=v.dtype,device=v.device)
    return I+a[...,None,None]*K+b[...,None,None]*(K@K)


@dataclass(frozen=True)
class DecodedMano:
    vertices:torch.Tensor       # [B,778,3] metres
    joints:torch.Tensor         # [B,21,3] OpenPose order, metres
    rotations:torch.Tensor      # [B,16,3,3] actual local rotations includingroot
    translation:torch.Tensor    # [B,3] metres
    raw_action:torch.Tensor     # [B,51] dt3,rootomega3,localomega45
    anchor51:torch.Tensor       # [B,51] t3/rootAA3/residual45
    _hands_mean:torch.Tensor

    def commit_pose51(self):
        """Detached chart only; identity rows retain exact original chart.

        Nonzero rows use robust principal SO(3) Log, including nearpi. Matrix
        equivalence, not a particular AA branch, defines the physical state.
        This function is never part of the differentiable FK path.
        """
        identity=(self.raw_action.detach()==0).all(-1)
        if bool(identity.all()):return self.anchor51.detach().clone()
        from scipy.spatial.transform import Rotation
        r=self.rotations.detach().to(device='cpu',dtype=torch.float64).numpy()
        chart=Rotation.from_matrix(r.reshape(-1,3,3)).as_rotvec().reshape(-1,48)
        chart[:,3:]-=self._hands_mean.detach().cpu().double().numpy().reshape(1,45)
        value=torch.cat((self.translation.detach(),torch.as_tensor(chart,device=self.translation.device,dtype=self.translation.dtype)),1)
        return torch.where(identity[:,None],self.anchor51.detach(),value).detach()


class FieldManoDecoder(nn.Module):
    """Fixedasset MANO matrix-FK, no trainable parameters.

    Forward(anchor51[B,51],beta[B,10],raw_action[B,51])->DecodedMano.
    Single unbatched inputs/batch1 may broadcast. Outputs always retainB.
    Rawdt scale=.01m; all16 rightlocal rotation vectors scale=.1rad.
    Input tensors move to this module's buffer device/dtype; autocast disabled
    for FK. FP32/FP64 are supported. Anchors/history should be detached by the
    caller when TBPTT1 is intended; this module does not silently detach them.
    """
    translation_scale_m=.01
    rotation_scale_rad=.1

    def __init__(self,asset=ASSET,*,dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32,torch.float64):raise ValueError('decoder FP32/FP64 required')
        self.asset_path=str(Path(asset).resolve())
        with np.load(asset,allow_pickle=False) as z:
            for name in ('v_template','shapedirs','posedirs','J_regressor','weights','hands_mean'):
                self.register_buffer(name,torch.tensor(np.array(z[name]),dtype=dtype))
            for name,source in [('tip_ids','fingertip_vertex_ids'),('joint_reorder','mano16_tips_to_openpose21'),('faces','f')]:
                self.register_buffer(name,torch.tensor(np.array(z[source]),dtype=torch.long))
            parents=np.array(z['kintree_table'][0],np.int64);parents[0]=-1
        self.parents=tuple(map(int,parents))
        self.asset_sha256=hashlib.sha256(Path(asset).read_bytes()).hexdigest()

    def _input(self,value,width,name):
        if not isinstance(value,torch.Tensor):value=torch.tensor(np.asarray(value),device=self.v_template.device,dtype=self.v_template.dtype)
        else:value=value.to(device=self.v_template.device,dtype=self.v_template.dtype)
        if value.ndim==1:value=value.unsqueeze(0)
        if value.ndim!=2 or value.shape[1]!=width or not bool(torch.isfinite(value).all()):
            raise ValueError(f'finite {name}[B,{width}] required')
        return value

    def anchor_rotations(self,anchor51):
        p=self._input(anchor51,51,'anchor51')
        actual=torch.cat((p[:,3:6],p[:,6:]+self.hands_mean[None]),1).reshape(-1,16,3)
        return rotation_exp(actual)

    def matrix_fk(self,translation,rotations,beta):
        """Full exact MANO formula with matrix rotations, localmeans already inR."""
        B=len(beta)
        shaped=self.v_template[None]+torch.einsum('bl,vkl->bvk',beta,self.shapedirs)
        restJ=torch.einsum('jv,bvk->bjk',self.J_regressor,shaped)
        I=torch.eye(3,device=beta.device,dtype=beta.dtype)
        feature=(rotations[:,1:]-I).reshape(B,135)
        posed=shaped+feature.matmul(self.posedirs.reshape(2334,135).T).reshape(B,778,3)
        relative=torch.stack([restJ[:,0]]+[restJ[:,j]-restJ[:,self.parents[j]] for j in range(1,16)],1)
        top=torch.cat((rotations,relative[...,None]),-1)
        bottom=torch.tensor([0.,0.,0.,1.],device=beta.device,dtype=beta.dtype).reshape(1,1,1,4).expand(B,16,1,4)
        local=torch.cat((top,bottom),-2)
        transforms=[local[:,0]]
        for j in range(1,16):transforms.append(transforms[self.parents[j]]@local[:,j])
        T=torch.stack(transforms,1);posedJ=T[:,:,:3,3]
        corrected_translation=posedJ-(T[:,:,:3,:3]@restJ[...,None])[...,0]
        corrected=torch.cat((torch.cat((T[:,:,:3,:3],corrected_translation[...,None]),-1),bottom),-2)
        blend=torch.einsum('vj,bjkl->bvkl',self.weights,corrected)
        vertices=(blend[:,:,:3,:3]@posed[...,None])[...,0]+blend[:,:,:3,3]+translation[:,None]
        allJ=torch.cat((posedJ+translation[:,None],vertices[:,self.tip_ids]),1)
        return vertices,allJ[:,self.joint_reorder]

    def forward(self,anchor51,beta,raw_action=None):
        anchor=self._input(anchor51,51,'anchor51');beta=self._input(beta,10,'beta')
        raw=torch.zeros_like(anchor) if raw_action is None else self._input(raw_action,51,'raw_action')
        B=max(len(anchor),len(beta),len(raw))
        if any(len(a) not in (1,B) for a in (anchor,beta,raw)):raise ValueError('batch sizes mustmatch or be1')
        anchor=anchor.expand(B,-1);beta=beta.expand(B,-1);raw=raw.expand(B,-1)
        with torch.autocast(device_type=self.v_template.device.type,enabled=False):
            translation=anchor[:,:3]+self.translation_scale_m*raw[:,:3]
            R=self.anchor_rotations(anchor)@rotation_exp(self.rotation_scale_rad*raw[:,3:].reshape(B,16,3))
            vertices,joints=self.matrix_fk(translation,R,beta)
        return DecodedMano(vertices,joints,R,translation,raw,anchor,self.hands_mean)

    def identity(self,anchor51,beta):return self(anchor51,beta)

    def metadata(self):
        return dict(asset=self.asset_path,asset_sha256=self.asset_sha256,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            dtype=str(self.v_template.dtype),device=str(self.v_template.device),
            trainable_parameters=sum(p.numel() for p in self.parameters()),buffer_bytes=sum(v.numel()*v.element_size() for v in self.buffers()),
            input='anchor[t_m3,rootAA3,residual45]; rawaction[dt3,rootomega3,localomega45]',
            translation_scale_m=self.translation_scale_m,rotation_scale_rad=self.rotation_scale_rad,
            operation='Ractual=Exp(anchorAA+handmean once)@Exp(.1rawomega);t=anchort+.01rawdt',
            FK='full778vertex exactLBS+16kinematicjoints+5tips,OpenPose21 reorder; matrixpath noLog',
            save_commit='commit_pose51()detaches;zeroaction preservesanchor chart exactly;nonzero principalLog minusmean',
            validity='MANOkinematic manifold;no additional jointlimit/contact/visibility enforcement',
            targets_or_K_or_loss_read=False)
