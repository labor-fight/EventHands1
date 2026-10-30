"""Deployment MANO silhouette queries for the raw ordered relation Z.

The action is the normal shape derivative of a *polygonal projected mesh
union* on each regular silhouette-edge branch.  Union visibility intervals
are clipped analytically; n is its 2D outward normal.  Samples near corners,
occlusion intersections and degenerate faces are explicitly flagged.  This
piecewise mesh approximation is not a differentiable smooth MANO surface,
historical visibility, a texture gradient, or calibrated uncertainty.

The detached committed S43 pose is [translation metres, root AA, residual45].
The hand mean is added once.  Derivatives use right local SO(3) actions and
never differentiate an axis-angle chart or take a logarithm near pi.  One
matrix FK plus its 51 analytical action directions replaces 51 full renders.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / 'assets/mano_right.npz'
COMPILE_FLAGS = ('-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off')
_BUILD_LOCK = threading.Lock()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value, shape, name):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return out


def _readonly(array):
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class GeometryConfig:
    num_samples: int = 128
    radius_px: float = 4.
    nearest_k: int = 4
    translation_scale_m: float = .005
    rotation_scale_rad: float = .1
    width: int = 240
    height: int = 180
    near_plane_m: float = .03
    visibility_epsilon_px: float = 1e-9
    corner_margin_px: float = .25

    def __post_init__(self):
        if (type(self.num_samples) is not int or self.num_samples < 1 or
            type(self.nearest_k) is not int or self.nearest_k < 1):
            raise ValueError('positive integer samples and nearest_k required')
        values = (self.radius_px, self.translation_scale_m, self.rotation_scale_rad,
                  self.near_plane_m, self.visibility_epsilon_px, self.corner_margin_px)
        if not np.isfinite(values).all() or any(x <= 0 for x in values):
            raise ValueError('finite positive geometry scales required')
        if (self.width, self.height) != (240, 180):
            raise ValueError('this geometry contract uses event pixels 240x180')


@lru_cache(maxsize=1)
def _constants():
    with np.load(ASSET, allow_pickle=False) as z:
        names = ('v_template', 'shapedirs', 'posedirs', 'J_regressor', 'weights',
                 'f', 'kintree_table', 'hands_mean')
        data = {k: np.array(z[k], dtype=np.int64 if k in ('f', 'kintree_table') else np.float64)
                for k in names}
    data['parents'] = data['kintree_table'][0].astype(int).tolist()
    data['parents'][0] = -1
    # Keep oriented edges and their incident faces; winding is resolved in image.
    incident = {}
    for f, tri in enumerate(data['f']):
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1) % 3])
            incident.setdefault(tuple(sorted((a, b))), []).append((f, a, b))
    if any(len(v) > 2 for v in incident.values()):
        raise ValueError('nonmanifold MANO edges unsupported')
    data['incident'] = tuple(incident.values())
    data['edge_faces'] = np.array([[v[0][0], v[1][0] if len(v)==2 else -1]
                                   for v in incident.values()], dtype=np.int64)
    data['edge_orientation'] = np.array([[(v[0][1],v[0][2]),
                                  (v[1][1],v[1][2]) if len(v)==2 else (v[0][1],v[0][2])]
                                  for v in incident.values()], dtype=np.int64)
    return data


def rotation_exp(vectors):
    """Stable axis-angle Exp, including exactly zero and angles near pi."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[-1:] != (3,) or not np.isfinite(v).all():
        raise ValueError('finite [...,3] rotation vectors required')
    with np.errstate(over='ignore'):
        t = np.linalg.norm(v, axis=-1)
    if not np.isfinite(t).all():
        raise ValueError('rotation norm overflow')
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -v[..., 2], v[..., 1]
    K[..., 1, 0], K[..., 1, 2] = v[..., 2], -v[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -v[..., 1], v[..., 0]
    a = np.sinc(t / np.pi)
    b = .5 * np.sinc(t / (2*np.pi)) ** 2
    return np.eye(3) + a[..., None, None]*K + b[..., None, None]*(K @ K)


def event_intrinsics(K_cam):
    """Explicit legacy 640x480 camera -> event 240x180 conversion, once."""
    K = _finite(K_cam, (3, 3), 'legacy camera K').copy()
    K[:2] *= .375
    return K


def _decode(prev):
    full = np.concatenate((prev[3:6], prev[6:] + _constants()['hands_mean'])).reshape(16, 3)
    return prev[:3].copy(), rotation_exp(full)


def mano_kinematics(prev51, betas, config=GeometryConfig(), *, delta=None, jacobian=True):
    """MANO vertices and optional exact derivatives of the local SO(3) action.

    ``delta[51]`` is a dimensionless right-retraction, useful for engineering
    finite differences.  No residual-chart roundtrip/mean re-addition occurs.
    """
    prev = _finite(prev51, (51,), 'prev51')
    beta = _finite(betas, (10,), 'betas')
    t, R = _decode(prev)
    if delta is not None:
        delta = _finite(delta, (51,), 'delta')
        t += config.translation_scale_m*delta[:3]
        R = R @ rotation_exp(config.rotation_scale_rad*delta[3:].reshape(16, 3))
    c = _constants()
    vs = c['v_template'] + np.einsum('vkl,l->vk', c['shapedirs'], beta)
    J = c['J_regressor'] @ vs
    pose_dirs = c['posedirs'].reshape(778*3,135)
    vp = vs + (pose_dirs @ (R[1:]-np.eye(3)).reshape(135)).reshape(778,3)
    rel = J.copy()
    rel[1:] -= J[np.asarray(c['parents'][1:])]
    local = np.broadcast_to(np.eye(4), (16, 4, 4)).copy()
    local[:, :3, :3], local[:, :3, 3] = R, rel
    T = np.empty_like(local)
    T[0] = local[0]
    if jacobian:
        generators = np.array([[[0,0,0],[0,0,-1],[0,1,0]],
                               [[0,0,1],[0,0,0],[-1,0,0]],
                               [[0,-1,0],[1,0,0],[0,0,0]]], dtype=np.float64)
        dR = np.zeros((51,16,3,3))
        for j in range(16):
            dR[3+3*j:6+3*j,j] = config.rotation_scale_rad*(R[j] @ generators)
        dvp = (dR[:,1:].reshape(51,135) @ pose_dirs.T).reshape(51,778,3)
        dlocal = np.zeros((51,16,4,4))
        dlocal[:,:,:3,:3] = dR
        dT = np.empty_like(dlocal)
        dT[:,0] = dlocal[:,0]
    for j in range(1,16):
        parent = c['parents'][j]
        T[j] = T[parent] @ local[j]
        if jacobian:
            dT[:,j] = dT[:,parent] @ local[j] + T[parent] @ dlocal[:,j]
    joint_rel = T.copy()
    joint_rel[:,:3,3] -= np.einsum('jkl,jl->jk', T[:,:3,:3], J)
    blended = (c['weights'] @ joint_rel.reshape(16,16)).reshape(778,4,4)
    vertices = np.einsum('vkl,vl->vk', blended[:,:3,:3], vp) + blended[:,:3,3] + t
    if not jacobian:
        return vertices
    drel = dT.copy()
    drel[:,:,:3,3] -= np.einsum('qjkl,jl->qjk', dT[:,:,:3,:3], J)
    dblend = (c['weights'] @ drel.transpose(1,0,2,3).reshape(16,51*16)).reshape(778,51,4,4).transpose(1,0,2,3)
    dv = (np.einsum('qvkl,vl->qvk', dblend[:,:,:3,:3], vp) +
          np.einsum('vkl,qvl->qvk', blended[:,:3,:3], dvp) + dblend[:,:,:3,3])
    dv[:3] = 0.
    dv[np.arange(3), :, np.arange(3)] = config.translation_scale_m
    return vertices, dv


def _project(vertices, K, derivative=None):
    h = vertices @ K.T
    xy = h[:,:2] / h[:,2:3]
    if derivative is None:
        return xy
    dh = derivative @ K.T
    J = (dh[:,:,:2] - xy[None]*dh[:,:,2:3]) / h[None,:,2:3]
    return xy, J


@lru_cache(maxsize=1)
def _visibility_backend():
    source = Path(__file__).with_name('geometry_cpu.cpp')
    flags = list(COMPILE_FLAGS)
    key = hashlib.sha256(source.read_bytes()+repr(flags).encode()).hexdigest()[:20]
    folder = Path(os.environ.get('S43_REL_BUILD_DIR', str(ROOT/'outputs/s43_rel/build')))
    folder.mkdir(parents=True, exist_ok=True)
    library = folder/f'geometry_{key}.so'
    with _BUILD_LOCK:
        if not library.exists():
            temporary = folder/f'geometry_{key}_{os.getpid()}.so'
            subprocess.run(['g++', *flags, str(source), '-o', str(temporary)],
                           check=True, capture_output=True, text=True)
            os.replace(temporary, library)
    lib = ctypes.CDLL(str(library))
    f = lib.s43_rel_visible_segments
    f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
                  ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                  ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    f.restype = ctypes.c_int
    return lib


def _silhouette(xy, config):
    c = _constants()
    tri = xy[c['f']]
    ab,ac=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    area = ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    fi=c['edge_faces']
    sides=area[fi]
    boundary=fi[:,1]<0
    degenerate=int(np.any(np.abs(sides)<=config.visibility_epsilon_px,axis=1).sum())
    candidate=boundary|(sides[:,0]*sides[:,1]<0)
    ci=np.flatnonzero(candidate)
    pick=((sides[ci,0]<=0)&(sides[ci,1]>0)&~boundary[ci]).astype(int)
    edges=c['edge_orientation'][ci,pick].copy()
    chosen=sides[ci,pick]
    flip=chosen<0
    edges[flip]=edges[flip,::-1]
    keep=(np.abs(chosen)>config.visibility_epsilon_px)&(np.linalg.norm(xy[edges[:,1]]-xy[edges[:,0]],axis=1)>1e-8)
    edges=np.ascontiguousarray(edges[keep],dtype=np.int64)
    if not len(edges):
        return edges, np.empty((0,3)), degenerate
    capacity = max(16, 8*len(edges))
    while True:
        output = np.empty((capacity,3), dtype=np.float64)
        n = _visibility_backend().s43_rel_visible_segments(
            np.ascontiguousarray(xy).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c['f'].ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(c['f']),
            edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(edges),
            config.width, config.height, config.visibility_epsilon_px,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), capacity)
        if n >= 0:
            return edges, output[:n].copy(), degenerate
        capacity = -n


@dataclass(frozen=True)
class GeometryBinding:
    """Immutable geometry and sparse edge-point links for one committed state."""
    uv: np.ndarray
    n: np.ndarray
    a: np.ndarray
    edge_ids: np.ndarray
    edge_slot: np.ndarray
    point_index: np.ndarray
    weights: np.ndarray
    normal_dot_delta: np.ndarray
    branch_regular: np.ndarray
    metadata: dict

    def read(self, snapshot):
        """Contract raw Z, keeping ordered p/q slots and nonnegative support."""
        if not np.array_equal(np.asarray(snapshot.edge_ids), self.edge_ids):
            raise ValueError('snapshot edge IDs/order do not match bound geometry')
        Z = np.asarray(snapshot.Z)
        if Z.ndim != 5 or Z.shape[0] != len(self.edge_ids) or Z.shape[1:3] != (2,2):
            raise ValueError('raw Z[E,2,2,R,L] required; F is not a replacement')
        if not np.isfinite(Z).all() or np.any(Z < 0):
            raise ValueError('Z must be finite and nonnegative')
        H = int(np.prod(Z.shape[1:]))
        m = np.zeros((len(self.uv),H), dtype=np.float64)
        lam = np.zeros_like(m)
        if len(self.edge_slot):
            coeff = self.weights*self.normal_dot_delta
            # Sparse multiplication accumulates only E_q*k*H terms and avoids
            # materializing link-by-H repeated mass or a dense 51-by-E operator.
            shape=(len(self.uv),len(self.edge_ids))
            rows,cols=self.point_index,self.edge_slot
            m=csr_matrix((coeff,(rows,cols)),shape=shape) @ Z.reshape(len(Z),H)
            lam=csr_matrix((coeff*self.normal_dot_delta,(rows,cols)),shape=shape) @ Z.reshape(len(Z),H)
        f = self.a.T @ m
        if not all(np.isfinite(v).all() for v in (m,lam,f)):
            raise ValueError('relation geometry contraction overflow')
        support = np.any(lam > 0,axis=1)
        return {'f':f, 'a':self.a, 'lam':lam, 'lambda':lam, 'm':m,
                'support':support, 'branch_regular':self.branch_regular,
                'uv':self.uv, 'normal':self.n,
                'metadata':dict(self.metadata, cutoff_ms=getattr(snapshot,'cutoff_ms',None),
                                sensor_version=getattr(snapshot,'version',None),
                                h_order='p,q,tau,age; C-order')}


class ManoGeometry:
    def __init__(self, config=GeometryConfig()):
        self.config = config
        self._tree = None
        self._topology = None

    def bind_source(self, prev51, betas, K_cam, topology):
        """Reader-facing alternative: K_cam in legacy 640x480 pixels."""
        result=self.bind(prev51,betas,event_intrinsics(K_cam),topology)
        return GeometryBinding(result.uv,result.n,result.a,result.edge_ids,result.edge_slot,
                               result.point_index,result.weights,result.normal_dot_delta,
                               result.branch_regular,dict(result.metadata,
                               K_input_grid=[640,480],camera_to_event_scale=.375))

    def bind(self, prev51, betas, K_event, topology):
        """Rebuild global geometry using only deployed prev/K/beta, detached.

        K_event is already in the 240x180 event grid.  This method does not
        modify sensor evidence; read only the returned edge_ids from memory.
        """
        start = time.perf_counter()
        K = _finite(K_event,(3,3),'event K')
        if K[0,0] <= 0 or K[1,1] <= 0 or not np.allclose(K[2],[0,0,1]):
            raise ValueError('positive focal lengths and standard event K required')
        vertices,dv = mano_kinematics(prev51,betas,self.config)
        if not np.isfinite(vertices).all() or not np.isfinite(dv).all():
            raise ValueError('MANO geometry overflow')
        if vertices[:,2].min() <= self.config.near_plane_m:
            raise ValueError('MANO intersects near plane; invalid geometry, no hold policy')
        xy,J = _project(vertices,K,dv)
        if not np.isfinite(xy).all() or not np.isfinite(J).all():
            raise ValueError('MANO projection/Jacobian overflow')
        fk_done = time.perf_counter()
        edges,segments,degenerate = _silhouette(xy,self.config)
        lengths = (np.linalg.norm(xy[edges[segments[:,0].astype(int),1]]-
                                  xy[edges[segments[:,0].astype(int),0]],axis=1)*
                   (segments[:,2]-segments[:,1])) if len(segments) else np.empty(0)
        good = lengths > 1e-7
        segments,lengths=segments[good],lengths[good]
        Ns = self.config.num_samples
        uv,n,a = np.zeros((Ns,2)),np.zeros((Ns,2)),np.zeros((Ns,51))
        regular=np.zeros(Ns,dtype=bool)
        if len(lengths):
            cum=np.cumsum(lengths)
            samples=(np.arange(Ns)+.5)*cum[-1]/Ns
            si=np.searchsorted(cum,samples)
            offset=samples-np.concatenate(([0.],cum[:-1]))[si]
            frac=offset/lengths[si]
            alpha=segments[si,1]+frac*(segments[si,2]-segments[si,1])
            ab=edges[segments[si,0].astype(int)]
            tang=xy[ab[:,1]]-xy[ab[:,0]]
            n=np.stack((tang[:,1],-tang[:,0]),axis=1)/np.linalg.norm(tang,axis=1)[:,None]
            uv=(1-alpha[:,None])*xy[ab[:,0]]+alpha[:,None]*xy[ab[:,1]]
            Jpoint=(1-alpha[None,:,None])*J[:,ab[:,0]]+alpha[None,:,None]*J[:,ab[:,1]]
            a=np.einsum('li,qli->lq',n,Jpoint)
            regular=(np.minimum(offset,lengths[si]-offset)>self.config.corner_margin_px)
        silhouette_done = time.perf_counter()
        if topology is not self._topology:
            mid=np.asarray(topology.midpoint_px,dtype=np.float64)
            if mid.ndim != 2 or mid.shape[1] != 2 or not np.isfinite(mid).all():
                raise ValueError('finite topology midpoint_px[E,2] required')
            self._tree=cKDTree(mid)
            self._topology=topology
        if len(lengths):
            lists=self._tree.query_ball_point(uv,self.config.radius_px)
            point=np.repeat(np.arange(Ns),[len(v) for v in lists])
            eids=np.concatenate(lists).astype(np.int64) if len(point) else np.empty(0,np.int64)
            d2=np.sum((np.asarray(topology.midpoint_px)[eids]-uv[point])**2,axis=1)
            inside=d2 < self.config.radius_px**2
            point,eids,d2=point[inside],eids[inside],d2[inside]
            order=np.lexsort((point,d2,eids))
            point,eids,d2=point[order],eids[order],d2[order]
            starts=np.r_[0,np.flatnonzero(eids[1:]!=eids[:-1])+1] if len(eids) else np.empty(0,int)
            ranks=np.arange(len(eids))-np.repeat(starts,np.diff(np.r_[starts,len(eids)]))
            keep=ranks<self.config.nearest_k
            point,eids,d2=point[keep],eids[keep],d2[keep]
        else:
            point,eids,d2=np.empty(0,int),np.empty(0,int),np.empty(0)
        edge_ids,edge_slot=np.unique(eids,return_inverse=True)
        weights=(1.-np.sqrt(d2)/self.config.radius_px)**2
        if len(weights):
            weights/=np.bincount(edge_slot,weights,minlength=len(edge_ids))[edge_slot]
        delta=np.asarray(topology.delta_px,dtype=np.float64)[eids]
        norms=np.linalg.norm(delta,axis=1)
        if np.any(norms<=0) or not np.isfinite(delta).all():
            raise ValueError('finite nonzero topology edge deltas required')
        nd=np.einsum('li,li->l',n[point],delta/norms[:,None]) if len(point) else np.empty(0)
        end=time.perf_counter()
        fingerprint=hashlib.sha256(np.concatenate((_finite(prev51,(51,),'prev51'),
                                  _finite(betas,(10,),'betas'),K.ravel())).tobytes()+
                                  repr(self.config).encode()).hexdigest()
        meta={'geometry_version':fingerprint,'method':'projected-triangle-union normal shape derivative',
              'approximation':'piecewise mesh; corners/intersections flagged; no historical visibility',
              'GT_used':False,'K_grid':[240,180],'mean_added_once':True,'local_action':'right SO3',
              'contour_length_px':float(lengths.sum()),'visible_intervals':len(segments),
              'degenerate_incident_edges':degenerate,'regular_samples':int(regular.sum()),
              'selected_edges':len(edge_ids),'edge_point_links':len(point),
              'binding_array_bytes':sum(v.nbytes for v in (uv,n,a,edge_ids,edge_slot,point,weights,nd,regular)),
              'static_spatial_index_data_bytes':int(self._tree.data.nbytes+self._tree.indices.nbytes),
              'seconds':{'fk_jacobian':fk_done-start,'silhouette':silhouette_done-fk_done,
                         'spatial_binding':end-silhouette_done,'total_bind':end-start},
              'cost_scope':'global FK/J/visibility/rebind; sparse Z read separately',
              'support_semantics':'empirical nonnegative Gram factors, not covariance/Fisher'}
        return GeometryBinding(*[_readonly(v) for v in (uv,n,a,edge_ids,edge_slot,point,
                                                       weights,nd,regular)],meta)


def provider_metadata():
    import scipy
    return {'source_sha256':_sha(__file__),
            'cpp_sha256':_sha(Path(__file__).with_name('geometry_cpu.cpp')),
            'mano_asset_sha256':_sha(ASSET),
            'mano_reference_sha256':_sha(ROOT/'model/mano_layer.py'),
            'numpy':np.__version__,'scipy':scipy.__version__,
            'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
            'compiler_flags':list(COMPILE_FLAGS),
            'numpy_dtype':'float64 FK/J; sensor native read',
            'visibility':'projected triangle union, analytic strict-interior interval subtraction',
            'global_costs':['MANO FK','51 local-action analytic Jacobian','visibility','spatial rebind'],
            'query_costs':'sparse E_q*k*H contraction plus Ns*51*H; no 51xE tensor',
            'historical_visibility_exact':False,'calibrated_uncertainty':False}
