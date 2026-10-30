"""Equivalent CPU geometry for the constant-public-radiance B-R integration.

All B+1 samples are rendered.  This module does not use GT, RGB or events and
does not alter the old renderer or frozen model.  The fast triangle-union
backend is exact for the supported public constant albedo/constant background
model; it is NOT a general textured/depth renderer.  Constant radiance remains
an uncalibrated imaging assumption on recorded data.

Camera input is the existing auxiliary 640x480 camera matrix.  It is converted
once by .375 to the 240x180 event grid, then by the reference pixel-centre map.
The MANO layer is loaded by an isolated module alias, without modifying the
global ``model`` name used by legacy S43 loading code.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'assets/mano_right.npz'
OUT=ROOT/'outputs/s43_integration_20261027'
IMAGE_SIZE=(24,32)
AMPLITUDES=(-.015,0.,.015)
SCALES=(.8,1.,1.2)
DELTA=.12
FOREGROUND=.7
BACKGROUND=.15


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _mano():
    name='_s43_integration_mano_layer_readonly'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'model/mano_layer.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    model=sys.modules[name].ManoLayer(ASSET,add_mean=True).cpu().eval()
    model.fast_chain=True
    return model


@lru_cache(maxsize=1)
def _raster_library():
    source=Path(__file__).with_name('raster.cpp')
    flags=['-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off']
    key=hashlib.sha256(source.read_bytes()+repr(flags).encode()).hexdigest()[:20]
    build=Path(os.environ.get('S43_INTEGRATION_BUILD_DIR',str(OUT/'build')))
    build.mkdir(parents=True,exist_ok=True)
    library=build/f'raster_{key}.so'
    if not library.exists():
        temporary=build/f'raster_{key}_{os.getpid()}.so'
        command=['g++',*flags,str(source),'-o',str(temporary)]
        started=time.perf_counter()
        subprocess.run(command,check=True,capture_output=True,text=True)
        os.replace(temporary,library)
        receipt={'command':command,'source_sha256':_sha(source),'seconds':time.perf_counter()-started,
                 'library':str(library),'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0]}
        path=build/f'raster_{key}_compile.json'
        if not path.exists(): path.write_text(json.dumps(receipt,indent=2)+'\n')
    loaded=ctypes.CDLL(str(library))
    method=loaded.s43_constant_coverage
    method.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int64),
                     ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                     ctypes.POINTER(ctypes.c_uint8)]
    method.restype=ctypes.c_int
    return loaded


def _rot_exp(v):
    theta=float(np.linalg.norm(v));x,y,z=v
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    a=1.-theta*theta/6. if theta<1e-7 else np.sin(theta)/theta
    b=.5-theta*theta/24. if theta<1e-7 else (1.-np.cos(theta))/theta**2
    return np.eye(3)+a*skew+b*(skew@skew)


def _rot_log(R):
    theta=float(np.arccos(np.clip((np.trace(R)-1.)/2.,-1.,1.)))
    skew=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    if theta<1e-7:return .5*skew
    if np.pi-theta<1e-5:raise ValueError('rotation too close to pi for fixed candidate chart')
    return theta/(2.*np.sin(theta))*skew


def candidate_states(history,bins=50):
    """The reference root-x family, with one repeated-rotation roundtrip only.

    Every local/root rotation is constant along these paths.  Canonicalizing
    each once is the same reference left-retraction at zero rotational delta;
    it avoids 3*(B+1)*16 identical Exp/Log computations.  The same near-pi
    unsupported branch as the reference is explicit.
    """
    h=np.asarray(history,dtype=np.float64)
    if h.shape!=(51,) or not np.isfinite(h).all() or type(bins)!=int or not 1<=bins<=50:
        raise ValueError('finite history[51] and bins=1..50 required')
    h=h.copy();h[3:6]=_rot_log(_rot_exp(h[3:6]))
    mean=_mano().hands_mean.cpu().numpy().reshape(15,3)
    for j in range(15):
        sl=slice(6+3*j,9+3*j)
        h[sl]=_rot_log(_rot_exp(h[sl]+mean[j]))-mean[j]
    states=np.broadcast_to(h,(3,bins+1,51)).copy()
    states[:,:,0]+=np.asarray(AMPLITUDES)[:,None]*np.linspace(0.,1.,bins+1)[None]
    return states


def fk_geometry(states,betas):
    states=np.asarray(states,dtype=np.float64)
    betas=np.asarray(betas,dtype=np.float32)
    if states.ndim<1 or states.shape[-1]!=51 or not np.isfinite(states).all() or betas.shape!=(10,) or not np.isfinite(betas).all():
        raise ValueError('finite states[...,51], betas[10] required')
    lead=states.shape[:-1]
    flat=torch.from_numpy(np.array(states.reshape(-1,51),dtype=np.float32))
    beta=torch.from_numpy(np.broadcast_to(betas,(len(flat),10)).copy())
    with torch.no_grad(): verts,joints=_mano()(beta,flat[:,3:6],flat[:,6:],flat[:,:3])
    return verts.numpy().reshape(lead+(778,3)),joints.numpy().reshape(lead+(21,3))


def fk_joints(states,betas): return fk_geometry(states,betas)[1]


def event_intrinsics(K_cam):
    matrix=np.asarray(K_cam,dtype=np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all():raise ValueError('finite raw camera[3,3] required')
    matrix=matrix.copy();matrix[:2]*=.375
    return matrix


def low_intrinsics(K_cam,image_size=IMAGE_SIZE):
    camera=event_intrinsics(K_cam)
    h,w=image_size;sx,sy=w/240.,h/180.
    camera[0]*=sx;camera[1]*=sy
    camera[0,2]+=.5*sx-.5;camera[1,2]+=.5*sy-.5
    return camera


def render_states(states,betas,K_cam,image_size=IMAGE_SIZE):
    """Constant public intensities of all states, no temporal approximation."""
    if tuple(image_size)!=IMAGE_SIZE:raise ValueError('integration geometry fixes image_size=(24,32)')
    vertices,_=fk_geometry(states,betas)
    vertices=vertices.reshape(-1,778,3).astype(np.float64)
    if np.min(vertices[:,:,2])<=.03:raise ValueError('MANO intersects near plane; unsupported geometry')
    camera=low_intrinsics(K_cam,image_size)
    xyh=vertices@camera.T
    xy=np.ascontiguousarray(xyh[:,:,:2]/xyh[:,:,2:3],dtype=np.float64)
    faces=np.ascontiguousarray(_mano().f.cpu().numpy(),dtype=np.int64)
    if np.min(faces)<0 or np.max(faces)>=778:raise ValueError('invalid MANO face index')
    h,w=image_size
    mask=np.zeros((len(vertices),h,w),dtype=np.uint8)
    method=_raster_library().s43_constant_coverage
    status=method(xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                  len(vertices),778,len(faces),h,w,
                  mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    if status:raise RuntimeError(f'coverage renderer failed: {status}')
    intensity=np.where(mask,FOREGROUND,BACKGROUND)
    return intensity.reshape(np.asarray(states).shape[:-1]+(h,w))


def render_paths(history,betas,K_cam,bins=50,image_size=IMAGE_SIZE):
    """Return fixed states[K,B+1,51] and integer levels[K,M,P,B+1]."""
    states=candidate_states(history,bins)
    images=render_states(states,betas,K_cam,image_size=image_size)
    log_bg=np.log(BACKGROUND)
    levels=log_bg+np.asarray(SCALES)[None,:,None,None,None]*(np.log(images[:,None])-log_bg)
    levels=np.rint(levels/DELTA).astype(np.int32)
    levels=levels.transpose(0,1,3,4,2).reshape(3,3,-1,bins+1)
    return states,levels


def provider_metadata():
    return {'provider':'constant-public MANO triangle coverage, C++ exact specialization',
            'all_time_samples':True,'image_size':IMAGE_SIZE,'K_input_grid':(480,640),
            'camera_to_event_scale':.375,'pixel_map':'reference pixel-centre map',
            'foreground':FOREGROUND,'background':BACKGROUND,'delta_log':DELTA,'eta':SCALES,
            'GT_or_RGB_used':False,'texture_supported':False,'uncalibrated_recorded_camera':True,
            'source_sha256':_sha(__file__),'cpp_sha256':_sha(Path(__file__).with_name('raster.cpp'))}


def engineering_check(out=OUT/'geometry_engineering.json'):
    """Six no-data engineering cases against unmodified reference geometry."""
    from tools.s43_br import geometry as reference
    torch.set_num_threads(1)
    source_hashes={str(p.relative_to(ROOT)):_sha(p) for p in [Path(__file__),Path(__file__).with_name('raster.cpp'),
                        ROOT/'model/mano_layer.py',ROOT/'tools/s43_br/geometry.py',ASSET]}
    before_model_module=sys.modules.get('model')
    _mano();_raster_library()
    assert sys.modules.get('model') is before_model_module
    rng=np.random.default_rng(20261027)
    camera=np.array([[800.,0.,320.],[0.,800.,240.],[0.,0.,1.]])
    camera_event=camera.copy();camera_event[:2]*=.375
    cases=[]
    for i in range(6):
        history=np.zeros(51);history[:3]=[rng.uniform(-.035,.035),rng.uniform(-.025,.025),rng.uniform(.4,.8)]
        history[3:6]=rng.normal(0,.35,3);history[6:]=rng.normal(0,.14,45)
        betas=rng.normal(0,.35,10)
        start=time.perf_counter();states,levels=render_paths(history,betas,camera);fast_seconds=time.perf_counter()-start
        start=time.perf_counter();old_states=reference.candidate_paths(history,amplitudes=AMPLITUDES,axis='tx')
        old_levels=reference.render_paths(old_states,betas,camera_event,image_size=IMAGE_SIZE,texture_seed=None)
        reference_seconds=time.perf_counter()-start
        new_v,new_j=fk_geometry(states,betas);old_v,old_j=reference.fk_geometry(old_states,betas)
        differences=int(np.count_nonzero(levels-old_levels))
        item={'case':i,'fast_seconds':fast_seconds,'reference_seconds':reference_seconds,
              'max_state_error':float(np.max(np.abs(states-old_states))),
              'max_vertex_error_m':float(np.max(np.abs(new_v-old_v))),
              'max_joint_error_m':float(np.max(np.abs(new_j-old_j))),
              'integer_level_mismatches':differences,'level_shape':list(levels.shape)}
        cases.append(item)
        print(json.dumps(item),flush=True)
        if differences or item['max_state_error']>1e-12:
            raise AssertionError(f'fast geometry differs from reference: {item}')
    # Uneventful and image-offscreen paths still have all finite samples.
    offscreen=np.zeros(51);offscreen[:3]=[10.,10.,.6]
    _,levels=render_paths(offscreen,np.zeros(10),camera,bins=7)
    assert levels.shape==(3,3,768,8) and np.isfinite(levels).all()
    assert np.all(levels==np.rint(np.log(BACKGROUND)/DELTA))
    near=np.zeros(51);near[2]=-.3
    try:render_paths(near,np.zeros(10),camera)
    except ValueError:near_rejected=True
    else:raise AssertionError('near-plane violation not rejected')
    receipt={'scope':'no-data geometry engineering, not accuracy evidence','cases':cases,
             'fast_median_seconds':float(np.median([c['fast_seconds'] for c in cases])),
             'reference_median_seconds':float(np.median([c['reference_seconds'] for c in cases])),
             'no_global_model_alias_changed':True,'near_plane_rejected':near_rejected,
             'offscreen_finite':True,'source_sha256':source_hashes,
             'provider':provider_metadata(),'all_source_hashes_unchanged':all(_sha(ROOT/p)==v for p,v in source_hashes.items())}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT/'geometry_engineering.json')
    args=parser.parse_args();engineering_check(args.out)
