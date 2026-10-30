"""Real MANO engineering contracts, independent of task/scientific scores."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tools.s43_rel.core import RelConfig, RelationMemory
from tools.s43_rel import geometry as geo


def conditions():
    pose=np.zeros(51,dtype=np.float64)
    pose[:3]=(.005,-.015,.6)
    pose[3:6]=(.15,-.12,.08)
    pose[6:]=np.random.default_rng(43).normal(0,.07,45)
    return pose,np.linspace(-.2,.2,10),np.array([[300.,0.,119.5],[0.,300.,89.5],[0.,0.,1.]])


def signed_union_distance(points, vertices, K, config):
    """Independent signed distances to a freshly rebuilt projected union."""
    xy=geo._project(vertices,K)
    edges,seg,_=geo._silhouette(xy,config)
    ab=edges[seg[:,0].astype(int)]
    v0=xy[ab[:,0]]; direction=xy[ab[:,1]]-v0
    a=v0+seg[:,1,None]*direction
    b=v0+seg[:,2,None]*direction
    d=b-a
    rel=points[:,None]-a[None]
    alpha=np.clip(np.einsum('psi,si->ps',rel,d)/np.sum(d*d,axis=1),0,1)
    dist=np.sqrt(np.min(np.sum((rel-alpha[:,:,None]*d)**2,axis=2),axis=1))
    tri=xy[geo._constants()['f']]
    inside=np.zeros(len(points),dtype=bool)
    for t in tri:
        edges2=np.roll(t,-1,axis=0)-t
        r=points[:,None]-t[None]
        side=edges2[None,:,0]*r[:,:,1]-edges2[None,:,1]*r[:,:,0]
        inside|=np.all(side>=0,axis=1)|np.all(side<=0,axis=1)
    return np.where(inside,-dist,dist)


class RealManoGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prev,cls.beta,cls.K=conditions()
        cls.mem=RelationMemory(RelConfig())
        cls.geom=geo.ManoGeometry()
        cls.binding=cls.geom.bind(cls.prev,cls.beta,cls.K,cls.mem.topology)

    def test_fk_matches_unmodified_real_mano_and_mean_once(self):
        name='_s43_rel_geometry_test_readonly_mano'
        spec=importlib.util.spec_from_file_location(name,geo.ROOT/'model/mano_layer.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mano=module.ManoLayer(geo.ASSET,add_mean=True,dtype=torch.float64).eval()
        p=torch.from_numpy(self.prev[None]);b=torch.from_numpy(self.beta[None])
        with torch.no_grad():
            reference,_=mano(b,p[:,3:6],p[:,6:],p[:,:3])
        vertices=geo.mano_kinematics(self.prev,self.beta,jacobian=False)
        # The reference AA Exp uses norm(v+1e-8); our true Exp removes that
        # numerical regularizer while preserving the identical MANO convention.
        np.testing.assert_allclose(vertices,reference.numpy()[0],atol=2e-8,rtol=0)

    def test_analytic_local_so3_actions_match_all_51_finite_differences(self):
        _,jac=geo.mano_kinematics(self.prev,self.beta)
        eps=1e-4
        for j in range(51):
            d=np.eye(51)[j]*eps
            plus=geo.mano_kinematics(self.prev,self.beta,delta=d,jacobian=False)
            minus=geo.mano_kinematics(self.prev,self.beta,delta=-d,jacobian=False)
            np.testing.assert_allclose((plus-minus)/(2*eps),jac[j],atol=3e-10,rtol=2e-7)

    def test_normal_action_matches_signed_distance_shape_derivative(self):
        binding=self.binding
        valid=binding.branch_regular
        self.assertGreater(valid.sum(),20)
        eps=1e-4
        directions=np.eye(51)[[0,3,6,15,31,50]]
        directions=np.concatenate((directions,np.random.default_rng(44).normal(size=(2,51))*.2))
        for direction in directions:
            plus=geo.mano_kinematics(self.prev,self.beta,delta=eps*direction,jacobian=False)
            minus=geo.mano_kinematics(self.prev,self.beta,delta=-eps*direction,jacobian=False)
            dp=signed_union_distance(binding.uv[valid],plus,self.K,self.geom.config)
            dm=signed_union_distance(binding.uv[valid],minus,self.K,self.geom.config)
            fd=-(dp-dm)/(2*eps)
            np.testing.assert_allclose(fd,binding.a[valid]@direction,atol=1e-5,rtol=3e-5)

    def test_equivalent_rotations_near_pi_have_identical_geometry_and_actions(self):
        p=self.prev.copy();q=p.copy()
        axis=np.array([.3,.4,.5]);axis/=np.linalg.norm(axis)
        p[3:6]=axis*(np.pi+1e-7)
        q[3:6]=-axis*(np.pi-1e-7)
        mean=geo._constants()['hands_mean']
        p[12:15]=axis*(np.pi+2e-7)-mean[6:9]
        q[12:15]=-axis*(np.pi-2e-7)-mean[6:9]
        vp,jp=geo.mano_kinematics(p,self.beta)
        vq,jq=geo.mano_kinematics(q,self.beta)
        np.testing.assert_allclose(vp,vq,atol=1e-14,rtol=0)
        np.testing.assert_allclose(jp,jq,atol=1e-14,rtol=0)
        bp=self.geom.bind(p,self.beta,self.K,self.mem.topology)
        bq=self.geom.bind(q,self.beta,self.K,self.mem.topology)
        np.testing.assert_allclose(bp.uv,bq.uv,atol=1e-9,rtol=0)
        np.testing.assert_allclose(bp.a,bq.a,atol=1e-9,rtol=0)

    def test_snapshot_read_matches_literal_ordered_Z_and_keeps_support(self):
        b=self.binding
        rng=np.random.default_rng(45)
        Z=rng.uniform(0,.2,(len(b.edge_ids),2,2,3,2)).astype(np.float32)
        snap=SimpleNamespace(Z=Z,edge_ids=b.edge_ids,cutoff_ms=12.,version=7)
        result=b.read(snap)
        m=np.zeros((128,24));lam=np.zeros((128,24))
        for i in range(len(b.edge_slot)):
            edge,point=b.edge_slot[i],b.point_index[i]
            mass=Z[edge].ravel();co=b.normal_dot_delta[i];w=b.weights[i]
            m[point]+=w*mass*co;lam[point]+=w*mass*co*co
        np.testing.assert_allclose(result['m'],m,atol=4e-6,rtol=4e-6)
        np.testing.assert_allclose(result['lam'],lam,atol=4e-6,rtol=4e-6)
        np.testing.assert_allclose(result['f'],b.a.T@m,atol=2e-5,rtol=4e-5)
        self.assertEqual(result['f'].shape,(51,24))
        self.assertTrue(np.all(result['lam']>=0))
        # A PSD Gram remains a factor pair, never a covariance or inverse.
        v=rng.normal(size=51)
        quadratic=np.sum(result['lam']*(b.a@v)[:,None]**2,axis=0)
        self.assertTrue(np.all(quadratic>=0))

    def test_antisymmetric_cross_polarity_slots_are_not_merged(self):
        b=self.binding
        e=int(b.edge_slot[np.argmax(np.abs(b.normal_dot_delta))])
        Z=np.zeros((len(b.edge_ids),2,2,3,2),np.float32)
        Z[e,0,1,0,0]=1.
        result=b.read(SimpleNamespace(Z=Z,edge_ids=b.edge_ids))
        self.assertGreater(np.linalg.norm(result['lam'][:,6]),0)
        self.assertEqual(np.linalg.norm(result['lam'][:,12]),0)

    def test_sparse_weights_are_k_nearest_radius_normalized_per_edge(self):
        b=self.binding
        self.assertLess(np.bincount(b.edge_slot).max(),5)
        np.testing.assert_allclose(np.bincount(b.edge_slot,b.weights),1.,atol=1e-12)
        d=np.linalg.norm(self.mem.topology.midpoint_px[b.edge_ids[b.edge_slot]]-b.uv[b.point_index],axis=1)
        self.assertTrue(np.all(d<4))
        self.assertTrue(np.all(np.abs(b.normal_dot_delta)<=1+1e-14))
        self.assertLess(len(b.edge_ids),self.mem.topology.n_edges)

    def test_zero_support_and_invalid_Z_contract(self):
        b=self.binding
        Z=np.zeros((len(b.edge_ids),2,2,3,2),np.float32)
        snap=SimpleNamespace(Z=Z,edge_ids=b.edge_ids)
        r=b.read(snap)
        self.assertFalse(r['support'].any())
        self.assertFalse(r['f'].any())
        self.assertFalse(r['lam'].any())
        Z[0,0,0,0,0]=np.nan
        with self.assertRaises(ValueError):b.read(snap)
        Z[0,0,0,0,0]=-1
        with self.assertRaises(ValueError):b.read(snap)
        with self.assertRaises(ValueError):
            b.read(SimpleNamespace(Z=np.zeros((len(b.edge_ids),24)),edge_ids=b.edge_ids))
        with self.assertRaises(ValueError):
            b.read(SimpleNamespace(Z=np.zeros((len(b.edge_ids),2,2,3,2)),edge_ids=b.edge_ids[::-1]))

    def test_geometry_rebind_changes_query_but_not_sensor_prefix(self):
        mem=RelationMemory(RelConfig(dtype='float64'),seq_id='geometry-unit')
        mem.reset('geometry-unit',boundary_ms=0)
        topology=mem.topology
        e=int(self.binding.edge_ids[len(self.binding.edge_ids)//2])
        src,dst=topology.centers_px[[topology.src[e],topology.dst[e]]].astype(int)
        events=np.array([[*src,1,0],[*dst,3,1],[*src,5,1]],dtype=np.int64)
        mem.ingest(events,watermark_ms=6,packet_ordinal=0)
        before=mem.readout(6)
        b=self.geom.bind(self.prev,self.beta,self.K,topology)
        first=b.read(mem.readout(6,edge_ids=b.edge_ids))
        moved=self.prev.copy();moved[0]+=.02
        other=self.geom.bind(moved,self.beta,self.K,topology)
        other.read(mem.readout(6,edge_ids=other.edge_ids))
        again=b.read(mem.readout(6,edge_ids=b.edge_ids))
        after=mem.readout(6)
        np.testing.assert_array_equal(before.A,after.A)
        np.testing.assert_array_equal(before.Z,after.Z)
        np.testing.assert_array_equal(first['f'],again['f'])
        self.assertNotEqual(b.metadata['geometry_version'],other.metadata['geometry_version'])
        self.assertFalse(np.array_equal(b.uv,other.uv))

    def test_invalid_geometry_reports_error_and_offscreen_support_is_zero(self):
        for bad in (np.full(51,np.nan),np.zeros(51)):
            with self.assertRaises(ValueError):self.geom.bind(bad,self.beta,self.K,self.mem.topology)
        p=self.prev.copy();p[0]=10.
        binding=self.geom.bind(p,self.beta,self.K,self.mem.topology)
        self.assertEqual(len(binding.edge_ids),0)
        pack=binding.read(SimpleNamespace(Z=np.zeros((0,2,2,3,2)),edge_ids=binding.edge_ids))
        self.assertFalse(pack['f'].any())
        self.assertFalse(pack['support'].any())


if __name__=='__main__':
    unittest.main()
