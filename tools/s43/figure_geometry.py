"""CPU-only MANO geometry for the S43 method illustration.

The mesh is generated from this repository's actual MANO right-hand asset, using
zero shape coefficients and 22% of its mean pose.  It is a schematic pose for a
method figure, not a measured pose or a real event-stream example.  ``xy`` and
``joints_xy`` use a common isotropic normalization into [0, 1], with y increasing
upward, fingers upward, and the thumb on the left.  ``depth`` remains in MANO's
native metric units; larger values face toward the viewer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = axis_angle / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def load_hand_geometry(repo_path: str | Path | None = None) -> dict:
    """Return actual MANO mesh, normalized projected mesh, and 21-joint skeleton.

    Main keys: ``xy`` (778, 2), ``depth`` (778,), ``faces`` (1538, 3),
    ``joints_xy`` (21, 2), and ``skeleton_edges`` (20, 2).  ``verts`` and
    ``joints`` hold the unnormalized posed 3-D MANO geometry; ``camera_vertices``
    and ``camera_joints`` are the common orthographic camera frame.
    """
    root = Path(repo_path) if repo_path else Path(__file__).resolve().parents[2]
    asset_path = root / "assets" / "mano_right.npz"
    with np.load(asset_path, allow_pickle=False) as asset:
        template = asset["v_template"].copy()
        rest_joints = asset["J_regressor"] @ template
        local_pose = asset["hands_mean"].reshape(15, 3) * 0.22
        rotations = np.stack([np.eye(3)] + [_rodrigues(p) for p in local_pose])
        pose_feature = (rotations[1:] - np.eye(3)).reshape(-1)
        posed_template = template + np.einsum("vcp,p->vc", asset["posedirs"], pose_feature)
        parents = asset["kintree_table"][0].astype(int)
        parents[0] = -1
        transforms = np.zeros((16, 4, 4))
        for i, parent in enumerate(parents):
            local = np.eye(4)
            local[:3, :3] = rotations[i]
            local[:3, 3] = rest_joints[i] - (rest_joints[parent] if parent >= 0 else 0)
            transforms[i] = transforms[parent] @ local if parent >= 0 else local
        joints16 = transforms[:, :3, 3].copy()
        skinning = transforms.copy()
        skinning[:, :3, 3] -= np.einsum("ijk,ik->ij", transforms[:, :3, :3], rest_joints)
        per_vertex = np.einsum("vj,jab->vab", asset["weights"], skinning)
        verts_h = np.c_[posed_template, np.ones(len(template))]
        verts = np.einsum("vab,vb->va", per_vertex, verts_h)[:, :3]
        tips = verts[asset["fingertip_vertex_ids"]]
        joints = np.r_[joints16, tips][asset["mano16_tips_to_openpose21"]]
        faces = asset["f"].astype(int).copy()

    # A hand-relative orthographic camera preserves the open silhouette.  A
    # slight 12-degree oblique view gives the depth panel a visible gradient.
    up = joints[12] - joints[0]
    up /= np.linalg.norm(up)
    right = joints[17] - joints[5]
    right -= up * np.dot(right, up)
    right /= np.linalg.norm(right)
    forward = np.cross(right, up)
    forward /= np.linalg.norm(forward)
    angle = np.deg2rad(12.0)
    right, forward = (
        np.cos(angle) * right + np.sin(angle) * forward,
        -np.sin(angle) * right + np.cos(angle) * forward,
    )
    camera_basis = np.stack([right, up, forward], axis=1)
    camera_vertices = (verts - joints[0]) @ camera_basis
    camera_joints = (joints - joints[0]) @ camera_basis
    low, high = camera_vertices[:, :2].min(0), camera_vertices[:, :2].max(0)
    center = (low + high) / 2
    scale = 0.88 / float(np.max(high - low))
    xy = (camera_vertices[:, :2] - center) * scale + 0.5
    joints_xy = (camera_joints[:, :2] - center) * scale + 0.5
    edges = [(0, base) for base in (1, 5, 9, 13, 17)]
    edges += [(base + k, base + k + 1) for base in (1, 5, 9, 13, 17) for k in range(3)]
    return {
        "xy": xy,
        "depth": camera_vertices[:, 2].copy(),
        "faces": faces,
        "verts": verts,
        "vertices": verts,
        "joints": joints,
        "joints_xy": joints_xy,
        "joints_depth": camera_joints[:, 2].copy(),
        "skeleton_edges": np.asarray(edges, dtype=int),
        "camera_vertices": camera_vertices,
        "camera_joints": camera_joints,
        "asset_path": str(asset_path),
        "illustration_note": "Actual MANO mesh in a schematic pose; not a measured event sample.",
    }
