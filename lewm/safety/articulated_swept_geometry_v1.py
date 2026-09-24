"""Deterministic articulated swept-geometry primitives for the H1 assay.

All quaternions use Genesis' wxyz convention.  Environment geometry is an
oriented box; Go2 collision geometry is represented by its URDF sphere,
capsule, or box primitive.  Capsule/box separation uses a fixed dense
centerline quadrature (33 samples), so distances are deterministic but have a
documented axial discretisation bound of length/64.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable

import numpy as np

CAPSULE_CENTERLINE_SAMPLES = 33


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, np.float64)
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], np.float64)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, np.float64); bw, bx, by, bz = np.asarray(b, np.float64)
    return np.asarray([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz,
        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], np.float64)


def inverse_transform(parent_pos, parent_quat, child_pos, child_quat):
    r = rotation(parent_quat)
    local_pos = r.T @ (np.asarray(child_pos)-np.asarray(parent_pos))
    inverse = np.asarray([parent_quat[0], -parent_quat[1], -parent_quat[2], -parent_quat[3]], np.float64)
    return local_pos, quaternion_multiply(inverse, child_quat)


def compose(parent_pos, parent_quat, local_pos, local_quat):
    return np.asarray(parent_pos)+rotation(parent_quat)@np.asarray(local_pos), quaternion_multiply(parent_quat, local_quat)


def box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray, yaw_or_rotation) -> np.ndarray:
    points = np.asarray(points, np.float64)
    if np.asarray(yaw_or_rotation).ndim == 0:
        yaw = float(yaw_or_rotation); c, s = math.cos(yaw), math.sin(yaw)
        r = np.asarray([[c,-s,0],[s,c,0],[0,0,1]], np.float64)
    else:
        r = np.asarray(yaw_or_rotation, np.float64)
    local = (points-np.asarray(center))@r
    q = np.abs(local)-np.asarray(half)
    return np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(np.max(q, axis=-1), 0)


def primitive_points(kind: str, data: np.ndarray, samples: int = CAPSULE_CENTERLINE_SAMPLES) -> tuple[np.ndarray, float]:
    data = np.asarray(data, np.float64)
    if kind == "sphere":
        return np.zeros((1, 3), np.float64), float(data[0])
    if kind == "capsule":
        radius, length = float(data[0]), float(data[1])
        z = np.linspace(-length/2, length/2, samples)
        return np.stack((np.zeros_like(z), np.zeros_like(z), z), -1), radius
    if kind == "box":
        half = data[:3]
        return np.asarray([[sx*half[0], sy*half[1], sz*half[2]]
            for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], np.float64), 0.0
    raise ValueError(kind)


def primitive_to_box(kind: str, data: np.ndarray, pos: np.ndarray, quat: np.ndarray,
                     box_center: np.ndarray, box_half: np.ndarray, box_yaw: float) -> float:
    """Conservative signed primitive/OBB clearance.

    Sphere and sampled capsule values are signed.  Box/box overlap is tested
    by the exact 15-axis SAT; separated distance uses symmetric vertex SDFs.
    """
    r = rotation(quat); points, radius = primitive_points(kind, data)
    world = points@r.T+pos
    forward = float(np.min(box_sdf(world, box_center, box_half, box_yaw))-radius)
    if kind != "box":
        return forward
    yaw = float(box_yaw); c, s = math.cos(yaw), math.sin(yaw)
    rb = np.asarray([[c,-s,0],[s,c,0],[0,0,1]], np.float64)
    corners = np.asarray([[sx*box_half[0], sy*box_half[1], sz*box_half[2]]
        for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], np.float64)@rb.T+box_center
    reverse = float(np.min(box_sdf(corners, pos, np.asarray(data[:3]), r)))
    # Exact OBB overlap via separating axes. Negative value is conservative
    # minimum projected overlap, adequate as a signed score.
    axes = [r[:,i] for i in range(3)]+[rb[:,i] for i in range(3)]
    axes += [np.cross(r[:,i], rb[:,j]) for i in range(3) for j in range(3)]
    overlap = math.inf; delta = np.asarray(box_center)-np.asarray(pos)
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm < 1e-10: continue
        axis = axis/norm
        ra = float(np.sum(np.asarray(data[:3])*np.abs(r.T@axis)))
        rb_extent = float(np.sum(np.asarray(box_half)*np.abs(rb.T@axis)))
        gap = abs(float(delta@axis))-(ra+rb_extent)
        if gap > 0: return max(0.0, min(forward, reverse))
        overlap = min(overlap, -gap)
    return -float(overlap)


def primitive_to_points(kind: str, data: np.ndarray, pos: np.ndarray, quat: np.ndarray,
                        points: np.ndarray) -> float:
    if not len(points): return math.inf
    local = (np.asarray(points, np.float64)-np.asarray(pos))@rotation(quat)
    data = np.asarray(data, np.float64)
    if kind == "sphere": return float(np.min(np.linalg.norm(local, axis=1)-data[0]))
    if kind == "capsule":
        z = np.clip(local[:,2], -data[1]/2, data[1]/2)
        delta = local-np.stack((np.zeros_like(z),np.zeros_like(z),z),-1)
        return float(np.min(np.linalg.norm(delta,axis=1)-data[0]))
    if kind == "box": return float(np.min(box_sdf(local, np.zeros(3), data[:3], np.eye(3))))
    raise ValueError(kind)


def min_scene_clearance(primitives: Iterable[dict], centers: np.ndarray, halves: np.ndarray,
                        yaws: np.ndarray, broad_radius: float = 2.0) -> tuple[float,int,int]:
    best, best_primitive, best_object = math.inf, -1, -1
    for pi, primitive in enumerate(primitives):
        pos = np.asarray(primitive["pos"], np.float64)
        nearby = np.flatnonzero(np.linalg.norm(centers[:,:2]-pos[:2],axis=1) <= broad_radius+
            np.linalg.norm(halves[:,:2],axis=1))
        for oi in nearby:
            value = primitive_to_box(primitive["kind"], primitive["data"], pos, primitive["quat"],
                                     centers[oi], halves[oi], float(yaws[oi]))
            if value < best: best, best_primitive, best_object = value, pi, int(oi)
    return float(best), best_primitive, best_object


def min_point_clearance(primitives: Iterable[dict], points: np.ndarray) -> tuple[float,int]:
    best, best_primitive = math.inf, -1
    points = np.asarray(points, np.float64)
    for pi, primitive in enumerate(primitives):
        pos = np.asarray(primitive["pos"], np.float64)
        subset = points[np.linalg.norm(points-pos,axis=1)<1.25]
        value = primitive_to_points(primitive["kind"], primitive["data"], pos, primitive["quat"], subset)
        if value < best: best, best_primitive = value, pi
    return float(best), best_primitive
