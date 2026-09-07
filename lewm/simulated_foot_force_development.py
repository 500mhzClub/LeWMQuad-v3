"""Acquisition-side ideal transducers; no terrain-role input or output."""
import numpy as np

from lewm.foot_load_sensor_development import FEET,IdealFootForceSample

FIELDS=frozenset(('geom_a','geom_b','force_a','force_b','valid_mask'))


def sample_ideal_foot_forces(packet,*,foot_geom_ids,rotation_world_from_foot,acquisition_identity,measured_ns,available_ns):
    """Sum ALL contact sides incident on each foot, then rotate to its axes.

    Native orientations occur ONLY at acquisition, like a simulator IMU model.
    They are not returned to the consumer. Geometry IDs identify transducers;
    the other contacting object's role/identity must never select a load.
    None means acquisition missing; an explicitly empty manifold means zero
    resultant under the declared ideal sensor hypothesis.
    """
    ids=tuple(foot_geom_ids)
    if len(ids)!=4 or any(type(v) is not int or v<0 for v in ids) or len(set(ids))!=4:
        raise ValueError('four unique verified foot geometry IDs in declared canonical order required')
    rotations=np.asarray(rotation_world_from_foot,float)
    if (rotations.shape!=(4,3,3) or not np.isfinite(rotations).all() or
            not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-10,rtol=0) or
            not np.allclose(np.linalg.det(rotations),1,atol=1e-10,rtol=0)):
        raise ValueError('four proper acquisition-side sensor-frame rotations required')
    if packet is None:
        return IdealFootForceSample(acquisition_identity,measured_ns,available_ns,np.full((4,3),np.nan),np.zeros(4,bool),np.zeros(4,bool))
    if not isinstance(packet,dict) or set(packet)!=FIELDS: raise ValueError('only declared raw contact force/geometry fields accepted; no terrain labels')
    a,b=np.asarray(packet['geom_a']),np.asarray(packet['geom_b']); valid=np.asarray(packet['valid_mask'])
    fa,fb=np.asarray(packet['force_a'],float),np.asarray(packet['force_b'],float)
    if (a.ndim!=1 or b.shape!=a.shape or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu' or
            valid.shape!=a.shape or valid.dtype!=bool or fa.shape!=a.shape+(3,) or fb.shape!=fa.shape):
        raise ValueError('explicit aligned unbatched contact arrays required')
    if (np.any(a[valid]<0) or np.any(b[valid]<0) or np.any(a[valid]==b[valid]) or
            not np.isfinite(fa[valid]).all() or not np.isfinite(fb[valid]).all()):
        raise ValueError('valid contact sides need distinct nonnegative IDs and finite forces')
    world=np.zeros((4,3))
    for i,geom in enumerate(ids):
        world[i]=fa[valid&(a==geom)].sum(axis=0)+fb[valid&(b==geom)].sum(axis=0)
    local=np.einsum('fji,fj->fi',rotations,world)
    return IdealFootForceSample(acquisition_identity,measured_ns,available_ns,local,np.ones(4,bool),np.zeros(4,bool))
