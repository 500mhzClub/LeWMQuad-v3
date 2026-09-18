"""Explain selected failed floor views from recorded public depth and poses."""
import json
import time

import cv2
import numpy as np
import torch

from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.body_projected_floor_geometry_development import T, FOCAL
from lewm.current_plane_floor_coverage_development import (
    CurrentPlaneCoverageGeometry, _RawValidQuadGeometry)
from lewm.local_inverse_depth_floor_development import local_depth
from lewm.multirate_routing_map_development import Geometry
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.replay_go2_no_early_release_map_entry_development import RecordedCurrentPlaneMap
from scripts import run_go2_frozen_readout_navigation_development as run


def quad_intersection(uv, x, y):
    """Exact convex separating-axis test for unit image quads (touch included)."""
    corners = np.stack((np.stack((x, y), -1), np.stack((x+1, y), -1),
        np.stack((x+1, y+1), -1), np.stack((x, y+1), -1)), axis=-2)
    edges = np.roll(uv, -1, axis=0)-uv
    axes = np.concatenate((np.array([[1., 0.], [0., 1.]]),
        np.stack((-edges[:, 1], edges[:, 0]), -1)))
    poly = uv@axes.T
    pixels = corners@axes.T
    return ((pixels.max(-2) >= poly.min(0)-1e-9)
        & (pixels.min(-2) <= poly.max(0)+1e-9)).all(-1)


def explain(packet, R, p, floor_height, cell, plane_available):
    depth, valid = packet['depth_m'], packet['valid']
    classifier = CurrentPlaneCoverageGeometry(plane_available)
    try:
        actual = classifier.floor_coverage(depth, valid, R, p, floor_height,
            cells=np.asarray([cell], dtype=int))
    finally:
        classifier.close()
    world = np.column_stack(((np.asarray(cell)+np.array([[0,0],[1,0],[1,1],[0,1]]))*.05,
        np.full(4, floor_height)))
    camera = ((world-p)@R-T[:3, 3])@T[:3, :3]
    z = camera[:, 2]
    uv = camera[:, :2]/np.maximum(z[:, None], 1e-12)*FOCAL+[319.5,239.5]
    visible = bool(((z >= .2)&(z <= 5.)).all() and (uv.min(0)-1e-9 >= 0).all()
        and (uv.max(0)+1e-9 < [639,479]).all())
    result = dict(covered=bool(actual['covered'][0]), fully_projected=visible,
        projected_corners_uv=uv.tolist(), optical_depth_m=z.tolist(),
        classifier='raw_valid_quads' if plane_available else 'local_depth_ground_mesh')
    if not visible:
        assert not result['covered']
        return result
    # Reconstruct the exact active classifier, including its unavailable-plane fallback.
    if not plane_available:
        depth, valid = local_depth(depth, valid)
    geometry = _RawValidQuadGeometry() if plane_available else Geometry()
    try:
        ground = geometry.index(depth, valid, R[2])['ground_cells']
        error = geometry.body_projection(depth)@R[2]+p[2]-floor_height
    finally:
        geometry.close()
    near = np.abs(error) <= .01
    height_good = near[:-1,:-1]&near[:-1,1:]&near[1:,:-1]&near[1:,1:]
    valid_quad = valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    a, b = actual['projected_lower_xy'][0], actual['projected_upper_xy'][0]
    y, x = np.mgrid[a[1]:b[1]+1, a[0]:b[0]+1]
    intersects = quad_intersection(uv, x, y)
    bad = ~(ground[y,x]&height_good[y,x])
    assert result['covered'] == (not bool(bad.any()))
    parts = {}
    for name, mask in (('intersects_projected_square', intersects),
            ('outside_projected_square', ~intersects)):
        xx, yy = x[mask], y[mask]
        height = np.stack((error[yy,xx], error[yy+1,xx],
            error[yy,xx+1], error[yy+1,xx+1]), -1)
        finite = height[np.isfinite(height)]
        parts[name] = dict(quads=int(mask.sum()), bad_quads=int(bad[mask].sum()),
            invalid_quads=int((~valid_quad[yy,xx]).sum()),
            ground_rejected_quads=int((~ground[yy,xx]).sum()),
            height_rejected_quads=int((~height_good[yy,xx]).sum()),
            signed_corner_height_error_mm_quantiles=(np.quantile(finite,[0,.05,.5,.95,1])*1000).tolist()
                if len(finite) else None)
    return result | dict(parts=parts,
        failure_only_outside_projected_square=bool(bad.any() and not bad[intersects].any()),
        reconstructed_rectangle_matches_active_classifier=True)


def main():
    if not (run.BASE/run.root_name(4)/'frozen_readout_navigation_readout_v1.json').exists():
        raise ValueError('finish the fixed native comparison before sensor replay')
    output = run.BASE/'go2_frozen_readout_floor_view_probe_v1.json'
    if output.exists():
        raise ValueError('preserve completed diagnostic')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1); configure()
    began = time.monotonic(); rows = []
    cases = {1: {828: [(40,-14)], 2508: [(40,-14)], 2516: [(40,-14)],
            2528: [(40,-14)], 2684: [(67,-14)], 2800: [(66,-14),(66,-13),(66,-12)]},
        2: {740: [(42,-16)], 796: [(40,-14)]},
        3: {1992: [(40,-14)], 2460: [(40,-14)], 2560: [(66,-15)], 4076: [(66,-15)]},
        4: {920: [(40,-14)], 4064: [(40,-14)]}}
    for assignment, frames in cases.items():
        root = run.BASE/run.root_name(assignment)
        retention_path = root/'depth_retention.json'
        retention = json.loads(retention_path.read_text()) if retention_path.exists() else None
        # No recursive discovery or privileged native-state read is needed.
        poses = {r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
        mapped = {r['frame'] for r in json.loads((root/'stage_events.json').read_text()) if r['stage']=='mapping'}
        assert {0,*frames} <= mapped
        reader = NoisyPublicReplay(root/'native'); mapper = RecordedCurrentPlaneMap()
        for frame in sorted({0,*frames}):
            policy, depth, _, _, auxiliary, now = reader.packet(frame)
            snap = mapper.update(policy, depth, poses[frame], auxiliary_depth=auxiliary, measured_ns=now)
            if frame not in frames:
                continue
            R = mapper.B@np.asarray(poses[frame]['rotation_initial_body_from_current_body'])
            p = mapper.B@np.asarray(poses[frame]['position_initial_body_m'])
            cameras = ((depth,R,p), (auxiliary,*reference_pose(R,p)))
            for cell in frames[frame]:
                views = {name:explain(packet,Q,q,mapper.floor_height,cell,mapper.last_floor_plane['available'])
                    for name,(packet,Q,q) in zip(('primary','auxiliary'),cameras,strict=True)}
                assert any(v['covered'] for v in views.values()) == (cell in snap.current_floor)
                row = dict(assignment=assignment, frame=frame, cell=cell, cameras=views,
                    paired_plane=mapper.last_floor_plane, position_map_m=p.tolist(),
                    floor_height_m=mapper.floor_height, depth_retention=retention)
                rows.append(row)
                print('FLOOR_VIEW_PROBE',assignment,frame,cell,
                    {k:(v['covered'],v.get('failure_only_outside_projected_square')) for k,v in views.items()},flush=True)
    result = dict(schema='frozen_readout_floor_view_probe.v1',rows=rows,
        delivered_noisy_depth_digests_verified=True, poses_are_recorded_estimator_outputs=True,
        native_state_or_wall_geometry_used=False, historical_accumulated_map_replayed=False,
        current_frame_classification_only=True, initial_map_basis_and_floor_height_reconstructed=True,
        controller_changed=False, alternative_navigation_outcome_proven=False,
        wall_seconds=time.monotonic()-began)
    run.previous.save(output,result)


if __name__ == '__main__':
    main()
