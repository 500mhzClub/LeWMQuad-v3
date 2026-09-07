"""Independent extended-precision endpoint checks of every admitted mesh cell."""
from itertools import product

import numpy as np

from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.probe_go2_bounded_depth_surface_development_v1 import OUTPUT, INPUT, HYPOTHESES, preflight
from scripts.probe_go2_multipixel_floor_plane_development_v1 import members
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def check_endpoints(surface):
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        raise ValueError('independent extended-precision arithmetic unavailable')
    cells = np.argwhere(surface._mask)
    corner = cells[:, None]+np.array([[0, 0], [0, 1], [1, 1], [1, 0]])[None]
    yy, xx = corner[..., 0], corner[..., 1]
    d = surface.frame._depth[yy, xx]; z = d.astype(np.longdouble)
    lower_bin = z-np.nextafter(d, np.float32(-np.inf)).astype(np.longdouble)
    upper_bin = np.nextafter(d, np.float32(np.inf)).astype(np.longdouble)-z
    error = np.longdouble(surface.range_error_m)+np.maximum(lower_bin, upper_bin)/2+np.longdouble(1e-12)
    lo, hi = z-error, z+error
    assert surface.frame._valid[yy, xx].all() and (lo >= .2).all() and (hi <= 5.).all()
    T = np.asarray(BODY_FROM_OPTICAL, np.longdouble)
    rays = np.stack(((xx.astype(np.longdouble)+.5-320)/FOCAL,
                     (yy.astype(np.longdouble)+.5-240)/FOCAL, np.ones_like(z)), axis=-1)@T[:3, :3].T
    anchor = np.asarray(surface.anchor, np.longdouble); normal = np.asarray(surface.normal, np.longdouble)
    up = np.asarray(surface.frame._up, np.longdouble)
    largest_residual = np.longdouble(0)
    for ranges in (lo, hi):
        points = ranges[..., None]*rays+T[:3, 3]
        residual = np.abs(np.sum((points-anchor)*normal, axis=-1))
        largest_residual = max(largest_residual, residual.max())
        assert (residual <= surface.surface_tube_m).all()
    smallest_oriented_margin = np.longdouble(np.inf)
    for triangle in ((0, 1, 2), (0, 2, 3)):
        ids = list(triangle)
        nominal = z[:, ids, None]*rays[:, ids]
        base = np.cross(nominal[:, 1]-nominal[:, 0], nominal[:, 2]-nominal[:, 0])
        orientation = np.sign(np.sum(base*up, axis=1))
        assert (orientation != 0).all() and (orientation == orientation[0]).all()
        for choices in product((False, True), repeat=3):
            ranges = np.where(np.asarray(choices)[None], hi[:, ids], lo[:, ids])
            p = ranges[..., None]*rays[:, ids]
            cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
            projection = orientation*np.sum(cross*up, axis=1)
            margin = projection-surface.up_error*np.sqrt(np.sum(cross*cross, axis=1))
            assert (margin > 0).all()
            smallest_oriented_margin = min(smallest_oriented_margin, margin.min())
    return dict(admitted_cells=len(cells), endpoint_triangles_checked=16*len(cells),
        maximum_admitted_endpoint_residual_m=float(largest_residual),
        minimum_oriented_endpoint_up_margin_m2=float(smallest_oriented_margin))


def audit():
    launch = read_json(OUTPUT, 'launch.json'); exact(preflight(), launch); verify(launch)
    result = read_json(OUTPUT, 'result.json')
    if result['status'] != 'BOUNDED_DEPTH_SURFACE_DIAGNOSTIC_COMPLETE': raise ValueError('completed diagnostic required')
    bound = {str((OUTPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    bound |= {str((OUTPUT/n).relative_to(ROOT)): digest(OUTPUT/n) for n in ('launch.json', 'result.json')}
    sources = discover_sources(('scripts/audit_go2_bounded_depth_surface_development_v1.py',), launch['source_sha256'])
    verify_bindings(sources | bound)
    first, _ = load_rgbd_observation(INPUT/'mission', 0)
    mean = first['sensor_state']['sensed']['specific_force']['values'].mean(axis=0); up = mean/np.linalg.norm(mean)
    end = read_json(INPUT/'mission', 'task_decisions.json')[-1]['controller']
    R = np.asarray(end['global_orientation']['rotation_initial_body_from_current_body'])
    saved = read_json(OUTPUT, 'surface_queries.json'); reports = []
    for index, direction in ((0, up), (218, R.T@up)):
        _, depth = load_rgbd_observation(INPUT/'mission', index)
        for name, ranges in members(depth, index):
            matches = [r for r in saved if r['frame_index'] == index and r['member'] == name]
            if not matches: continue
            assert len(matches) == 1
            row = matches[0]; surface = BoundedDepthSurface(ranges, depth['valid'], direction, **HYPOTHESES)
            assert row['status'] == surface.status and row['source_depth_sha256'] == surface.frame.depth_sha256
            assert surface.status == 'BOUNDED_MEASURED_SURFACE_AVAILABLE'
            check = check_endpoints(surface)
            reports.append(dict(frame=index, member=name, **check))
            print(index, name, check, flush=True)
    assert len(reports) == len(saved) == 6
    verify(launch); verify_bindings(sources | bound)
    return dict(status='BOUNDED_SURFACE_EXTENDED_PRECISION_ENDPOINT_AUDIT_COMPLETE',
        source_sha256=sources, input_sha256=bound, members=reports,
        admitted_cells_checked=sum(r['admitted_cells'] for r in reports),
        endpoint_triangles_checked=sum(r['endpoint_triangles_checked'] for r in reports),
        extended_precision_epsilon=float(np.finfo(np.longdouble).eps),
        surface_selection_and_perturbation_generation_shared=True,
        endpoint_arithmetic_independent=True, physical_calibration_validated=False,
        real_surface_interpolation_validated=False, navigation_action_permitted=False)


if __name__ == '__main__':
    target = OUTPUT/'endpoint_audit.json'
    if target.exists() or target.is_symlink(): raise ValueError('fresh exclusive audit only')
    result = audit(); write_json(target, result); print(result['status'], flush=True)
