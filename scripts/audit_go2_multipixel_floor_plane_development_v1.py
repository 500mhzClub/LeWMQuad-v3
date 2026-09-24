"""Check recorded patch fits with covariance eigenvectors and reference queries."""
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.multipixel_floor_plane_development import fit_multipixel_plane, query_multipixel_floor
from lewm.paired_rgbd_physical_plane_development import minimum_gaps
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_multipixel_floor_plane_development_v1 import OUTPUT, INPUT, members, preflight
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def audit():
    launch = read_json(OUTPUT, 'launch.json'); exact(preflight(), launch); verify(launch)
    result = read_json(OUTPUT, 'result.json')
    if result['status'] != 'FINITE_FLOAT32_MULTIPIXEL_PLANE_DIAGNOSTIC_COMPLETE':
        raise ValueError('completed declared diagnostic required')
    bound = {str((OUTPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    bound |= {str((OUTPUT/n).relative_to(ROOT)): digest(OUTPUT/n) for n in ('launch.json', 'result.json')}
    sources = discover_sources(('scripts/audit_go2_multipixel_floor_plane_development_v1.py',), launch['source_sha256'])
    verify_bindings(bound | sources)
    first, _ = load_rgbd_observation(INPUT/'mission', 0)
    terminal, _ = load_rgbd_observation(INPUT/'mission', 218)
    mean = first['sensor_state']['sensed']['specific_force']['values'].mean(axis=0); up = mean/np.linalg.norm(mean)
    end = read_json(INPUT/'mission', 'task_decisions.json')[-1]['controller']
    R = np.asarray(end['global_orientation']['rotation_initial_body_from_current_body'])
    p = np.asarray(end['sensor_fusion']['position_initial_body_m'])
    q = terminal['sensor_state']['sensed']['joints']['values'][-1, :12]
    geometry = ArticulatedCollisionGeometry(URDF); T = np.asarray(BODY_FROM_OPTICAL)
    reports = []; queries = 0; accepted = 0
    for index, direction, rotation, translation in ((0, up, R, p), (218, R.T@up, np.eye(3), np.zeros(3))):
        saved = read_json(OUTPUT, f'frame_{index:03d}.json')
        _, packet = load_rgbd_observation(INPUT/'mission', index)
        for row, (name, depth) in zip(saved['members'], members(packet, index), strict=True):
            assert row['member'] == name
            frame = PreparedFloorFrame(depth, packet['valid'], direction); plane = fit_multipixel_plane(frame)
            assert row['status'] == plane.status and row['plane']['depth_sha256'] == frame.depth_sha256
            assert row['plane']['seed_cell_rc'] == list(plane.seed_cell_rc)
            r, c = plane.seed_cell_rc; k = plane.rules.radius_cells
            yy, xx = np.mgrid[r-k:r+k+2, c-k:c+k+2]
            z = depth[yy, xx].astype(float)
            optical = np.stack((z*(xx+.5-320)/FOCAL, z*(yy+.5-240)/FOCAL, z), axis=-1)
            points = optical@T[:3, :3].T+T[:3, 3]; centre = points.mean(axis=(0, 1))
            centred = (points-centre).reshape(-1, 3)
            _, vectors = np.linalg.eigh(centred.T@centred)
            normal = vectors[:, 0]
            if normal@direction < 0: normal = -normal
            maximum_residual = float(np.abs(centred@normal).max())
            np.testing.assert_allclose(maximum_residual, row['plane']['maximum_pixel_residual_m'], atol=1e-12, rtol=0)
            a = points[:-1, :-1]; maximum_triangle_difference = 0.
            for b, cpoint in ((points[:-1, 1:], points[1:, 1:]), (points[1:, 1:], points[1:, :-1])):
                n = np.cross(b-a, cpoint-a); n /= np.linalg.norm(n, axis=-1)[..., None]
                n[(n@direction) < 0] *= -1
                maximum_triangle_difference = max(maximum_triangle_difference, float(np.linalg.norm(n-normal, axis=-1).max()))
            report = dict(frame=index, member=name, status=plane.status,
                maximum_pixel_residual_m=maximum_residual, maximum_triangle_normal_difference=maximum_triangle_difference)
            if plane.status == 'MEASURED_PATCH_CONSISTENT':
                accepted += 1
                np.testing.assert_allclose(plane.normal, normal, atol=1e-12, rtol=0)
                np.testing.assert_allclose(plane.anchor, centre, atol=1e-12, rtol=0)
                independent_gap = minimum_gaps(geometry, q, centre, normal, rotation, translation)
                np.testing.assert_allclose(independent_gap, row['multipixel_gap_m'], atol=1e-12, rtol=0)
                query = query_multipixel_floor(frame, plane, geometry, q,
                    rotation_observation_from_body=rotation, translation_observation_from_body=translation,
                    normal_error=.002, up_error=.001, plane_offset_error=.001,
                    point_error_by_shape=dict.fromkeys(saved['shape_ids'], 0.), floor_backend='reference')
                exact(plain(query), row['physical_query']); queries += 1
            else:
                assert 'multipixel_gap_m' not in row and 'physical_query' not in row
                assert plane.status == 'FIXED_PATCH_TRIANGLE_DISAGREEMENT'
                assert maximum_triangle_difference > plane.rules.maximum_triangle_normal_error
            reports.append(report)
    verify(launch); verify_bindings(bound | sources)
    return dict(status='MULTIPIXEL_EIGEN_FIT_AND_REFERENCE_QUERY_AUDIT_COMPLETE',
        source_sha256=sources, input_sha256=bound, checked_members=len(reports),
        accepted_eigen_fits_checked=accepted, complete_reference_queries_checked=queries,
        members=reports, perturbation_generator_shared_with_runner=True,
        eigen_fit_arithmetic_independent=True, triangle_arithmetic_independent=True,
        physical_support_and_reference_query_implementation_shared=True,
        uncertainty_calibrated=False, navigation_action_permitted=False)


if __name__ == '__main__':
    target = OUTPUT/'reference_audit.json'
    if target.exists() or target.is_symlink(): raise ValueError('fresh exclusive audit only')
    result = audit(); write_json(target, result); print(result['status'], flush=True)
