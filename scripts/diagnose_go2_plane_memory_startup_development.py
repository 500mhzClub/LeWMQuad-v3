"""Separate local measured geometry from a failed historical-fusion startup.

Read-only diagnostics of a completed short command tape, not a memory restart,
post-stop control run, contact inference or permission to move.
"""
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.causal_sensor_state import SensorContractError
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneHypothesis, MeasuredPlaneObstacleMemory
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.probe_go2_measured_plane_obstacle_memory_development import DIRECTORY, IDENTITIES, SOURCES
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest


def cause_chain(error):
    messages = []
    while error is not None:
        messages.append(str(error)); error = error.__cause__
    return messages


def local_geometry(geometry, policy, depth, up):
    """Same-frame q/depth geometry, with no historical translation input."""
    now = depth['measured_ns']; joints = policy['sensor_state']['sensed']['joints']
    if joints['measured_ns'][-1] != now or not joints['valid'][-1, :12].all():
        raise SensorContractError('current measured posture required')
    q = joints['values'][-1, :12]
    frame = PreparedFloorFrame(depth['depth_m'], depth['valid'], up)
    h = MeasuredPlaneHypothesis.from_frame(frame)
    shapes = geometry.supports(q, np.eye(3))['shapes']
    feet = np.array([s['center_body_m'] for s in shapes if s['shape_id'] in FOOT_SHAPES])
    optical = (feet - np.asarray(BODY_FROM_OPTICAL)[:3, 3]) @ np.asarray(BODY_FROM_OPTICAL)[:3, :3]
    covered = []; gap_candidates = []; separated = []; low_gaps = []
    if h.cell_rc is not None:
        floor = frame.query(geometry, q, h.cell_rc, rotation_observation_from_body=np.eye(3),
            translation_observation_from_body=np.zeros(3), point_error_by_shape={s['shape_id']: 0. for s in shapes},
            normal_error=.002, up_error=.001, plane_offset_error=.001, floor_backend='cached')
        covered = [sid for sid, value in floor['floor_coverage'].items() if value]
        for gap in floor['gap_bounds']['primitives']:
            sid = gap['shape_id']; lo = gap['minimum_gap_lower_m']; hi = gap['minimum_gap_upper_m']
            low_gaps.append(lo)
            if lo > 0: separated.append(sid)
            if sid in FOOT_SHAPES and lo <= 0 <= hi: gap_candidates.append(sid)
    return dict(hypothesis_cell_rc=h.cell_rc, depth_sha256=h.depth_sha256,
        floor_covered_primitives=covered, plane_separated_primitives_without_coverage=separated,
        foot_gap_straddles_without_coverage=gap_candidates,
        foot_centre_optical_depths_m=optical[:, 2].tolist(),
        minimum_conditional_plane_gap_lower_m=min(low_gaps) if low_gaps else None,
        historical_translation_used=False, contact_permitted=False, navigation_qualified=False)


def main():
    bindings = {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = json.loads((DIRECTORY / 'launch.json').read_text())
    result = json.loads((DIRECTORY / 'result.json').read_text())
    reader = json.loads((DIRECTORY / 'native_box_reader_correction.json').read_text())
    assert reader['result']['interface_check_pass'] and result['rgbd_frames'] == 26
    bindings |= launch['source_sha256'] | launch['input_sha256'] | reader['reader_source_sha256']
    bindings |= {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    development = {p: digest(ROOT / p) for p in (*SOURCES,
        'scripts/diagnose_go2_plane_memory_startup_development.py',
        'lewm/tests/test_plane_memory_startup_development.py')}
    verify_bindings(bindings); verify_bindings(development)
    print(json.dumps({'development_source_sha256': development}), flush=True)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    assert len(records) == 26
    geometry = ArticulatedCollisionGeometry(URDF)
    model = MeasuredPlaneObstacleMemory(geometry, normal_error=.002, up_error=.001,
        plane_offset_error=.001, range_error_m=.001, beam_backend='compiled')
    first, _ = load_rgbd_observation(DIRECTORY, 0)
    force = first['sensor_state']['sensed']['specific_force']
    assert force['valid'].all()
    up = force['values'].mean(axis=0); norm = np.linalg.norm(up)
    assert abs(norm - 9.81) <= .75
    up = up / norm  # Same explicit quiet-initial-gravity hypothesis as fusion.
    stop = None; rows = []
    try:
        for tick, record in enumerate(records):
            p, d = load_rgbd_observation(DIRECTORY, tick); now = p['sensor_state']['decision_ns']
            relative = record['observer']
            if stop is None:
                try: model.observe(p, d, relative, now_ns=now)
                except SensorContractError as error:
                    stop = dict(tick=tick, measured_ns=now, causes=cause_chain(error), fault_latched=model.failed)
                    print(json.dumps({'historical_memory_stop': stop}), flush=True)
            R = np.asarray(relative['relative_orientation']['rotation_initial_body_from_current_body'])
            np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-7)
            row = local_geometry(geometry, p, d, R.T @ up)
            assert row['depth_sha256'] == relative['local_surfaces']['depth_sha256']
            motion = relative['motion']
            row.update(tick=tick, measured_ns=now, depth_rank=motion['rank'] if motion else None,
                weak_directions_previous_body=motion['weak_directions_previous_body'] if motion else [],
                post_memory_stop_local_diagnostic_only=stop is not None)
            rows.append(row); print(json.dumps(row), flush=True)
    finally:
        verify_bindings(bindings); verify_bindings(development)
    print(json.dumps({'status': 'LOCAL_GEOMETRY_DIAGNOSTIC_COMPLETE_HISTORICAL_MEMORY_STOP_PRESERVED',
        'frames': len(rows), 'historical_memory_stop': stop,
        'frames_with_plane_hypothesis': sum(r['hypothesis_cell_rc'] is not None for r in rows),
        'frames_with_any_own_body_floor_coverage': sum(bool(r['floor_covered_primitives']) for r in rows),
        'maximum_current_foot_centre_optical_depth_m': max(max(r['foot_centre_optical_depths_m']) for r in rows),
        'rank_two_intervals': sum(r['depth_rank'] == 2 for r in rows),
        'all_bindings_verified_before_and_after': True,
        'quiet_initial_gravity_assumed': True, 'sensor_errors_calibrated': False,
        'historical_memory_qualified': False, 'contact_permitted': False, 'navigation_qualified': False}), flush=True)


if __name__ == '__main__': main()
