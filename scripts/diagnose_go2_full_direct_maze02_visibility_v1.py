"""Localize a completed direct-model visibility failure without changing its score."""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_first_surface_depth_development import expected_optical_depth
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.diagnose_go2_recorded_raster_failures_v1 import projected_edges, nearest_edge
from scripts.navigation_artifact_root_development import BASE, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json, read_npz
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/diagnose_go2_full_direct_maze02_visibility_v1.py'
PROOF = Path('docs/go2_all_phase_adapter_full_direct_maze02_verification_2026-09-10.json')
PROOF_SHA = '35c30a45137912829e114306bb8ff63e471d6d48d8b2e4c7e2879cd717219a30'
OUTPUT = Path('docs/go2_full_direct_maze02_visibility_diagnosis_2026-09-10.json')
ROOT = BASE / 'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
CASE = 'all_phase_full_direct_residual_maze_02'
FRAMES = (1319, 1320, 1321)


def main():
    if not __debug__:
        raise ValueError('assertions required')
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive diagnostic output required')
    assert digest(PROOF) == PROOF_SHA
    proof = json.loads(PROOF.read_text())
    assert proof['status'] == 'COMPLETED_FULL_DIRECT_ADAPTER_NATIVE_EVIDENCE_AUTHENTICATED'
    assert proof['case'] == CASE and proof['primary_strict_failed_frames'] == [1320]
    assert proof['auxiliary_failed_frames'] == [] and not proof['verified_round_trip']
    sources = discover_sources([SOURCE, str(PROOF)], proof['source_sha256'])
    verify(sources)
    names = ['launch.json', CASE + '_worker_terminal.json', CASE + '_audit.json']
    names += [CASE + '/' + n for n in ('specification.json', 'camera_audit.json', 'depth_camera_audit.json')]
    names += [f'{CASE}/native_depth_{i:04d}.npz' for i in FRAMES]
    bindings = {n: proof['artifact_sha256'][n] for n in names}
    verify_artifacts(ROOT, bindings)
    worker = read_json(ROOT, CASE + '_worker_terminal.json')
    audit = read_json(ROOT, CASE + '_audit.json')
    assert worker['strict_physical_visibility_pass'] is False
    assert worker['hard_measurement_failed_frames'] == []
    assert worker['verified_round_trip'] is False
    directory = ROOT / CASE
    spec = read_json(directory, 'specification.json')
    cameras = read_json(directory, 'camera_audit.json')
    depths = read_json(directory, 'depth_camera_audit.json')
    boxes = spec['geometry']['wall_boxes']
    rows = []
    for frame in FRAMES:
        camera, depth = cameras[frame], depths[frame]
        T = np.asarray(camera['world_from_optical'], dtype=float)
        native = read_npz(directory, f'native_depth_{frame:04d}.npz')['optical_depth_m']
        raw_sha = hashlib.sha256(native.tobytes()).hexdigest()
        assert raw_sha == depth['native_depth_sha256']
        assert depth['physical_sample_index'] == camera['physical_sample_index'] == 749 + 50 * frame
        assert depth['native_near_m'] == spec['render_near_m'] == .005
        report = evaluate_footprint(native, boxes, T, render_near_m=.005)
        assert report == audit['footprint_checks'][frame]
        assert report['original_strict_score'] == audit['depth_checks'][frame]['physical_visibility']
        ref = expected_optical_depth(boxes, T)
        expected = ref['expected_depth_m']
        measured = native[np.ix_(ref['rows'], ref['columns'])]
        use = ref['surface_interior'] & np.isfinite(expected) & (expected < 4.98)
        bad = use & (~np.isfinite(measured) | (np.abs(measured - expected) > .001))
        edges = projected_edges(boxes, T)
        pixels = []
        for iy, ix in np.argwhere(bad):
            v, u = int(ref['rows'][iy]), int(ref['columns'][ix])
            direction = T[:3, :3] @ np.array([(u + .5 - 320) / FOCAL, (v + .5 - 240) / FOCAL, 1.])
            pixels.append(dict(
                row=v, column=u, expected_m=float(expected[iy, ix]), native_m=float(measured[iy, ix]),
                expected_object=ref['object_names'][ref['object_index'][iy, ix]],
                expected_hit=(T[:3, 3] + expected[iy, ix] * direction).tolist(),
                native_ray_endpoint=(T[:3, 3] + measured[iy, ix] * direction).tolist(),
                native_3x3=native[v-1:v+2, u-1:u+2].tolist(),
                nearest_physical_edge=nearest_edge(edges, v, u)))
        assert len(pixels) == report['stable_interior_bad_rays'] + report['boundary_bad_rays']
        assert hashlib.sha256(native.tobytes()).hexdigest() == raw_sha
        rows.append(dict(frame=frame, original_report_reproduced=True,
                         footprint_report=report, failed_sampled_rays=pixels))
    verify(sources)
    verify_artifacts(ROOT, bindings)
    assert digest(PROOF) == PROOF_SHA
    write_json(OUTPUT, dict(
        status='FULL_DIRECT_MAZE02_RECORDED_VISIBILITY_LOCALIZED_NOT_RESCORED',
        source_sha256=sources, input_sha256=bindings, verification_sha256=PROOF_SHA,
        case=CASE, frames=rows, original_strict_physical_visibility_pass=False,
        original_hard_measurement_failed_frames=[], original_verified_round_trip=False,
        original_failure_preserved=True, native_execution=False, model_inference=False,
        policy_pixels_changed=False, qualification_granted=False, goal_achieved=False,
        limitations=[
            'Only three original recorded frames recomputed; complete original raw audit not rerun.',
            'Projected box edges include hidden/internal edges; proximity is not causal proof.',
            'Boundary accounting does not certify those pixels or override the original strict failure.',
            'No rerender, threshold change, sensor correction, model selection or hardware qualification.']))
    print(json.dumps(dict(output=str(OUTPUT), sha256=digest(OUTPUT), source_count=len(sources),
                          failed_sampled_rays={r['frame']: r['failed_sampled_rays'] for r in rows})), flush=True)


if __name__ == '__main__':
    main()
