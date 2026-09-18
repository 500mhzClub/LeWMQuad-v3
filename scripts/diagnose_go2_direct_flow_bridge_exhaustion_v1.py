"""Read-only diagnosis of the closed tracking collection; not its native audit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.maze_decision_stream_development import read_rows

ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_no_rgb_jepa_direct_flow_maze02_pilot_v1_attempt_001')
CASE = 'no_rgb_jepa_direct_flow_anchored_maze_02'
LAUNCH_SHA = '5b7a287f2ae7fd2fe1ca2745f5e7ccb980453b45d95894a7b4ab546f5aa55efe'
STREAM_SHA = '297c1ee2190f331f5ca80c4906d296faabff8e7460b5cb9466ad3198727a6b43'
COLLECTION_SHA = '0474b620aefeb0509014894395fc3b6a0829b4cc64705bfff971628bb2f7a545'
OUTPUT = Path('docs/go2_direct_flow_bridge_exhaustion_diagnosis_2026-09-11.json')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sources(sources):
    for name, expected in sources.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and
                not any(p == 'sealed' or p.startswith('sealed_') for p in path.parts),
                'ordinary relative source path required')
        require(digest(path) == expected, 'source changed: ' + name)


def main():
    require(not OUTPUT.exists(), 'exclusive diagnosis output required')
    bound = {ROOT/'launch.json': LAUNCH_SHA,
             ROOT/CASE/'result.json': COLLECTION_SHA,
             ROOT/CASE/'context_decisions.jsonl.gz': STREAM_SHA}
    for path, expected in bound.items():
        require(digest(path) == expected, 'fixed input changed: ' + str(path))
    launch = json.loads((ROOT/'launch.json').read_text())
    sources = launch['source_sha256']
    require(len(sources) == 2120, 'original launch source population required')
    verify_sources(sources)
    collection = json.loads((ROOT/CASE/'result.json').read_text())
    require(collection['decisions'] == 874 and collection['schedule_terminal'] == 'SENSOR_OR_MODEL_FAILURE',
            'expected completed collection required')
    require(collection['mission_receipt']['arrivals'] == [] and
            collection['mission_receipt']['verified_round_trip'] is False, 'negative mission retained')
    selected = {}
    first_terminal = None
    count = 0
    for row in read_rows(ROOT/CASE):
        count += 1
        decision = row['decision'] or {}
        if decision.get('terminal') is not None and first_terminal is None:
            first_terminal = row['tick']
        if 859 <= row['tick'] <= 863:
            selected[row['tick']] = decision
    require(count == 874 and first_terminal == 863, 'complete stream and first terminal required')
    bridge_history = []
    for tick in range(859, 863):
        decision = selected[tick]
        continuity = decision['original_visual_evidence']['continuity_evidence']
        require(decision['terminal'] is None and decision['requested_command'] == [0., 0., -.45]
                and continuity['status'] == 'MEASURED_INCREMENT_BRIDGE'
                and continuity['bridge_frames'] == tick-852, 'exact accepted bridge history required')
        bridge_history.append(dict(frame=tick, requested_command=decision['requested_command'],
            bridge_frames=continuity['bridge_frames'], bridge_path_m=continuity['bridge_path_m']))
    decision = selected[863]
    visual = decision['original_visual_evidence']
    fallback = visual['direct_corner_flow_fallback']
    require(decision['requested_command'] == [0., 0., 0.] and visual['current_pose'] is None,
            'terminal zero command and unavailable pose retained')
    require(fallback['accepted'] is False and fallback['failure'] ==
            'bounded measured bridge exhausted without anchor observation', 'bridge failure required')
    cameras = {
        'primary': (visual['camera_selection']['primary_continuity'],
                    visual['camera_selection']['primary_reference_selection']),
        'auxiliary': (visual['continuity_evidence'], visual['reference_selection']),
    }
    measurements = {}
    for camera, (continuity, selection) in cameras.items():
        require(continuity['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED' and
                continuity['incremental_available'] is True and continuity['anchor_available'] is False,
                'both cameras have qualified increment but no anchor')
        attempts = selection['attempts']
        require(len(attempts) == 8 and all(a['status'] == 'REJECTED' for a in attempts),
                'all eight retained references rejected')
        witness = continuity['incremental_rotation_witness']
        require(witness['reference_frame'] == 862 and witness['current_frame'] == 863 and
                witness['camera'] == camera and witness['candidate_envelope_passed'] is True,
                'current qualified incremental witness required')
        measurements[camera] = dict(anchor_attempts=attempts,
            incremental_witness={k: witness[k] for k in ('reference_frame', 'current_frame', 'inliers',
                'inlier_fraction', 'reference_grid_cells', 'current_grid_cells', 'residual_rms_m',
                'gyro_disagreement_rad', 'candidate_envelope_passed', 'witness_alone_grants_pose')})
    local_sources = {str(Path(__file__).relative_to(Path.cwd())): digest(Path(__file__)),
                     'scripts/maze_decision_stream_development.py': digest(Path('scripts/maze_decision_stream_development.py'))}
    verify_sources(sources)
    for path, expected in bound.items():
        require(digest(path) == expected, 'input changed during diagnosis')
    report = dict(status='DIRECT_FLOW_CLOSED_COLLECTION_BRIDGE_EXHAUSTION_DIAGNOSED',
        utc=datetime.now(timezone.utc).isoformat(), input_sha256={str(p): h for p, h in bound.items()},
        source_sha256=sources | local_sources, original_launch_source_count=len(sources),
        decision_rows=count, first_terminal_frame=first_terminal, accepted_bridge_history=bridge_history,
        terminal_failure=visual['terminal_failure'], camera_measurements=measurements,
        terminal_fallback=fallback, terminal_requested_command=decision['requested_command'],
        diagnosis='Qualified current increments remain; ten-frame bridge budget is exhausted because no retained anchor qualifies.',
        next_investigation='Inspect correspondence and overlap between retained anchors and current public RGB-D; test anchor reacquisition under unchanged rigid and continuity gates.',
        scope=dict(closed_collection_only=True, native_audit_verified=False, physical_prefix_verified=False,
            raw_sensor_registration_reexecuted=False, controller_replayed=False, native_execution_launched=False,
            bridge_budget_changed=False, thresholds_changed=False, navigation_success=False,
            independent_layout_evidence=False, real_time_evidence=False, hardware_evidence=False))
    with OUTPUT.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), sha256=digest(OUTPUT), status=report['status'],
                         first_terminal_frame=first_terminal, rows=count)))


if __name__ == '__main__':
    main()
