#!/usr/bin/env python3
"""Read-only event-interface replay on four completed development trajectories.

No physics, learned inference, counterfactual return or outcome rescoring. The
previous full audit supplies verified causal attitude/executor events. Only the
actual current packets for those events enter the new memory. Evaluation labels
are appended separately afterwards and never correct the provisional memory.
"""
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.continuation_rgb_dataset_development import load_continuation_observation
from lewm.memory.episodic_route_hypotheses_development import EpisodicRouteHypotheses
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest

STUDY = ROOT/'.generated/go2_task_acquisition_continuation_development_v1_attempt_001'
IDENTITIES = {
    'launch.json': 'ed564ec9729d1f724f871438e15a3fdfddafc654afa5092771ef34ab8319eb06',
    'result.json': '5ac8a8109ab0c4a3442955891f4a0365ff5a342175fd6bd33b0dbee808c0e135',
    'raw_artifact_audit.json': '5b0ab7f848c0432eeb762f0694c65b58ee59323f158585bc3d4199f49e9c37ca'}


def replay_events(rows, load_packet):
    """No world geometry, outcome or scene identity in this memory-facing API."""
    memory = EpisodicRouteHypotheses()
    used = []
    for index, row in enumerate(rows):
        controller = row['controller']
        child = controller.get('child')
        selected = child.get('selected_exit_candidate') if child else None
        terminal = bool(child and child['terminal'])
        acquired_view = (controller.get('selected_view_proposals') is not None
                         or controller.get('initial_proposal_rows') is not None)
        if index != 0 and selected is None and not terminal and not acquired_view:
            continue
        packet = load_packet(row['observation_index'])
        used.append(row['observation_index'])
        now = row['decision_ns']
        attitude = controller['global_orientation']
        if index == 0:
            memory.start(packet, attitude, now_ns=now)
        if acquired_view:
            memory.remember_view(packet, attitude, now_ns=now)
        if selected is not None:
            memory.begin(packet, attitude, selected, now_ns=now)
        if terminal:
            status = 'ARRIVAL_CANDIDATE' if child['status'] == 'ARRIVAL_CANDIDATE' else 'FAILED_EXECUTION'
            if memory.snapshot()['pending'] is not None:
                memory.finish(packet, attitude, now_ns=now, status=status)
            else:
                memory.abort(now_ns=now, status='FAILED_EXECUTION')
    return memory, used


def main():
    if len(sys.argv) != 1:
        raise ValueError('fixed read-only development replay takes no arguments')
    verify_bindings({str((STUDY/name).relative_to(ROOT)): sha for name, sha in IDENTITIES.items()})
    launch = json.loads((STUDY/'launch.json').read_text())
    report = json.loads((STUDY/'result.json').read_text())
    audit = json.loads((STUDY/'raw_artifact_audit.json').read_text())
    assert report['status'] == 'COMPLETE' and audit['status'] == 'PASS' and audit['audited_trials'] == 28
    verify_bindings(launch['source_sha256'] | launch['input_sha256'] | launch['gait_sha256'])
    results = []
    for case in range(4):
        scene = f'task-acquisition-development-v1-{case:02d}-fixed_forward-both'
        trial, = [row for row in report['trials'] if row['scene_id'] == scene]
        directory = STUDY/scene
        names = ['continuation_decisions.json', 'policy_observations.json', 'policy_histories.npz']
        names += [f'rgb_{i:04d}.png' for i in range(trial['rgb_packets'])]
        verify_bindings({str((directory/name).relative_to(ROOT)): trial['artifact_sha256'][name] for name in names})
        rows = json.loads((directory/'continuation_decisions.json').read_text())
        memory, used = replay_events(rows, lambda index: load_continuation_observation(directory, index))
        # The native stop signal is an executor event, not a place/geometry fix.
        if not trial['response']['checks']['no_native_stop']:
            terminal = load_continuation_observation(directory, trial['rgb_packets']-1)
            memory.abort(now_ns=terminal['sensor_state']['decision_ns'], status='PHYSICAL_STOP')
        snapshot = memory.snapshot()
        assert not snapshot['mission_complete'] and snapshot['trusted_graph_edges'] == 0
        assert all(v['place_identity'] is None for v in snapshot['visits'])
        results.append({'scene_id': scene, 'event_rgb_indices': used,
                        'memory_snapshot_sha256': hashlib.sha256(json.dumps(snapshot, sort_keys=True,
                            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                        'memory_summary': {'phase': snapshot['phase'], 'visit_events': len(snapshot['visits']),
                            'attempts': len(snapshot['attempts']), 'route_depth': snapshot['hypothesized_route_depth'],
                            'return_intent': snapshot['return_intent'], 'faults': snapshot['faults'],
                            'trusted_graph_edges': 0, 'mission_complete': False,
                            'acquired_views_per_visit': [1+len(v['context_views']) for v in snapshot['visits']],
                            'appearance_associations': [v['association'] for v in snapshot['visits']]},
                        'separate_previous_evaluation': {
                            'task_success': trial['response']['task_two_leg_integration_success'],
                            'false_arrival_candidates': sum(bool(leg['response'] and leg['response']['false_arrival_candidate'])
                                                            for leg in trial['response']['legs'])}})
    sources = ('lewm/memory/episodic_route_hypotheses_development.py',
               'lewm/tests/test_episodic_route_hypotheses_development.py',
               'scripts/check_go2_episodic_route_memory_development_v1.py')
    print(json.dumps({'status': 'PASS', 'scope': 'event-interface replay only; no return executed',
                      'source_sha256': {p: digest(ROOT/p) for p in sources}, 'prior_identities': IDENTITIES,
                      'trials': results}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
