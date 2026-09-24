"""Check the endpoint-fit candidate against original recorded public pose evidence."""
import json
from pathlib import Path

import numpy as np

from scripts import probe_go2_chained_first_bridge_anchor_pairs_v1 as probe
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows
from lewm.joint_rgbd_rigid_pose_development import angle, proper, RIGID_RULES
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES

SOURCE = 'scripts/verify_go2_chained_first_bridge_pose_candidate_v1.py'
OUTPUT = Path('docs/go2_chained_first_bridge_pose_candidate_verification_2026-09-11.json')
PROBE_SHA = '7e2dd4bf40d066d558f61ec1b9ecf901230a82aa62886901cf2537ee4ef90660'


def main():
    diagnosis = probe.prior_probe.diagnosis
    require, digest = diagnosis.require, diagnosis.digest
    require(not OUTPUT.exists() and digest(probe.OUTPUT) == PROBE_SHA, 'exclusive output and fixed endpoint probe required')
    prior = json.loads(probe.OUTPUT.read_text())
    sources = discover_sources((SOURCE,), prior['source_sha256'])
    diagnosis.verify_sources(sources)
    candidates = [p for p in prior['pairs'] if p['registration']['qualified']]
    require(len(candidates) == 1 and candidates[0]['camera'] == 'auxiliary' and
            candidates[0]['reference_frame'] == 850 and candidates[0]['current_frame'] == 853,
            'unique original all-pair probe candidate required')
    directory = diagnosis.ROOT/diagnosis.CASE
    stream = directory/'context_decisions.jsonl.gz'
    require(digest(stream) == diagnosis.STREAM_SHA, 'bound closed decisions required')
    rows, count = {}, 0
    for row in read_rows(directory):
        count += 1
        if row['tick'] in (850, 852, 853): rows[row['tick']] = row['decision']['original_visual_evidence']
    require(count == 874, 'complete closed decision population required')
    reference = rows[850]['current_pose']
    previous = rows[852]['current_pose']
    current = rows[853]['current_pose']
    require(reference['frame'] == 850 and reference['promoted_keyframe'] is True and
            previous['frame'] == 852 and current['frame'] == 853, 'actual retained and current measured pose identities required')
    selection = rows[853]['reference_selection']
    require(selection['attempts'][0]['reference_frame'] == 850 and
            rows[853]['continuity_evidence']['bridge_frames'] == 1, 'latest retained anchor at actual first bridge required')
    registration = candidates[0]['registration']
    local_R = proper(registration['reference_body_from_current_body'])
    local_t = np.asarray(registration['translation_reference_body_m'])
    R = proper(np.asarray(reference['rotation_initial_body_from_current_body'])@local_R)
    p = np.asarray(reference['position_initial_body_m']) + np.asarray(reference['rotation_initial_body_from_current_body'])@local_t
    previous_distance = float(np.linalg.norm(p-np.asarray(previous['position_initial_body_m'])))
    previous_rotation = angle(np.asarray(previous['rotation_initial_body_from_current_body']).T@R)
    envelope = dict(distance_m=previous_distance, rotation_rad=previous_rotation,
        passes=bool(previous_distance <= RIGID_RULES['maximum_increment_translation_m'] and
                    previous_rotation <= RIGID_RULES['maximum_increment_rotation_rad']))
    witnesses = list(rows[853]['continuity_evidence']['rotation_measurement_witnesses'])
    primary = rows[853]['camera_selection'].get('primary_continuity') or {}
    witnesses += primary.get('rotation_measurement_witnesses', [])
    require(len(witnesses) > 0, 'original qualified measurements required')
    comparisons = []
    for witness in witnesses:
        distance = float(np.linalg.norm(p-np.asarray(witness['position_initial_body_m'])))
        rotation = angle(np.asarray(witness['composed_rotation_initial_body_from_current_body']).T@R)
        comparisons.append(dict(camera=witness['camera'], reference_frame=witness['reference_frame'],
            distance_m=distance, rotation_rad=rotation,
            agrees=bool(distance <= CONTINUITY_RULES['maximum_measured_disagreement_m'] and
                        rotation <= CONTINUITY_RULES['maximum_measured_disagreement_rad'])))
    diagnosis.verify_sources(sources)
    require(digest(stream) == diagnosis.STREAM_SHA and digest(probe.OUTPUT) == PROBE_SHA, 'inputs unchanged after verification')
    report = dict(status='CHAINED_FIRST_BRIDGE_RECORDED_POSE_CANDIDATE_CHECKED',
        source_sha256=sources, probe_sha256=PROBE_SHA, decision_stream_sha256=diagnosis.STREAM_SHA,
        decision_rows=count, candidate=candidates[0], previous_pose_envelope=envelope,
        original_measurement_comparisons=comparisons,
        original_measurement_checks_pass=bool(envelope['passes'] and all(x['agrees'] for x in comparisons)),
        scope=dict(recorded_public_visual_pose_only=True, native_state_used=False,
            full_observer_history_replayed=False, reference_promotion_executed=False,
            controller_replayed=False, command_selected=False, physics_executed=False,
            navigation_recovered=False, goal_achieved=False))
    with OUTPUT.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), sha256=digest(OUTPUT), envelope=envelope,
        comparisons=comparisons, passes=report['original_measurement_checks_pass'])), flush=True)


if __name__ == '__main__':
    main()
