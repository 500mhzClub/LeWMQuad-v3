"""Post-hoc release/re-latch events in three retained development recordings."""
import json
import math
from pathlib import Path
import statistics

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
ROOTS = (
    'go2_nogil_replication_jepa_noise_2mm_native_layout01_4800_v1_attempt_001',
    'go2_recovery_survey_transfer_limited_survey_jepa_noise_2mm_native_layout01_4800_v1_attempt_001',
    'go2_recovery_survey_transfer_original_jepa_noise_2mm_native_layout01_4800_v1_attempt_001',
)
RELEASE = 'FULL_RESERVE_PREFERRED_HEADING_REJOINS_ROUTE_OR_VIEW'
LATCH = 'CLEAR_ALTERNATIVE_TURN_LATCHED'


def distribution(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values)) if values else dict(count=0)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def read_case(name):
    root = BASE / name
    rows = [r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r]
    poses = {r['frame']: r['registered_pose'] for r in json.loads((root/'poses.json').read_text())
             if r.get('registered_pose') is not None}
    pairs = []
    release_count = 0
    for i, row in enumerate(rows):
        event = row['selection'].get('clearance_turn', {})
        if event.get('event') != RELEASE:
            continue
        release_count += 1
        following = next((r for r in rows[i+1:]
                          if r['selection'].get('clearance_turn', {}).get('event') == LATCH), None)
        if following is None:
            continue
        latch = following['selection']['clearance_turn']
        action = row['selection']['full_reserve_heading_release']['selected_action']
        blocked = latch['blocked_preferred_action']
        before = next(r for r in row['selection']['memory_forecast_candidates'] if r['action'] == action)
        after = next(r for r in following['selection']['memory_forecast_candidates'] if r['action'] == blocked)
        a, b = poses[row['frame']], poses[following['frame']]
        pa, pb = a['position_initial_body_m'], b['position_initial_body_m']
        ra, rb = a['rotation_initial_body_from_current_body'], b['rotation_initial_body_from_current_body']
        pairs.append(dict(release_frame=row['frame'], relatch_frame=following['frame'],
            gap_s=(following['measured_ns']-row['measured_ns'])/1e9,
            release_action=action, next_blocked_preferred_action=blocked,
            same_preferred_turn=action == blocked,
            same_mission_generation=(row['mission_generation'] == following['mission_generation']
                if 'mission_generation' in row and 'mission_generation' in following else None),
            release_full_reserve_margin_mm=1000*(before['minimum_predicted_path_clearance_m']-before['required_path_clearance_m']),
            relatch_reserve_shortfall_mm=1000*(after['required_path_clearance_m']-after['minimum_predicted_path_clearance_m']),
            registered_xy_displacement_m=math.hypot(pb[0]-pa[0], pb[1]-pa[1]),
            registered_heading_change_rad=wrap(math.atan2(rb[1][0],rb[0][0])-math.atan2(ra[1][0],ra[0][0])),
            target_heading_change_rad=wrap(latch['target_heading_rad']-event['previous_target_heading_rad']),
            release_on_time=row['on_time'], relatch_on_time=following['on_time']))
    return dict(root=name, plans=len(rows), release_events=release_count,
        subsequent_relatch_pairs=len(pairs), pairs=pairs,
        gap_s=distribution([r['gap_s'] for r in pairs]),
        registered_xy_displacement_m=distribution([r['registered_xy_displacement_m'] for r in pairs]),
        absolute_target_heading_change_rad=distribution([abs(r['target_heading_change_rad']) for r in pairs]),
        release_full_reserve_margin_mm=distribution([r['release_full_reserve_margin_mm'] for r in pairs]),
        relatch_reserve_shortfall_mm=distribution([r['relatch_reserve_shortfall_mm'] for r in pairs]),
        same_preferred_turn=sum(r['same_preferred_turn'] for r in pairs),
        both_plans_on_time=sum(r['release_on_time'] and r['relatch_on_time'] for r in pairs))


if __name__ == '__main__':
    result = dict(schema='turn_release_cycle_diagnosis.v1', exploratory_posthoc=True,
        cases=[read_case(name) for name in ROOTS],
        event_pairing='each release and its next later alternative-turn latch',
        overlapping_event_pairs_possible=True, unexecuted_action_safety_inferred=False,
        counterfactual_navigation_success_inferred=False, causal_defect_isolated=False)
    output = BASE/'go2_turn_release_cycle_diagnosis_v1_attempt_001'
    output.mkdir()
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k != 'pairs'} for r in result['cases']], indent=2))
