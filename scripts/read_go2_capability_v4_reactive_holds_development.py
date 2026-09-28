"""Frozen V4 episode reader with an evaluator-only erratum for reactive-selector holds.

C2's reactive selector logs candidates as {action, eligible, normalized_command_distance_squared}
without utility scores or forecast-clearance records, so the frozen hold classifier
raises KeyError('utility_m') on any C2 hold. This wrapper routes exactly those rows
(no 'scan_utilities' and a candidate without 'utility_m') to a reactive classification;
every other row goes to the unchanged classifier, so previously readable episodes are
unaffected by construction. Arrival, safety, SPL and stall computations are unchanged.
"""
import argparse
import math
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.navigation_capability_hold_taxonomy_development import classify as frozen_classify
from lewm.navigation_capability_hold_taxonomy_development import override_label as frozen_override_label
from scripts import read_go2_navigation_capability_episode_development as reader
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
ERRATUM = 'docs/go2_navigation_capability_reactive_hold_reader_erratum_2026-09-28.md'


def reactive_row(row):
    selection = row['selection']
    candidates = selection.get('candidates', [])
    return 'scan_utilities' not in selection and bool(candidates) and not all('utility_m' in c for c in candidates)


def classify(row):
    if not reactive_row(row):
        return frozen_classify(row)
    s = row['selection']
    candidates = {c['action']: c for c in s['candidates']}
    eligible = [a for a in ACTIONS if a in candidates and candidates[a].get('eligible')]
    moving = [a for a in eligible if a != 'hold']
    result = dict(frame=row['frame'], route_status=row.get('route_status'), category='insufficient_evidence',
        observation_action_space_exclusions=[], motion_clearance_exclusions=[], stopping_projection_exclusions=[],
        clearance_modes={}, override_reason=None, eligible_movement=moving,
        reactive_ineligible_movement=[a for a in ACTIONS[1:] if a in candidates and not candidates[a].get('eligible')],
        reactive_selector=True, no_eligible_movement_even_if_override=not moving,
        exclusion_rule_logged=False)
    distances = {a: candidates[a].get('normalized_command_distance_squared') for a in eligible}
    terminal = s.get('heading_first_terminal', {})
    if s['action'] == 'hold' and terminal.get('changed') and terminal.get('selected_action') == 'hold':
        result.update(category='explicit_override', override_reason='HEADING_FIRST_TERMINAL_HOLD',
            binding_rule='reactive terminal rule: at the goal, hold before the measured-heading pulse',
            intended_arrival_settling=True, previous_action=terminal.get('previous_action'),
            reason='Retained reactive terminal rule explicitly selected hold at the goal; not itself a failure')
    elif set(candidates) != set(ACTIONS) or any(d is None or not math.isfinite(d) for d in distances.values()):
        result['reason'] = 'reactive candidates incomplete or non-finite command distances'
    elif not moving:
        result.update(category='no_eligible_movement', reason='reactive selector logs eligibility, not the excluding rule')
    elif 'hold' in eligible and all(distances['hold'] <= distances[a] for a in moving):
        result.update(category='movement_lost_recorded_score_or_tie',
            exact_tie_with_movement=any(distances['hold'] == distances[a] for a in moving),
            tie_rule='reactive: eligible candidate nearest the desired instantaneous command; hold first in canonical order')
    else:
        result['reason'] = 'eligible movement nearer the desired command than hold; no retained override explains the hold'
    result['inherited_stage_a_category'] = result['category']
    return result


def override_label(row):
    if row.get('override_reason') == 'HEADING_FIRST_TERMINAL_HOLD':
        return 'planned arrival settling'
    return frozen_override_label(row)


def report(root):
    def save(path, value):
        if path.name == 'episode_evaluation.json':
            value = dict(value, label=('Capability qualification' if value.get('role') == 'validation' else 'Paired-floor development harness evidence; not validation'),
                harness_version='v4_completed_support', reader_owner_sha256=owner.sha(__file__),
                reactive_hold_reader_erratum=dict(path=ERRATUM, sha256=owner.sha(owner.REPO/ERRATUM)),
                frozen_v4_reader_sha256=owner.sha(owner.REPO/'scripts/read_go2_navigation_capability_completed_support_v4_development.py'))
        owner.save(path, value)
    return bind(reader.report, Budget=owner.Budget, PROTOCOL=owner.PROTOCOL, save=save, classify=classify, override_label=override_label)(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    r = report(p.parse_args().root)
    print({k: r.get(k) for k in ('controller', 'episode_id', 'round_trip_success', 'disallowed_contact_samples', 'failure_and_stall_taxonomy')})
