"""Evaluate physical navigation and actual auxiliary-only turn requests."""
import argparse
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_multiseed_navigation_development as previous
from scripts import run_go2_auxiliary_turn_recovery_development as recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=recovery.ARMS, required=True)
    args = parser.parse_args()
    study = SimpleNamespace(**(vars(previous.study) | dict(ROOT=recovery.ROOT)))
    result = bind(previous.evaluate, study=study)(1, args.arm)
    root = previous.previous.path(result['root_name'])
    read = previous.previous.read
    rows = read(root, 'requests.json')
    degraded = [r for r in rows if r.get('auxiliary_only_turn_recovery')]
    moving = [r for r in degraded if any(r['requested_command'])]
    for row in moving:
        if (any(row['requested_command'][:2])
                or row['reason'] != 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'
                or row['current_valid_return_counts'][0] != 0
                or row['current_valid_return_counts'][1] <= 0
                or not 0 <= row['observation_age_ns'] <= 200_000_000
                or not row['nominal_connector']['nominal_disk_connector_clear']):
            raise ValueError('actual degraded request violates turn-only guard scope')
    first = None if not moving else moving[0]['now_ns']
    after = [] if first is None else [r for r in rows if r['now_ns'] > first]
    restored = [] if first is None else [r for r in read(root, 'independent_depth_receipts.json')
        if r['measured_ns'] > first and 'retained_obstacle_cells' in r
        and not r.get('auxiliary_only_current_obstacles')]
    receipt = dict(auxiliary_only_guard_requests=len(degraded),
        auxiliary_only_nonzero_turn_requests=len(moving), first_auxiliary_only_turn_ns=first,
        primary_blind_translation_vetoes=sum(r['reason']=='PRIMARY_DEPTH_UNAVAILABLE_TRANSLATION_VETO' for r in rows),
        later_nonzero_translation_requests=sum(any(r['requested_command'][:2]) for r in after),
        first_later_paired_obstacle_frame=None if not restored else restored[0]['frame'],
        first_later_paired_obstacle_measured_ns=None if not restored else restored[0]['measured_ns'],
        actual_degraded_turn_scope_verified=bool(moving), round_trip=result['round_trip'],
        contacts=result['contacts'], physical_recovery_requires_turn_exposure_and_resumed_navigation=True,
        hardware_validated=False)
    previous.previous.save_or_read(root, 'auxiliary_turn_recovery_evaluation_v1.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__': main()
