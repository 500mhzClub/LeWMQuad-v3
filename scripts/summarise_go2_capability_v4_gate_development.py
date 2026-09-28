"""Per-episode C0 gate summary for the report: outcomes, safety, prefix erratum and veto counts."""
import json
from pathlib import Path

from scripts import run_go2_navigation_capability_completed_support_v4_development as owner


def main():
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    gate = json.loads((base/'cohorts/v4_completed_support_C0_gate/result.json').read_text())
    rows = []
    for row in gate['rows']:
        root = base/'runs'/row['assignment']
        e = json.loads((root/'episode_evaluation.json').read_text())
        p = json.loads((root/'oracle_prefix_erratum_evaluation.json').read_text())
        r = json.loads((root/'result.json').read_text())
        rows.append(dict(episode=row['episode_id'], round_trip=row['round_trip_success'], beacon=e['beacon_success'], home=e['home_success'],
            simulated_s=r['simulated_s'], wall_s=round(r['wall_s']), contacts=row['disallowed_contacts'], hard=row['hard_violations'],
            hard_unresolved=row['hard_unresolved'], operating=e['safety']['operating']['confirmed_violation_samples'],
            min_separation_m=round(e['safety']['hard']['minimum_separation_lower_m'], 4),
            fk_interval_failures=e['safety']['hard']['interval_robustness_failure_count'],
            spl=(round(e['outbound']['spl'], 3), round(e['return_leg']['spl'], 3)),
            decisions=p['decisions'], comparable_rows=p['comparable_rows'], max_err=(p['comparable_maximum_position_error_m'], p['comparable_maximum_yaw_error_deg']),
            no_match=p['no_matching_branch_decisions'], no_match_causes=p['no_matching_branch_by_cause'],
            vetoed=p['vetoed_selections'], vetoed_movement=p['vetoed_movement_selections'],
            substitutions=p['dispatch_substitution_ticks_by_cause'], frozen_checker_passed=p['frozen_checker_passed'],
            taxonomy=row['failure_and_stall_taxonomy']))
    print(json.dumps(dict(passed=gate['passed'], successes=gate['successes'], episodes=gate['episodes'], stops=gate['stops'], rows=rows), indent=1))


if __name__ == '__main__':
    main()
