"""Use the hash-bound launch source table omitted from the execution receipt."""
import argparse
import json

from scripts import verify_go2_sustained_hold_reorientation_raw_completion_v1 as original
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify, write_json, digest
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/verify_go2_sustained_hold_reorientation_raw_completion_v2.py'
TEST = 'lewm/tests/test_sustained_raw_completion_execution_sources_development.py'
FAILURE = 'docs/go2_sustained_hold_reorientation_raw_completion_checker_v1_failure_2026-09-11.json'
FAILURE_SHA = '4d2529bee2171a1e655cee747a313514e3a257029797f556e86857e4809571fb'
OUTPUT = ROOT/'docs/go2_sustained_hold_reorientation_raw_completion_verification_v2_2026-09-11.json'


def execution_with_sources(execution, sources):
    original.require_ended()
    root = original.prefix.replay.OUTPUT
    if execution['launch_sha256'] != original.prefix.LAUNCH_SHA:
        raise ValueError('exact recorded original launch required')
    verify_artifacts(root, {'launch.json':execution['launch_sha256']})
    launch = read_json(root, 'launch.json'); table = launch['source_sha256']
    if (len(table) != execution['source_count']
            or any(sources.get(n) != h for n,h in table.items())
            or launch['owner_pid'] != execution['owner']['pid']
            or launch['boot_id'] != execution['boot_id']):
        raise ValueError('recorded source count and owner must match exact frozen launch')
    return execution | dict(source_sha256=dict(table))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--result-sha256', required=True); args = parser.parse_args()
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive V2 verification output required')
    verify({FAILURE:FAILURE_SHA})
    sources, execution = original.prepared_sources()
    sources = discover_sources((SOURCE, TEST, FAILURE), sources); verify(sources)
    augmented = execution_with_sources(execution, sources)
    result = original.verify_completed(args.result_sha256, sources, augmented)
    result.update(status='SUSTAINED_HOLD_REORIENTATION_RAW_COMPLETION_V2_VERIFIED',
        initial_checker_failure_sha256=FAILURE_SHA, execution_source_table_from_exact_bound_launch=True,
        original_execution_receipt_unchanged=True, original_raw_experiment_restarted=False)
    verify(sources); write_json(OUTPUT, result)
    print('SUSTAINED_RAW_COMPLETION_V2_VERIFIED', digest(OUTPUT), len(sources), result['report']['frames'], flush=True)


if __name__ == '__main__': main()
