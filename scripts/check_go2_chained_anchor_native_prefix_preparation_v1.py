"""Check source preparation against the actual completed controller prefix."""
from contextlib import closing
from itertools import islice
import json
from pathlib import Path

from scripts import chained_anchor_native_prefix_development as prefix
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE = 'scripts/check_go2_chained_anchor_native_prefix_preparation_v1.py'
TEST = 'lewm/tests/test_chained_anchor_native_prefix_development.py'
PROTOCOL = 'docs/go2_chained_anchor_native_prefix_preparation_v1_2026-09-11.md'
OUTPUT = Path('docs/go2_chained_anchor_native_prefix_preparation_2026-09-11.json')
COMPLETION_SHA = '6b0c3298bb4df47aa59affc80d5dcb9236c1e4d9ae5cabd07550800c94d26425'
PREPARATION = Path('docs/go2_chained_anchor_native_source_preparation_2026-09-11.json')
PREPARATION_SHA = '808456de9a1cdbfbc236dd0a971d9b4495118d8cd246af6c0de731f4db196de6'


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive preparation check required')
    proof_path = prefix.completed.OUTPUT
    verify({str(proof_path): COMPLETION_SHA, str(PREPARATION): PREPARATION_SHA})
    proof = json.loads(proof_path.read_text())
    preparation = json.loads(PREPARATION.read_text())
    if (proof['status'] != 'CHAINED_ANCHOR_CONTROLLER_COMPLETION_VERIFIED'
            or proof['result_sha256'] != prefix.completed.RESULT_SHA
            or proof['owner'] != prefix.completed.OWNER or proof['owner_ended'] is not True
            or prefix.replay.observer.owner_live(prefix.completed.OWNER)):
        raise ValueError('exact completed controller verification and ended owner required')
    inherited = merge_sources(proof['source_sha256'], preparation['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, str(proof_path), str(PREPARATION)), inherited)
    verify(sources)
    verify_artifacts(prefix.replay.OUTPUT, proof['controller_artifact_sha256'])
    verify_artifacts(prefix.replay.native.OUTPUT, proof['worker_artifact_sha256'])
    report = proof['report']
    prefix.boundary(report)
    prior = prefix.replay.native.OUTPUT/prefix.replay.native.CASE[0]
    tape = read_json(prior, 'command_tape.json')
    with closing(prefix.read_rows(prior)) as old, closing(prefix.read_rows(prefix.replay.OUTPUT)) as saved, \
            closing(prefix.read_rows(prefix.replay.observer.OUTPUT)) as visual:
        forecasts = prefix.reconstruct(report, islice(old, prefix.FRAMES), saved, tape, visual)
    verify(sources)
    verify_artifacts(prefix.replay.OUTPUT, proof['controller_artifact_sha256'])
    verify_artifacts(prefix.replay.native.OUTPUT, proof['worker_artifact_sha256'])
    write_json(OUTPUT, dict(status='CHAINED_ANCHOR_NATIVE_PREFIX_SOURCE_AND_ACTUAL_REPLAY_CHECKED',
        source_sha256=sources, controller_completion_sha256=COMPLETION_SHA,
        collector_auditor_preparation_sha256=PREPARATION_SHA,
        original_controller_result_sha256=prefix.completed.RESULT_SHA,
        actual_prefix_observations=prefix.FRAMES, first_intervention_frame=prefix.INTERVENTION,
        original_forecasts_reconstructed=forecasts, future_required_physical_prefix_samples=prefix.PHYSICS_SAMPLES,
        future_required_boundary_command_samples=50, actual_command_changed_at_boundary=False,
        native_launcher_prepared=False, native_simulation_created=False,
        actual_fresh_physical_prefix_compared=False, navigation_qualified=False, goal_achieved=False))
    print('CHAINED_NATIVE_PREFIX_PREPARATION_CHECKED', digest(OUTPUT), len(sources), forecasts, flush=True)


if __name__ == '__main__':
    main()
