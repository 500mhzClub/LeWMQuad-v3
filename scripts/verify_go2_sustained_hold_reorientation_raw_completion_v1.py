"""Reconstruct the completed sustained-turn raw prefix without another model replay."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from scripts import sustained_hold_reorientation_native_prefix_development as prefix
from scripts import sustained_hold_reorientation_native_inputs_development as inputs
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.maze_decision_stream_development import NAME

SOURCE = 'scripts/verify_go2_sustained_hold_reorientation_raw_completion_v1.py'
EXECUTION = 'docs/go2_sustained_hold_reorientation_raw_execution_2026-09-11.json'
EXECUTION_SHA = '90fd2434238a5af8fa3c3c2a0bd02193704b52686a3dc1deeaba39b7d9c3a7fa'
PREPARATION = 'docs/go2_sustained_hold_reorientation_native_prefix_preparation_2026-09-11.json'
PREPARATION_SHA = 'a50a56836c91d882408a3ddb8f64b17bae718ba7052438e46dea9eac02b91d84'
OUTPUT = ROOT/'docs/go2_sustained_hold_reorientation_raw_completion_verification_2026-09-11.json'


def prepared_sources():
    verify({EXECUTION:EXECUTION_SHA, PREPARATION:PREPARATION_SHA})
    execution = json.loads((ROOT/EXECUTION).read_text())
    prepared = json.loads((ROOT/PREPARATION).read_text())
    if (execution['owner'] != inputs.RAW_OWNER or execution['boot_id'] != BOOT
            or execution['launch_sha256'] != prefix.LAUNCH_SHA
            or prepared['status'] != 'SUSTAINED_RAW_COMPLETION_AND_NATIVE_PREFIX_HELPER_PREPARED'):
        raise ValueError('original recorded execution and tested prefix verifier required')
    sources = discover_sources((SOURCE, EXECUTION, PREPARATION), prepared['source_sha256'])
    verify(sources)
    return sources, execution


def require_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT or owner_live(inputs.RAW_OWNER):
        raise ValueError('original sustained raw owner must have ended on recorded boot')


def verify_completed(result_sha, sources, execution):
    require_ended()
    if type(result_sha) is not str or re.fullmatch('[0-9a-f]{64}', result_sha) is None:
        raise ValueError('exact completed result SHA-256 required')
    run = prefix.replay; root = run.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original raw failure must be preserved')
    fixed = {'result.json':result_sha, 'launch.json':prefix.LAUNCH_SHA}
    verify_artifacts(root, fixed)
    result = read_json(root, 'result.json'); launch = read_json(root, 'launch.json')
    if (result['source_sha256'] != launch['source_sha256']
            or result['source_sha256'] != execution['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or set(result['artifact_sha256']) != {'launch.json', NAME}
            or result['artifact_sha256']['launch.json'] != prefix.LAUNCH_SHA
            or result['goal_achieved'] is not False
            or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0
            or launch['boot_id'] != BOOT or launch['owner_pid'] != inputs.RAW_OWNER['pid']
            or launch['input_admission'] != execution['launch_input_admission']):
        raise ValueError('same complete original execution, artifacts and source closure required')
    # This recomputes the original hold completion/input bindings, with no new
    # model forecast or full training-ancestry reconstruction.
    if run.admit(sources) != launch['input_admission']:
        raise ValueError('complete original raw-input admission must reconstruct')
    report = prefix.admit_prefix(root, result)
    require_ended(); verify(sources); verify_artifacts(root, result['artifact_sha256'] | fixed)
    return dict(status='SUSTAINED_HOLD_REORIENTATION_RAW_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        original_source_count=len(result['source_sha256']), result_sha256=result_sha,
        artifact_sha256=result['artifact_sha256'], original_owner=inputs.RAW_OWNER, original_owner_ended=True,
        report=report, original_input_admission_reconstructed=True,
        complete_saved_decisions_and_forecasts_recompared=True, all_raw_public_packet_fingerprints_recomputed=True,
        no_observation_after_first_changed_request_consumed=True,
        raw_sensor_model_replay_reexecuted=False, full_training_ancestry_reexecuted=False,
        changed_command_executed=False, native_execution=False, navigation_qualified=False, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--result-sha256')
    parser.add_argument('--source-preflight-only', action='store_true'); args = parser.parse_args()
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completion verification output required')
    sources, execution = prepared_sources()
    if args.source_preflight_only:
        print('SUSTAINED_RAW_COMPLETION_SOURCE_PREFLIGHT', len(sources), flush=True); return
    result = verify_completed(args.result_sha256, sources, execution)
    write_json(OUTPUT, result)
    print('SUSTAINED_RAW_COMPLETION_VERIFIED', digest(OUTPUT), len(sources), result['report']['frames'], flush=True)


if __name__ == '__main__': main()
