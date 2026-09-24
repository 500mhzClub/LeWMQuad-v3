"""Authenticate ended owners and reconstruct the full external profile summary."""
import argparse
from datetime import datetime, timezone

from scripts import run_go2_body_projection_external_profile_v1 as run
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts

OUTPUT = ROOT/'docs/go2_body_projection_external_profile_completion_verification_2026-09-11.json'
ARTIFACTS = {'launch.json','execution.json','child_execution.json','profiler_execution.json',
    'child_ready.json','child_result.json','comparison.jsonl','profile.json',
    'child_stdout.txt','child_stderr.txt','profiler_stdout.txt','profiler_stderr.txt'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result-sha256',required=True)
    parser.add_argument('--launch-sha256',required=True)
    args=parser.parse_args()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive external profile completion verification required')
    launch=run.authenticate_launch(args.launch_sha256)
    if run.owner_live(launch['owner']):
        raise ValueError('original parent must be ended before final verification')
    verify_artifacts(run.OUTPUT, {'result.json':args.result_sha256})
    result=run.read('result.json')
    if (result['status'] != 'BODY_PROJECTION_EXTERNAL_PROFILE_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or set(result['artifact_sha256']) != ARTIFACTS
            or result['artifact_sha256']['launch.json'] != args.launch_sha256
            or result['actual_reference_reauthenticated_after_child_exit'] is not True
            or result['controller'] != 'BodyProjectedTiledController'
            or any(result[key] is not False for key in ('native_execution','real_time_qualified',
                'navigation_qualified','goal_achieved'))):
        raise ValueError('complete original profile artifacts and negative scope required')
    verify_artifacts(run.OUTPUT,result['artifact_sha256'])
    prior=run.admission.admit_completed()
    child,summary=run.inspect_completed_child(args.launch_sha256,prior)
    if result['summary'] != summary or result['sensing_scope'] != child['sensing_scope']:
        raise ValueError('complete sampled summary and original negative sensing scope must reconstruct')
    verify(result['source_sha256'])
    verify_artifacts(run.OUTPUT,result['artifact_sha256'] | {'result.json':args.result_sha256})
    write_json(OUTPUT,dict(status='BODY_PROJECTION_EXTERNAL_PROFILE_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=result['source_sha256'],
        result_sha256=args.result_sha256, launch_sha256=args.launch_sha256,
        artifact_sha256=result['artifact_sha256'], original_parent_ended=True,
        original_child_and_profiler_ended=True, actual_raw_model_reference_reauthenticated=True,
        original_rows=1428, original_forecasts=1425, original_state_witnesses=7,
        sampled_observations=30, summary=summary, sensing_scope=result['sensing_scope'],
        descriptive_state_size_snapshots_independently_reconstructed=False,
        profiler_overhead_removed=False, native_execution=False, real_time_qualified=False,
        navigation_qualified=False, goal_achieved=False))
    print('BODY_PROJECTION_EXTERNAL_PROFILE_COMPLETION_VERIFIED',digest(OUTPUT),flush=True)


if __name__ == '__main__':
    main()
