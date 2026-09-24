"""Audit completed paired registration evidence; do not rerun registration."""
from contextlib import closing
from itertools import islice
import json
import math
from pathlib import Path
import re

from scripts import replay_go2_density_routed_floor_registration_prefix_v1 as run
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE='scripts/verify_go2_density_routed_floor_registration_completion_v1.py'
TEST='lewm/tests/test_density_routed_floor_registration_completion_development.py'
EXECUTION='docs/go2_density_routed_floor_registration_execution_2026-09-11.json'
EXECUTION_SHA='4aca6e8a962d12beb5efbe6cfc2eb65bb3779cbfe052e0928fb3173f2610680f'
OUTPUT=Path('docs/go2_density_routed_floor_registration_completion_verification_2026-09-11.json')


def validate_row(row, frame):
    if type(row.get('frame')) is not int or row['frame']!=frame:
        raise ValueError('every ordered comparison frame required')
    order=['original','candidate'] if frame%2==0 else ['candidate','original']
    if row.get('order')!=order:raise ValueError('alternating pair order required')
    for key in ('recorded_evidence_exact','complete_registration_state_exact','public_inputs_unchanged'):
        if row.get(key) is not True:raise ValueError('strict successful comparison required: '+key)
    if re.fullmatch('[0-9a-f]{64}',row.get('public_input_sha256','')) is None:
        raise ValueError('exact public input identity required')
    times=row.get('wall_ms')
    if not isinstance(times,dict) or set(times)!={'original','candidate'}:
        raise ValueError('both timing populations required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in times.values()):
        raise ValueError('finite positive times required')
    return times


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive completion audit required')
    verify({EXECUTION:EXECUTION_SHA})
    execution=json.loads(Path(EXECUTION).read_text())
    if run.reference.observer.owner_live(execution['owner']):raise ValueError('paired replay must be ended')
    verify_artifacts(run.OUTPUT,{'launch.json':execution['launch_sha256']})
    launch=read_json(run.OUTPUT,'launch.json');result=read_json(run.OUTPUT,'result.json')
    result_sha=digest(run.OUTPUT/'result.json')
    if result.get('status')!='DENSITY_ROUTED_FLOOR_REGISTRATION_PREFIX_V1_COMPLETE':
        raise ValueError('completed paired replay required')
    if result['source_sha256']!=launch['source_sha256']:raise ValueError('same execution sources required')
    if launch['owner_pid']!=execution['owner']['pid'] or launch['boot_id']!=execution['owner']['boot_id']:
        raise ValueError('same actual process owner required')
    if set(result['artifact_sha256'])!={'launch.json','comparison.jsonl'}:
        raise ValueError('complete fixed replay artifacts required')
    if result['artifact_sha256']['launch.json']!=execution['launch_sha256']:
        raise ValueError('same launch identity required')
    verify_artifacts(run.OUTPUT,result['artifact_sha256'])
    sources=discover_sources((SOURCE,TEST,EXECUTION),launch['source_sha256']);verify(sources)
    inputs=run.reference.observer.admit_worker(sources)
    if inputs!=launch['input_artifact_sha256']:raise ValueError('same original recorded inputs required')
    for key in ('all_recorded_registration_receipts_exact','all_registration_states_exact','public_inputs_unchanged','shared_host_timings'):
        if result.get(key) is not True:raise ValueError('strict completed result flags required')
    for key in ('model_loaded','raw_observer_reexecuted','full_controller_reexecuted','following_observation_consumed',
                'native_execution','real_time_qualified','navigation_qualified','goal_achieved'):
        if result.get(key) is not False:raise ValueError('component replay scope must remain explicit')
    comparisons=[];totals=dict(original=0,candidate=0)
    with (run.OUTPUT/'comparison.jsonl').open() as rows:
        for frame,line in enumerate(rows):
            if frame>=run.FRAMES:raise ValueError('no following comparison permitted')
            row=json.loads(line);times=validate_row(row,frame);comparisons.append(row)
            for name in totals:totals[name]+=times[name]
    if len(comparisons)!=run.FRAMES or result['frames']!=run.FRAMES or launch['frames']!=run.FRAMES:
        raise ValueError('all 854 paired observations required')
    if totals!=result['total_registration_wall_ms']:raise ValueError('timing totals must reconstruct exactly')
    reduction=100*(totals['original']-totals['candidate'])/totals['original']
    if reduction!=result['total_time_reduction_percent']:raise ValueError('timing reduction must reconstruct exactly')
    directory=run.reference.native.OUTPUT/run.reference.native.CASE[0]
    reader=run.reference.observer.IntentReturnRGBDReplay(directory)
    acquisitions=read_json(directory,'auxiliary_camera_audit.json');checked=0
    with closing(run.reference.observer.read_rows(directory)) as rows:
        for frame,row in enumerate(islice(rows,run.FRAMES)):
            if row['tick']!=frame or row['observation_index']!=frame:raise ValueError('ordered actual input required')
            p,d,f,now=reader.packet(frame)
            image,aux=run.reference.observer.packet(directory,frame,p,
                run.reference.observer.public_acquisition(acquisitions[frame]),now_ns=now)
            saved_raw=row['decision']['original_visual_evidence']
            raw=dict(saved_raw,identity=run._identity(tuple(saved_raw['identity'])))
            if run.reference.observer.fingerprint((p,d,f,image,aux,raw))!=comparisons[frame]['public_input_sha256']:
                raise ValueError('actual original public input fingerprint mismatch')
            checked+=1
    if checked!=run.FRAMES:raise ValueError('complete actual input population required')
    verify(sources);verify_artifacts(run.reference.native.OUTPUT,inputs)
    verify_artifacts(run.OUTPUT,result['artifact_sha256']|{'result.json':result_sha})
    write_json(OUTPUT,dict(status='DENSITY_ROUTED_FLOOR_REGISTRATION_COMPLETION_VERIFIED',
        source_sha256=sources,execution_sha256=EXECUTION_SHA,launch_sha256=execution['launch_sha256'],
        result_sha256=result_sha,frames=checked,total_registration_wall_ms=totals,reduction_percent=reduction,
        all_actual_public_input_fingerprints_reconstructed=True,all_ordered_comparison_flags_strict=True,
        paired_timing_totals_reconstructed=True,source_and_raw_artifact_bindings_verified=True,
        numerical_registration_independently_reexecuted=False,full_controller_reexecuted=False,
        candidate_adopted=False,native_execution=False,real_time_qualified=False,goal_achieved=False))
    print('DENSITY_ROUTED_REGISTRATION_COMPLETION_VERIFIED',digest(OUTPUT),reduction,flush=True)


if __name__=='__main__':main()
