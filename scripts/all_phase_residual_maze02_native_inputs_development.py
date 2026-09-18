"""Admit exact expanded models, old maze2 reference and completed native owner."""
from scripts import run_go2_recent_qualified_direct_flow_maze03_pilot_v1 as native
from scripts import run_go2_residual_anchored_continuation_maze_pilot_v1 as reference
from scripts import await_go2_all_phase_translation_bias_v1 as corrected_wait
from scripts.all_phase_translation_bias_model_admission_development import admit as admit_correction
from scripts.all_phase_fit_execution_development import OUTPUT as FITS
from scripts.run_go2_prepared_native_queue_v1 import JOBS, authenticate_completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit, merge_sources
from lewm.all_phase_residual_maze02_study_development import FROZEN

WAIT_SHA='4329326dc8b3d07983bde26f7149141d9aee7f2eccb57343949a197b4b935238'
NATIVE_LAUNCH_SHA='25c34387e44b035f05a4f372ecf70cd337e875de508ba3a9b037ccf741da0eb7'
REFERENCE_SHA='818a598ca6336866cf5f4768c11edaf67c8c1ca60896c305f93f69fd0ed5230c'
REFERENCE_LAUNCH_SHA='c4681e31baaf5dc1c8fa368854ddbcf19866090787741b946c4d37b9a3a477b3'
NATIVE_OWNER=dict(pid=2636286,created=1789017984.16,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',
    'scripts/run_go2_recent_qualified_direct_flow_maze03_pilot_v1.py',
    '--prefix-result-sha256',native.PREFIX_SHA,'--contact-wait-result-sha256',
    'dc57fe4cd5fb8083ec6ac43ae11d85e6d3e048db62d9b03f06361eb621378536'])
CORRECTION_OWNER=dict(pid=2639661,created=1789019633.03,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',corrected_wait.SOURCE])


def prepared_sources(seeds):
    inherited=dict(FROZEN)
    for root,sha in ((native.OUTPUT,NATIVE_LAUNCH_SHA),(reference.OUTPUT,REFERENCE_LAUNCH_SHA),
            (corrected_wait.OUTPUT,WAIT_SHA)):
        verify_artifacts(root,{'launch.json':sha})
        inherited=merge_sources(inherited,read_json(root,'launch.json')['source_sha256'])
    verify(inherited);sources=discover_sources(seeds,inherited);verify(sources)
    return sources


def completed(root,sha,launch_sha,sources):
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('completed predecessor failed; preserve its evidence')
    verify_artifacts(root,{'result.json':sha,'launch.json':launch_sha})
    result=read_json(root,'result.json');launch=read_json(root,'launch.json')
    ids=result['artifact_sha256']|{'result.json':sha}
    if (ids['launch.json']!=launch_sha or result['source_sha256']!=launch['source_sha256']
            or any(sources.get(n)!=h for n,h in result['source_sha256'].items())):
        raise ValueError('exact completed original source and launch identities required')
    verify_artifacts(root,ids);return result,launch,ids


def admit_native(result,launch,audit,prefix,terminal):
    if (result['status']!='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE'
            or len(result['conditions'])!=1 or result['conditions'][0]!=terminal
            or terminal['status']!='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_COLLECTED_AND_RAW_AUDITED'
            or terminal['case']!=native.CASE[0] or terminal['layout_index']!=3
            or launch['planned_case']!=list(native.CASE) or 'failure' in terminal
            or terminal['prefix_comparison']!=prefix or terminal['model_state_unchanged'] is not True
            or result['measured_round_trip_successes']!=int(terminal['verified_round_trip'])):
        raise ValueError('complete original maze3 worker and actual prefix required')
    require_raw_audit(terminal,audit,learned=True)
    for flag in ('physical_and_public_prefix_exact','all_preintervention_requested_commands_exact',
            'complete_candidate_decisions_match_prospective_prefix','candidate_intervention_command_completed'):
        if prefix[flag] is not True:raise ValueError('original maze3 physical intervention failed: '+flag)
    if (prefix['common_prefix_frames'],prefix['first_intervention_frame'],prefix['physical_prefix_samples'])!=(1207,1206,61050):
        raise ValueError('fixed original maze3 intervention boundary required')
    if any(result['artifact_sha256'].get(n)!=h for n,h in terminal['artifact_sha256'].items()):
        raise ValueError('all original native worker artifacts required')
    return dict(measured_round_trip_successes=result['measured_round_trip_successes'],
        all_raw_audits_pass=True,physical_prefix_pass=True,scientific_success_required=False)


def admit_inputs(wait_result_sha,native_result_sha,sources):
    if corrected_wait.owner_live(NATIVE_OWNER) or corrected_wait.owner_live(CORRECTION_OWNER):
        raise ValueError('original native and correction owners must finish before admission')
    waited,_,wait_ids=completed(corrected_wait.OUTPUT,wait_result_sha,WAIT_SHA,sources)
    if (waited['status']!='ALL_PHASE_TRANSLATION_BIAS_WAIT_COMPLETE'
            or waited['all_models']!=18 or waited['trained_heads']!=30 or waited['fitted_scalars']!=480
            or waited['all_coefficients_reconstructed'] is not True
            or waited['native_execution'] is not False or waited['automatic_retry'] is not False):
        raise ValueError('original complete expanded correction handoff required')
    correction=admit_correction(waited['correction_result_sha256'])
    if (correction!=read_json(corrected_wait.OUTPUT,'correction_admission.json')
            or correction['base_admission']['study_result_sha256']!=waited['fit_result_sha256']):
        raise ValueError('complete correction and full-fit admission must reproduce exactly')
    result,launch,native_ids=completed(native.OUTPUT,native_result_sha,NATIVE_LAUNCH_SHA,sources)
    name=native.CASE[0]
    for suffix in ('_audit.json','_prefix_comparison.json','_worker_terminal.json','_worker.log'):
        if name+suffix not in native_ids:raise ValueError('complete original native evidence must be bound')
    native.verify_inputs(launch)
    latest=admit_native(result,launch,read_json(native.OUTPUT,name+'_audit.json'),
        read_json(native.OUTPUT,name+'_prefix_comparison.json'),read_json(native.OUTPUT,name+'_worker_terminal.json'))
    _,_,reference_ids=completed(reference.OUTPUT,REFERENCE_SHA,REFERENCE_LAUNCH_SHA,sources)
    original=authenticate_completed(JOBS[1],sources,expected_launch_sha=REFERENCE_LAUNCH_SHA)
    if original['result_sha256']!=REFERENCE_SHA:raise ValueError('unchanged original maze2 reference required')
    admission=dict(correction_wait_result_sha256=wait_result_sha,correction_wait_artifact_sha256=wait_ids,
        correction_admission=correction,native_result_sha256=native_result_sha,native_artifact_sha256=native_ids,
        native_completion=latest,reference_result_sha256=REFERENCE_SHA,reference_artifact_sha256=reference_ids,
        reference_completion=original,complete_original_verifiers_reexecuted=True)
    verify_bound_inputs(admission,sources)
    return admission


def verify_bound_inputs(admission,sources):
    if admission['complete_original_verifiers_reexecuted'] is not True:
        raise ValueError('complete original input admission required')
    verify(sources)
    for n,h in FROZEN.items():
        if sources.get(n)!=h:raise ValueError('unchanged prospective controller and science required')
    correction=admission['correction_admission'];base=correction['base_admission']
    for root,ids in ((corrected_wait.OUTPUT,admission['correction_wait_artifact_sha256']),
            (native.OUTPUT,admission['native_artifact_sha256']),
            (reference.OUTPUT,admission['reference_artifact_sha256']),
            (corrected_wait.CORRECTION,correction['correction_artifact_sha256']),
            (FITS,base['fit_artifact_sha256']|{'result.json':base['study_result_sha256']})):
        verify_artifacts(root,ids)
    if (admission['correction_wait_artifact_sha256']['result.json']!=admission['correction_wait_result_sha256']
            or admission['native_artifact_sha256']['result.json']!=admission['native_result_sha256']
            or admission['reference_artifact_sha256']['result.json']!=REFERENCE_SHA
            or admission['reference_result_sha256']!=REFERENCE_SHA):
        raise ValueError('complete original admitted result bindings required')
