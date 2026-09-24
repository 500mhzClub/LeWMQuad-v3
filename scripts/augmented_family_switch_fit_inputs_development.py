"""Authenticate both complete development input populations for a fixed mixture."""
from lewm.moving_action_switch_family_development import TRIALS
from lewm.moving_action_switch_accounting_development import summarize
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView
from lewm.augmented_family_switch_view_development import AugmentedFamilySwitchView
from scripts.family_transition_fit_inputs_development import authenticate as family_authenticate,stream as family_stream,CHECK_SHA as FAMILY_CHECK_SHA
from scripts.check_go2_moving_action_switch_inputs_v1 import OUTPUT as SWITCH_CHECK,OPTIMIZATION_SEEDS
from scripts.moving_action_switch_policy_stream_development import INPUT as SWITCH_INPUT,MovingActionSwitchPolicyStream
from scripts.augmented_family_switch_stream_development import AugmentedFamilySwitchStream
from scripts.moving_action_switch_runtime_development import verify
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json


def authenticate(switch_check_sha):
    family_definition,_,_=family_authenticate()
    verify_artifacts(SWITCH_CHECK,{'result.json':switch_check_sha});checked=read_json(SWITCH_CHECK,'result.json')
    if (checked['status']!='MOVING_ACTION_SWITCH_INPUTS_COMPLETE' or checked['cells']!=144
            or checked['optimizer_steps']!=0 or not checked['complete_measurement_and_prefix_gates_pass']
            or not checked['transition_fit_inputs_validated'] or checked['optimization_seeds']!=list(OPTIMIZATION_SEEDS)):
        raise ValueError('complete unchanged 144-cell input check required')
    verify_artifacts(SWITCH_CHECK,checked['artifact_sha256']);launch=read_json(SWITCH_CHECK,'launch.json');verify(launch)
    verify_artifacts(SWITCH_INPUT,launch['collection_sha256']);collection=read_json(SWITCH_INPUT,'result.json')
    if checked['collection_result_sha256']!=launch['collection_sha256']['result.json']:
        raise ValueError('exact collection/input-check identity required')
    reports=[read_json(SWITCH_INPUT,t+'_audit.json') for t in TRIALS];summary=summarize(reports)
    if not summary['all_measurement_and_prefix_gates_pass'] or any(collection[k]!=v for k,v in summary.items()):
        raise ValueError('full native population accounting must reproduce')
    branch=MovingActionSwitchView(reports);saved=read_json(SWITCH_CHECK,'training_schedules.json')
    if saved!={str(seed):branch.schedule(updates=1200,batch_size=6,seed=seed) for seed in OPTIMIZATION_SEEDS}:
        raise ValueError('complete fixed branch schedules required')
    for field in ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
            'opencv_binary_sha256','opencv_version','rules'):
        if family_definition[field]!=launch[field]:raise ValueError('source populations disagree on native/input identity')
    sources=dict(family_definition['source_sha256'])
    for name,h in checked['source_sha256'].items():
        if name in sources and sources[name]!=h:raise ValueError('inconsistent source identities across populations')
        sources[name]=h
    definition=launch|dict(source_sha256=sources);verify(definition)
    view=AugmentedFamilySwitchView(family_stream().view,branch)
    schedules={str(seed):view.schedule(updates=1200,batch_size=6,seed=seed) for seed in OPTIMIZATION_SEEDS}
    return definition,checked,schedules


def stream(switch_check_sha):
    verify_artifacts(SWITCH_CHECK,{'result.json':switch_check_sha})
    checked=read_json(SWITCH_CHECK,'result.json');verify_artifacts(SWITCH_CHECK,checked['artifact_sha256'])
    launch=read_json(SWITCH_CHECK,'launch.json');verify_artifacts(SWITCH_INPUT,{'result.json':checked['collection_result_sha256']})
    reports=[read_json(SWITCH_INPUT,t+'_audit.json') for t in TRIALS]
    branch=MovingActionSwitchPolicyStream(MovingActionSwitchView(reports),output=SWITCH_INPUT,
        bindings=launch['collection_sha256'],tensor_index=read_json(SWITCH_CHECK,'tensor_index.json'))
    return AugmentedFamilySwitchStream(family_stream(),branch)
