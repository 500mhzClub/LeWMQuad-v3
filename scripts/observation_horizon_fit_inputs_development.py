"""Complete short-target/input admission and index-checked private streams."""
from copy import deepcopy
from scripts.check_go2_observation_horizon_inputs_v1 import OUTPUT as CHECK,TARGETS,TARGET_SHA,SWITCH_CHECK_SHA,SEEDS as OPTIMIZATION_SEEDS
from scripts.augmented_family_switch_fit_inputs_development import authenticate as original_authenticate,stream as original_stream
from scripts.observation_horizon_policy_stream_development import ObservationHorizonPolicyStream
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def authenticate(input_check_sha):
    original_authenticate(SWITCH_CHECK_SHA)
    verify_artifacts(CHECK,{'result.json':input_check_sha});checked=read_json(CHECK,'result.json')
    if (checked['status']!='OBSERVATION_HORIZON_INPUTS_COMPLETE' or checked['optimizer_steps']!=0
            or checked['context_slots']!=912 or checked['available_contexts']!=828
            or not checked['all_twenty_four_branch_prefix_tensor_groups_exact']
            or not checked['original_past_tensors_and_command_prefixes_exact']
            or checked['transfer_future_materialization'] is not False
            or checked['optimization_seeds']!=list(OPTIMIZATION_SEEDS)
            or not checked['transition_fit_inputs_validated']):
        raise ValueError('complete unchanged short-horizon input admission required')
    verify_artifacts(CHECK,checked['artifact_sha256']);definition=read_json(CHECK,'launch.json');verify(definition)
    verify_artifacts(TARGETS,definition['target_artifact_sha256'])
    if checked['target_result_sha256']!=TARGET_SHA or definition['target_artifact_sha256']['result.json']!=TARGET_SHA:
        raise ValueError('exact complete short-target population required')
    data=stream(input_check_sha)
    schedules={str(seed):data.view.schedule(updates=1200,batch_size=6,seed=seed) for seed in OPTIMIZATION_SEEDS}
    if schedules!=read_json(CHECK,'training_schedules.json'):raise ValueError('all three original mixed schedules required')
    if len(data.view.indices('train'))!=408 or len(data.view.indices('geometry_transfer'))!=420:
        raise ValueError('complete unchanged source role denominators required')
    return definition,checked,schedules


class CheckedObservationHorizonStream(ObservationHorizonPolicyStream):
    def __init__(self,original,rows,tensor_index):
        super().__init__(original,rows)
        if [r['sample_id'] for r in tensor_index]!=[r['sample_id'] for r in self.view.rows]:
            raise ValueError('complete exact input-check tensor index required')
        self.index=deepcopy(tensor_index)

    def _one(self,index,*,training):
        sample=super()._one(index,training=training);inputs=sample['inputs'] if training else sample
        witness=self.index[index]
        if (witness['materialized'] is not True
                or witness['data_role']!=self.view.rows[index]['data_role']
                or witness['history_sha256']!={k:fingerprint(v.numpy()) for k,v in inputs['observation_history'].items()}
                or witness['known_action_sha256']!=fingerprint(inputs['known_action_blocks'].numpy())
                or witness['known_action_valid_sha256']!=fingerprint(inputs['known_action_valid'].numpy())):
            raise ValueError('short-model input tensors differ from completed admission')
        return sample


def stream(input_check_sha):
    verify_artifacts(CHECK,{'result.json':input_check_sha});checked=read_json(CHECK,'result.json')
    verify_artifacts(CHECK,checked['artifact_sha256']);launch=read_json(CHECK,'launch.json')
    verify_artifacts(TARGETS,launch['target_artifact_sha256'])
    return CheckedObservationHorizonStream(original_stream(SWITCH_CHECK_SHA),read_json(TARGETS,'windows.json'),
        read_json(CHECK,'tensor_index.json'))
