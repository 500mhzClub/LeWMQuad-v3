"""Validate every expanded training context without fitting model weights."""
import json
import time
from pathlib import Path
import torch
from lewm.observation_horizon_plan_development import validate_plan
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.observation_horizon_learning_development import training_loss
from lewm.pulse_timed_learning_development import active_parameters, CONDITIONS
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.all_phase_training_policy_stream_development import (
    AllPhaseTrainingStream, policy_leaves, tensor_bytes)
from scripts.derive_go2_all_phase_training_targets_v1 import (
    OUTPUT as TARGETS, INPUT as ORIGINAL_TARGETS, ROOTS)
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_all_phase_training_inputs_v1_attempt_001'
OLD_INPUTS = BASE/'go2_observation_horizon_inputs_v1_attempt_001'
TARGET_SHA = '4d300f77849d174cc9d7bd2a35d276e0996d1b5ac8795bee4f419291ada6b328'
OLD_INPUT_SHA = '73e11e168f933633dbbd3b82668a5b021c076e18bbfc7704987008b4d290c1b5'
PROTOCOL = 'docs/go2_all_phase_training_inputs_v1_2026-09-10.md'
SOURCE = 'scripts/check_go2_all_phase_training_inputs_v1.py'
TEST = 'lewm/tests/test_all_phase_training_policy_stream_development.py'
ALLOWANCE = 64*1024**2


def input_identity(inputs):
    if set(inputs) != {'observation_history','known_action_blocks','known_action_valid'}:
        raise ValueError('only causal history and known commands may enter model inputs')
    shapes = dict(rgb=(4,3,96,128),body=(4,20,63),control=(4,15,7))
    history = inputs['observation_history']
    if set(history) != set(shapes): raise ValueError('original policy-only modalities required')
    for name,shape in shapes.items():
        value = history[name]
        if (value.shape != shape or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('exact finite original CPU history tensor shape required')
    _,offsets = validate_plan(inputs['known_action_blocks'][None],inputs['known_action_valid'][None],1)
    return dict(history_sha256={k:fingerprint(v.numpy()) for k,v in history.items()},
        known_action_sha256=fingerprint(inputs['known_action_blocks'].numpy()),
        known_action_valid_sha256=fingerprint(inputs['known_action_valid'].numpy()),
        target_offsets_ns=offsets[0].tolist())


def gradient_contract(samples):
    batch = stack_samples(samples); records = []; initial = set()
    for condition in CONDITIONS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026091001); model = ObservationHorizonRGBBodyJEPA(32).cpu()
        before = state_digest(model.state_dict()); initial.add(before)
        model.eval()
        with torch.no_grad(): output = model(**batch['inputs'])
        if not torch.equal(output['target_offsets_ns'],batch['targets']['target_offsets_ns']):
            raise ValueError('new materialized inputs must retain model forecast clocks')
        model.train(); loss,parts = training_loss(model,batch,condition); loss.backward()
        parameters = active_parameters(model,condition)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('finite gradients for every original active parameter required')
        if (state_digest(model.state_dict()) != before
                or any(p.grad is not None for p in model.target_encoder.parameters())):
            raise ValueError('no parameter updates or EMA target gradients permitted')
        records.append(dict(condition=condition,initial_model_sha256=before,loss=float(loss.detach()),
            parts=parts,all_active_gradients_finite=True,parameter_updates=0,
            optimizer_created=False,model_state_unchanged=True))
    if len(initial) != 1: raise ValueError('matched untrained model initialization required')
    return records


def main():
    if not __debug__: raise ValueError('assertions required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive full-population input check')
    resources = hardware()
    if (resources['memory_available_bytes'] < 40*1024**3
            or resources['artifact_free_bytes'] < 40*1024**3+ALLOWANCE):
        raise ValueError('8GiB input check plus32GiB concurrent native and storage reserve required')
    verify_artifacts(TARGETS,{'result.json':TARGET_SHA})
    result = read_json(TARGETS,'result.json'); target_ids = result['artifact_sha256']|{'result.json':TARGET_SHA}
    verify_artifacts(TARGETS,target_ids); old = read_json(TARGETS,'launch.json'); verify(old)
    if (result['status'] != 'ALL_PHASE_TRAINING_TARGETS_V1_COMPLETE'
            or result['source_sha256'] != old['source_sha256']
            or result['report']['context_slots'] != 4800 or result['report']['available_contexts'] != 4010):
        raise ValueError('complete bound expanded training target population required')
    verify_artifacts(ORIGINAL_TARGETS,old['original_target_artifact_sha256'])
    original = read_json(ORIGINAL_TARGETS,'launch.json')
    verify_artifacts(OLD_INPUTS,{'result.json':OLD_INPUT_SHA})
    checked = read_json(OLD_INPUTS,'result.json')
    old_ids = checked['artifact_sha256']|{'result.json':OLD_INPUT_SHA}
    verify_artifacts(OLD_INPUTS,old_ids); verify(read_json(OLD_INPUTS,'launch.json'))
    if checked['status'] != 'OBSERVATION_HORIZON_INPUTS_COMPLETE':
        raise ValueError('original complete input tensor witnesses required')
    old_index = {r['sample_id']:r for r in read_json(OLD_INPUTS,'tensor_index.json')}
    rows = read_json(TARGETS,'windows.json')
    bindings = {s:original['source_collection_artifact_sha256'][str(root)] for s,root in ROOTS.items()}
    consumed = {s:{} for s in ROOTS}
    for row in rows:
        if not row['available']: continue
        names,_,_ = policy_leaves(row,include_future=True)
        for name in names:
            path = row['trial']+'/'+name; consumed[row['source']][path] = bindings[row['source']][path]
    for source,ids in consumed.items(): verify_artifacts(ROOTS[source],ids)
    inherited = dict(result['source_sha256'])
    for name,sha in checked['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('incompatible old input source: '+name)
        inherited[name] = sha
    sources = discover_sources((SOURCE,TEST,PROTOCOL),inherited)
    launch = old | dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        all_phase_target_result_sha256=TARGET_SHA,all_phase_target_artifact_sha256=target_ids,
        original_input_result_sha256=OLD_INPUT_SHA,original_input_artifact_sha256=old_ids,
        consumed_training_policy_sha256=consumed,hardware=resources,
        memory_allowance_bytes=8*1024**3,concurrent_native_allowance_bytes=32*1024**3,
        output_allowance_bytes=ALLOWANCE,training_sample_cache_bytes=0,
        optimizer_updates=0,model_parameter_updates=0,native_scene_workers=0,
        private_training_future_materialization=True,geometry_transfer_future_materialization=False,
        full_population_causal_input_comparison_required=True)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch)
    started = time.perf_counter(); print('ALL_PHASE_TRAINING_INPUTS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        stream = AllPhaseTrainingStream(rows,consumed,maximum_cache_bytes=0)
        index = []; matched = available = 0; representative = []; maximum_sample_bytes = 0
        for i,row in enumerate(rows):
            record = dict(index=i,sample_id=row['sample_id'],source=row['source'],data_role='train',
                materialized=row['available'],control_phase_modulo_five=row['control_phase_modulo_five'])
            if row['available']:
                inputs = stream.materialize(i,training=False); identity = input_identity(inputs)
                past_scope = dict(stream.last_access)
                sample = stream.materialize(i,training=True)
                if identity != input_identity(sample['inputs']):
                    raise ValueError('private future targets must not change any causal input tensor')
                if identity['target_offsets_ns'] != sample['targets']['target_offsets_ns'].tolist():
                    raise ValueError('target clocks must equal known action prefix clocks')
                old_id = row['original_sample_id']
                if old_id is not None:
                    witness = old_index[old_id]
                    if (witness['materialized'] is not True or witness['data_role'] != 'train'
                            or any(witness[k] != v for k,v in identity.items())):
                        raise ValueError('all original available training input tensors must remain exact')
                    matched += 1
                size = tensor_bytes(sample); maximum_sample_bytes = max(maximum_sample_bytes,size)
                if len(representative) < 6: representative.append(sample)
                record.update(**identity,original_sample_id=old_id,original_inputs_exact=old_id is not None,
                    inference_and_training_inputs_exact=True,past_access=past_scope,
                    training_access=dict(stream.last_access),sample_tensor_bytes=size,
                    future_tensor_sha256={k:fingerprint(v.numpy()) for k,v in sample['targets']['future_observations'].items()},
                    target_tensor_sha256={k:fingerprint(v.numpy()) for k,v in sample['targets'].items() if k!='future_observations'})
                available += 1
                if available%128==0: print('ALL_PHASE_TRAINING_INPUTS_PROGRESS',available,'index',i,flush=True)
            index.append(record)
        if len(index)!=4800 or available!=4010 or matched!=408:
            raise ValueError('complete expanded and exact old training input populations required')
        contracts = gradient_contract(representative)
        write_json(OUTPUT/'tensor_index.json',index)
        write_json(OUTPUT/'model_contract_checks.json',contracts)
        if sum((OUTPUT/n).stat().st_size for n in ('launch.json','tensor_index.json','model_contract_checks.json')) > ALLOWANCE-1024**2:
            raise ValueError('input-check metadata allowance exceeded; evidence retained')
        verify(launch); verify_artifacts(TARGETS,target_ids); verify_artifacts(OLD_INPUTS,old_ids)
        verify_artifacts(ORIGINAL_TARGETS,old['original_target_artifact_sha256'])
        for source,ids in consumed.items(): verify_artifacts(ROOTS[source],ids)
        ids = {n:digest(OUTPUT/n) for n in ('launch.json','tensor_index.json','model_contract_checks.json')}
        verify_artifacts(OUTPUT,ids)
        report = dict(context_slots=4800,materialized_training_contexts=available,
            original_available_training_inputs_exact=matched,all_causal_and_training_inputs_exact=True,
            maximum_sample_tensor_bytes=maximum_sample_bytes,
            full_4010_sample_tensor_cache_bytes=4010*maximum_sample_bytes,
            runtime_training_sample_cache_bytes=stream.cache_bytes,
            consumed_training_policy_leaves=sum(len(v) for v in consumed.values()),
            separate_past_and_future_packet_scopes_verified=True,
            geometry_transfer_future_materialization=False,native_artifacts_opened_by_stream=False,
            untrained_model_contract_conditions=[r['condition'] for r in contracts],
            optimizer_updates=0,parameter_updates=0,matched_retraining_completed=False,
            native_execution=False,navigation_qualified=False)
        write_json(OUTPUT/'result.json',dict(status='ALL_PHASE_TRAINING_INPUTS_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,
            all_phase_target_result_sha256=TARGET_SHA,original_input_result_sha256=OLD_INPUT_SHA,
            wall_s=time.perf_counter()-started,hardware_after=hardware(),goal_achieved=False))
        print('ALL_PHASE_TRAINING_INPUTS_COMPLETE',digest(OUTPUT/'result.json'),report,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ALL_PHASE_TRAINING_INPUT_CHECK_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
