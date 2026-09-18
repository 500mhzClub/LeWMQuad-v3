"""Complete short-horizon causal tensor admission; no optimizer or native run."""
from collections import defaultdict
import json
import time
import cv2
import torch
from lewm.observation_horizon_plan_development import validate_plan
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.observation_horizon_learning_development import training_loss
from lewm.observation_horizon_input_ablation_development import transform_inputs,transform_training_batch
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.derive_go2_observation_horizon_family_v1 import OUTPUT as TARGETS,SWITCH_CHECK_SHA
from scripts.augmented_family_switch_fit_inputs_development import authenticate,stream
from scripts.observation_horizon_policy_stream_development import ObservationHorizonPolicyStream
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_observation_horizon_inputs_v1_attempt_001'
PROTOCOL='docs/go2_observation_horizon_inputs_v1_2026-09-08.md'
TARGET_SHA='fe3ab252e6da0ebadba13927c0dad7410d2084145c3b50439f6848ea5b65a775'
SEEDS=(2026091001,2026091401,2026091402)


def main():
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive short-horizon tensor admission')
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('input-check resource allowance unavailable')
    authenticate(SWITCH_CHECK_SHA)
    verify_artifacts(TARGETS,{'result.json':TARGET_SHA});target=read_json(TARGETS,'result.json')
    if (target['status']!='OBSERVATION_HORIZON_FAMILY_TARGETS_COMPLETE' or target['context_slots']!=912
            or target['available_contexts']!=828 or not target['all_shared_half_second_native_targets_exact']):
        raise ValueError('complete unchanged short-horizon derivation required')
    target_ids={'result.json':TARGET_SHA,**target['artifact_sha256']};verify_artifacts(TARGETS,target_ids)
    original=read_json(TARGETS,'launch.json');verify(original)
    data=ObservationHorizonPolicyStream(stream(SWITCH_CHECK_SHA),read_json(TARGETS,'windows.json'))
    schedules={str(seed):data.view.schedule(updates=1200,batch_size=6,seed=seed) for seed in SEEDS}
    if schedules!=read_json(TARGETS,'training_schedules.json'):raise ValueError('unchanged complete three-seed schedules required')
    sources=discover_sources((PROTOCOL,'scripts/check_go2_observation_horizon_inputs_v1.py',
        'lewm/tests/test_observation_horizon_learning_development.py','lewm/tests/test_observation_horizon_stream_development.py',
        'docs/go2_observation_horizon_family_targets_result_2026-09-08.md'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),target_artifact_sha256=target_ids,
        hardware=resources,workers=1,threads=1,private_untrained_model_seed=2026091600,
        optimizer_steps=0,native_execution=False,model_training=False,transfer_future_materialization=False,
        concurrency_reason='single bounded private training tensor cache; no optimizer or native scene')
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    print('OBSERVATION_HORIZON_INPUTS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        index=[];counts={role:dict(contexts=0,future_observations_materialized=0,motion_targets=0,contact_targets=0,contact_positives=0)
            for role in ('train','geometry_transfer')};groups=defaultdict(list)
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for i,row in enumerate(data.view.rows):
                witness=dict(sample_id=row['sample_id'],source=row['source'],data_role=row['data_role'],materialized=row['available'])
                if row['available']:
                    role=row['data_role'];counts[role]['contexts']+=1
                    if role=='train':
                        sample=data.training_batch([i]);inputs=sample['inputs'];t=sample['targets']
                        counts[role]['future_observations_materialized']+=int(t['future_valid'].sum())
                        counts[role]['motion_targets']+=int(t['motion_valid'].sum())
                        counts[role]['contact_targets']+=int(t['contact_valid'].sum())
                        counts[role]['contact_positives']+=int(t['contact'][t['contact_valid']].sum())
                    else:inputs=data.inference_batch([i],role=role)
                    active,offsets=validate_plan(inputs['known_action_blocks'],inputs['known_action_valid'],1)
                    if active[0].tolist()!=[t['in_plan'] for t in row['targets']] or offsets[0].tolist()!=[t['offset_ns'] for t in row['targets']]:
                        raise ValueError('actual tensor/target clock disagreement')
                    witness.update(history_sha256={k:fingerprint(v[0].numpy()) for k,v in inputs['observation_history'].items()},
                        known_action_sha256=fingerprint(inputs['known_action_blocks'][0].numpy()),
                        known_action_valid_sha256=fingerprint(inputs['known_action_valid'][0].numpy()),
                        target_offsets_ns=offsets[0].tolist())
                    if row['source']=='switch':groups[(row['cluster'],row['prefix_action'])].append(witness['history_sha256'])
                index.append(witness)
                if i%64==0:monitor.write(json.dumps(dict(index=i,elapsed_s=time.perf_counter()-started,**hardware()))+'\n');monitor.flush()
        assert counts['train']['contexts']==408 and counts['geometry_transfer']['contexts']==420 and len(data._cache)==408
        assert len(groups)==24 and all(len(v)==6 and all(r==v[0] for r in v) for v in groups.values())
        ids=schedules[str(SEEDS[0])]['batches'][0];sample=data.training_batch(ids)
        with torch.random.fork_rng(devices=[]):torch.manual_seed(2026091600);model=ObservationHorizonRGBBodyJEPA(32).eval()
        before=state_digest(model.state_dict());compatibility=[]
        with torch.inference_mode():
            for variant in ('full','no_rgb'):
                output=model(**transform_inputs(sample['inputs'],input_variant=variant))
                active,offsets=validate_plan(sample['inputs']['known_action_blocks'],sample['inputs']['known_action_valid'],6)
                assert torch.equal(output['prediction_valid'],active) and torch.equal(output['target_offsets_ns'],offsets)
                for head in ('direct_outcomes','rollout_outcomes'):
                    assert output[head].shape==(6,8,5) and torch.isfinite(output[head]).all() and not output[head][~active].any()
                for condition in ('direct','supervised_rollout','jepa'):
                    loss,parts=training_loss(model,transform_training_batch(sample,input_variant=variant),condition)
                    assert torch.isfinite(loss);compatibility.append(dict(variant=variant,condition=condition,finite_loss=True))
        assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        write_json(OUTPUT/'tensor_index.json',index);write_json(OUTPUT/'training_schedules.json',schedules)
        bindings={n:digest(OUTPUT/n) for n in ('launch.json','resource_monitor.jsonl','tensor_index.json','training_schedules.json')}
        verify(launch);authenticate(SWITCH_CHECK_SHA);verify_artifacts(TARGETS,target_ids);verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='OBSERVATION_HORIZON_INPUTS_COMPLETE',source_sha256=sources,
            artifact_sha256=bindings,target_result_sha256=TARGET_SHA,context_slots=912,available_contexts=828,
            counts=counts,all_twenty_four_branch_prefix_tensor_groups_exact=True,
            original_past_tensors_and_command_prefixes_exact=True,optimization_seeds=list(SEEDS),
            private_untrained_model_state_sha256=before,private_model_state_unchanged=True,compatibility=compatibility,
            optimizer_steps=0,transfer_future_materialization=False,transition_fit_inputs_validated=True,
            wall_s=time.perf_counter()-started,hardware_after=hardware(),native_execution=False,model_training=False,
            navigation_qualified=False,goal_achieved=False))
        print('OBSERVATION_HORIZON_INPUTS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_OBSERVATION_HORIZON_INPUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
