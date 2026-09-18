"""Whole-population causal input check after complete native branch auditing."""
import argparse
from collections import Counter
import json
import time
import cv2
import torch
from lewm.moving_action_switch_family_development import TRIALS
from lewm.moving_action_switch_accounting_development import summarize
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView
from lewm.moving_action_switch_learning_sample_development import inference_inputs,materialize_training
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
from lewm.cumulative_pulse_learning_development import training_loss
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.moving_action_switch_policy_stream_development import INPUT,policy_leaves,_PolicyReader
from scripts.moving_action_switch_runtime_development import verify,hardware
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_moving_action_switch_inputs_v1_attempt_001'
PROTOCOL='docs/go2_moving_action_switch_inputs_v1_2026-09-08.md'
OPTIMIZATION_SEEDS=(2026091001,2026091401,2026091402)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--collection-result-sha256',required=True);args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive new input check; no retry/resume')
    collection_ids={'result.json':args.collection_result_sha256};verify_artifacts(INPUT,collection_ids)
    collection=read_json(INPUT,'result.json')
    if (collection['status']!='MOVING_ACTION_SWITCH_COLLECTION_AND_AUDIT_COMPLETE'
            or not collection['all_measurement_and_prefix_gates_pass'] or collection['cells']!=144):
        raise ValueError('complete passing 144-cell collection and exact prefix gates required')
    collection_ids|=collection['artifact_sha256'];verify_artifacts(INPUT,collection_ids)
    original=read_json(INPUT,'launch.json');verify(original)
    reports=[read_json(INPUT,t+'_audit.json') for t in TRIALS]
    summary=summarize(reports)
    if any(collection[k]!=v for k,v in summary.items()):raise ValueError('complete collection accounting must reproduce')
    view=MovingActionSwitchView(reports)
    schedules={str(seed):view.schedule(updates=1200,batch_size=6,seed=seed) for seed in OPTIMIZATION_SEEDS}
    sources=discover_sources((PROTOCOL,'scripts/check_go2_moving_action_switch_inputs_v1.py',
        'lewm/tests/test_moving_action_switch_learning_development.py',
        'lewm/tests/test_moving_action_switch_policy_stream_development.py'),original['source_sha256'])
    definition=original|dict(source_sha256=sources);verify(definition);resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('bounded input check RAM and storage allowance required')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026091400);model=CumulativePulseRGBBodyJEPA(32).eval()
    before=state_digest(model.state_dict())
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',definition|dict(output_root=str(OUTPUT),protocol=PROTOCOL,
        collection_sha256=collection_ids,hardware=resources,untrained_model_sha256=before,
        optimization_seeds=list(OPTIMIZATION_SEEDS),optimizer_steps=0,threads=1))
    write_json(OUTPUT/'training_schedules.json',schedules)
    print('MOVING_ACTION_SWITCH_INPUTS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    start=time.perf_counter();index=[];stats={}
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for number,report in enumerate(reports):
                role=report['data_role'];available=report['outcome']['branch_available']
                row=dict(trial=report['trial'],data_role=role,materialized=available)
                cell=stats.setdefault(role,dict(planned_cells=0,materialized_cells=0,motion_targets=0,contact_targets=0,
                    contact_positive_targets=0,future_images_read_as_targets=0,future_images_read_for_inputs=False))
                cell['planned_cells']+=1
                if available:
                    training=role=='train';names=policy_leaves(report,include_future=training)
                    paths=[report['trial']+'/'+n for n in names]
                    bindings={n:collection_ids[n] for n in paths};verify_artifacts(INPUT,bindings)
                    reader=_PolicyReader(INPUT/report['trial'],[int(n[4:8]) for n in names if n.startswith('rgb_')])
                    sample=materialize_training(reader,report) if training else inference_inputs(reader,report)
                    inputs=sample['inputs'] if training else sample
                    batch=stack_samples([sample]);batched_inputs=batch['inputs'] if training else batch
                    active,offsets=validate_timed_plan(batched_inputs['known_action_blocks'],batched_inputs['known_action_valid'],1)
                    with torch.inference_mode():
                        out=model(**batched_inputs)
                        if training:
                            loss,_=training_loss(model,batch,'jepa')
                            if not torch.isfinite(loss):raise ValueError('finite complete target contract required')
                    if not torch.equal(out['prediction_valid'],active) or not torch.equal(out['target_offsets_ns'],offsets):
                        raise ValueError('exact model clocks and active horizons required')
                    if not all(torch.isfinite(out[k][active]).all() for k in ('direct_outcomes','rollout_outcomes')):
                        raise ValueError('finite untrained output interface required')
                    row.update(history_sha256={k:fingerprint(v.numpy()) for k,v in inputs['observation_history'].items()},
                        known_action_sha256=fingerprint(inputs['known_action_blocks'].numpy()),
                        known_action_valid_sha256=fingerprint(inputs['known_action_valid'].numpy()))
                    if training:
                        t=sample['targets'];cell['motion_targets']+=int(t['motion_valid'].sum())
                        cell['contact_targets']+=int(t['contact_valid'].sum());cell['contact_positive_targets']+=int(t['contact'][t['contact_valid']].sum())
                        cell['future_images_read_as_targets']+=int(t['future_valid'].sum())
                    verify_artifacts(INPUT,bindings);cell['materialized_cells']+=1
                index.append(row)
                if number%12==0:
                    monitor.write(json.dumps(dict(cells_completed=number+1,**hardware()))+'\n');monitor.flush()
                    print('MOVING_ACTION_SWITCH_INPUT_CELL',number+1,flush=True)
        # Exact raw-prefix equality must remain equality after preprocessing.
        for comparison in collection['prefix_comparisons']:
            if comparison['all_six_exact']:
                siblings=[r for r in index if r['trial'] in comparison['trials']]
                if not all(r['materialized'] and r['history_sha256']==siblings[0]['history_sha256'] for r in siblings):
                    raise ValueError('counterfactual preprocessing changed matched past contexts')
        if state_digest(model.state_dict())!=before or any(p.grad is not None for p in model.parameters()):
            raise ValueError('input check changed model or gradients')
        write_json(OUTPUT/'tensor_index.json',index)
        verify(definition);verify_artifacts(INPUT,collection_ids)
        result=dict(status='MOVING_ACTION_SWITCH_INPUTS_COMPLETE',roles=stats,cells=144,
            complete_measurement_and_prefix_gates_pass=True,transition_fit_inputs_validated=True,
            optimization_seeds=list(OPTIMIZATION_SEEDS),training_draw_counts={seed:dict(Counter(
                reports[i]['trial'] for b in schedule['batches'] for i in b)) for seed,schedule in schedules.items()},
            optimizer_steps=0,model_trained=False,untrained_model_sha256=before,wall_s=time.perf_counter()-start,
            collection_result_sha256=args.collection_result_sha256,source_sha256=sources,
            artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','training_schedules.json','tensor_index.json','resource_monitor.jsonl')},
            native_execution=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print('MOVING_ACTION_SWITCH_INPUTS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MOVING_ACTION_SWITCH_INPUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
