"""Whole-population actual policy tensor validation; no optimizer or checkpoint."""
import argparse
from collections import Counter
import json
import resource
import shutil
import time
import cv2
import torch
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.geometry_progress_family_policy_stream_development import FamilyPolicyStream
from scripts.read_go2_geometry_progress_family_causal_v1 import OUTPUT as DERIVATION,INPUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify

OUTPUT=BASE/'go2_geometry_progress_family_policy_stream_v1_attempt_001'
PROTOCOL='docs/go2_geometry_progress_family_policy_stream_v1_2026-09-08.md'
SCHEDULE_SEED=2026090950


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--causal-result-sha256',required=True);args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive full policy-stream check; no retry/resume')
    causal_ids={'result.json':args.causal_result_sha256};verify_artifacts(DERIVATION,causal_ids)
    derived=read_json(DERIVATION,'result.json')
    if derived['status']!='GEOMETRY_PROGRESS_FAMILY_CAUSAL_READOUT_COMPLETE' or not derived['ready_for_separately_frozen_learning']:
        raise ValueError('complete measurement-admissible causal derivation required')
    causal_ids|=derived['artifact_sha256'];verify_artifacts(DERIVATION,causal_ids)
    ids={'result.json':derived['collection_sha256']};verify_artifacts(INPUT,ids)
    collection=read_json(INPUT,'result.json');ids|=collection['artifact_sha256'];verify_artifacts(INPUT,ids)
    launch=read_json(INPUT,'launch.json');verify(launch)
    sources=discover_sources((PROTOCOL,'scripts/check_go2_geometry_progress_family_policy_stream_v1.py',
        'lewm/tests/test_geometry_progress_family_learning_view_development.py',
        'lewm/tests/test_geometry_progress_family_policy_stream_development.py'),derived['source_sha256'])
    definition=launch|dict(source_sha256=sources);verify(definition)
    if shutil.disk_usage(BASE.parent).free<40*1024**3+64*1024**2:raise ValueError('check budget plus40GiB reserve required')
    view=FamilyWindowView(read_json(DERIVATION,'windows.json'))
    stream=FamilyPolicyStream(view,output=INPUT,bindings=collection['artifact_sha256'],tensor_index=read_json(DERIVATION,'tensor_index.json'))
    schedule=view.schedule(updates=1200,batch_size=6,seed=SCHEDULE_SEED)
    draws=Counter(view.windows[i]['trial'] for b in schedule['batches'] for i in b)
    if len(draws)!=48 or set(draws.values())!={150}:raise ValueError('equal training-episode draw counts required')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026090951);model=CumulativePulseRGBBodyJEPA(32).eval()
    before=state_digest(model.state_dict());create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_sha256=ids,causal_sha256=causal_ids,
        protocol=PROTOCOL,model_initial_sha256=before,optimizer_steps=0))
    write_json(OUTPUT/'training_schedule.json',schedule);stats={};start=time.perf_counter()
    try:
        for role in ('train','geometry_transfer'):
            indices=view.indices(role);count=0;active_slots=0
            for at in range(0,len(indices),6):
                batch_ids=indices[at:at+6]
                if role=='train':batch=stream.training_batch(batch_ids);inputs=batch['inputs']
                else:inputs=stream.inference_batch(batch_ids,role=role)
                active,offsets=validate_timed_plan(inputs['known_action_blocks'],inputs['known_action_valid'],len(batch_ids))
                with torch.inference_mode():out=model(**inputs)
                if not torch.equal(out['prediction_valid'],active) or not torch.equal(out['target_offsets_ns'],offsets):
                    raise ValueError('untrained interface returned wrong actual horizons')
                if not all(torch.isfinite(out[k][active]).all() for k in ('direct_outcomes','rollout_outcomes')):
                    raise ValueError('nonfinite model interface prediction')
                count+=len(batch_ids);active_slots+=int(active.sum())
            stats[role]=dict(materialized_windows=count,active_prediction_slots=active_slots,
                future_images_read_for_inputs=False,future_images_read_as_training_targets=role=='train')
            print('FAMILY_POLICY_STREAM',role,stats[role],flush=True)
        if state_digest(model.state_dict())!=before or any(p.grad is not None for p in model.parameters()):
            raise ValueError('interface validation cannot modify model weights or gradients')
        verify(definition);verify_artifacts(INPUT,ids);verify_artifacts(DERIVATION,causal_ids)
        result=dict(status='GEOMETRY_PROGRESS_FAMILY_POLICY_STREAM_COMPLETE',causal_result_sha256=args.causal_result_sha256,
            collection_result_sha256=derived['collection_sha256'],roles=stats,
            training_schedule_sha256=schedule['schedule_sha256'],training_episode_draw_counts=dict(draws),
            untrained_model_sha256=before,optimizer_steps=0,wall_s=time.perf_counter()-start,
            maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            model_trained=False,learned_benefit_established=False,navigation_qualified=False,goal_achieved=False,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','training_schedule.json')})
        write_json(OUTPUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FAMILY_POLICY_STREAM_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
