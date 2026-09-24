"""Exact recorded-controller replay with bounded diagnostic instrumentation."""
import cProfile
import json
import pstats
import time

import cv2
import torch

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.observation_replan_goal_probe_development import ObservationReplanGoalProbe
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.augmented_family_switch_model_admission_development import load_assigned
from scripts.run_go2_augmented_family_switch_fits_v1 import OUTPUT as FITS
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

INPUT=BASE/'go2_observation_replan_goal_probe_v1_attempt_001'
INPUT_SHA='8cdfd800eda961de5b1a58a3b64fe8fd471c0f8c8165aae9c61f699c500010eb'
OUTPUT=BASE/'go2_observation_replan_controller_profile_v1_attempt_001'
PROTOCOL='docs/go2_observation_replan_controller_profile_v1_2026-09-08.md'
CASE='full_direct_family_episode_039'
MODEL='seed_2026091001_full_direct'


def main():
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive diagnostic attempt')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA});result=read_json(INPUT,'result.json')
    assert result['status']=='OBSERVATION_REPLAN_GOAL_PROBE_COMPLETE'
    bindings={'result.json':INPUT_SHA,**result['artifact_sha256']};verify_artifacts(INPUT,bindings)
    original=read_json(INPUT,'launch.json');verify(original)
    assert digest(URDF)==original['robot_urdf_sha256']
    sources=discover_sources((PROTOCOL,'scripts/profile_go2_observation_replan_controller_v1.py'),result['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('diagnostic resource allowance unavailable')
    admission=original['all_eighteen_admission']
    model,condition,variant=load_assigned(admission,MODEL)
    assert (condition,variant)==('direct','full')
    fit_ids={'result.json':admission['study_result_sha256'],**admission['fit_artifact_sha256']}
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=resources,
        profile_input_artifact_sha256=bindings,profile_case=CASE,profile_model=MODEL,
        native_execution=False,model_training=False,instrumented_replay=True)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('OBSERVATION_REPLAN_CONTROLLER_PROFILE_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter()
    try:
        reader=IntentReturnRGBDReplay(INPUT/CASE);rows=read_json(INPUT/CASE,'context_decisions.json')
        assert len(rows)==len(reader.frames)==40
        controller=ObservationReplanGoalProbe(model,ArticulatedCollisionGeometry(URDF),
            condition=condition,variant=variant,persistent=True)
        before=state_digest(model.state_dict());profile=cProfile.Profile();timings=[]
        for tick,row in enumerate(rows):
            assert row['tick']==row['observation_index']==tick
            p,d,f,now=reader.packet(tick);at=time.perf_counter()
            profile.enable()
            try:replayed=controller.observe(p,d,f,now_ns=now)
            finally:profile.disable()
            timings.append(dict(tick=tick,instrumented_controller_wall_s=time.perf_counter()-at))
            assert json.loads(json.dumps(replayed))==row['decision'],('exact recorded decision',tick)
        assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        functions=[]
        for (filename,line,name),(primitive,total,self_s,cumulative_s,_) in pstats.Stats(profile).stats.items():
            functions.append(dict(filename=filename,line=line,name=name,primitive_calls=primitive,
                total_calls=total,self_s=self_s,cumulative_s=cumulative_s))
        functions.sort(key=lambda r:(-r['cumulative_s'],r['filename'],r['line'],r['name']))
        write_json(OUTPUT/'profile.json',dict(functions=functions,observations=timings,
            cumulative_times_overlap=True,profiling_overhead_included=True,concurrent_training=True))
        verify(launch);verify_artifacts(INPUT,bindings);verify_artifacts(FITS,fit_ids)
        assert digest(URDF)==original['robot_urdf_sha256']
        write_json(OUTPUT/'result.json',dict(status='OBSERVATION_REPLAN_CONTROLLER_PROFILE_COMPLETE',
            source_sha256=sources,input_result_sha256=INPUT_SHA,case=CASE,model=MODEL,
            exact_replayed_decisions=len(rows),model_state_unchanged=True,
            artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','profile.json')},
            hardware_after=hardware(),wall_s=time.perf_counter()-started,
            native_execution=False,model_training=False,real_time_qualified=False,goal_achieved=False))
        print('OBSERVATION_REPLAN_CONTROLLER_PROFILE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_CONTROLLER_PROFILE_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
