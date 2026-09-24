"""One fixed exposed-maze JEPA mission testing projected-polygon floor coverage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.projected_polygon_floor_coverage_development import initialize_mapping
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import run_go2_frozen_readout_navigation_development as previous

BASE=previous.BASE
ROOT='go2_polygon_floor_jepa_readout_noise_2mm_native_layout01_4800_v1_attempt_001'
PLAN=Path('docs/go2_polygon_floor_navigation_plan_2026-09-17.json')
REFERENCE=BASE/previous.root_name(2)
native=previous.native
study=previous.study


def sources():
    return previous.source_hashes()|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'lewm/projected_polygon_floor_coverage_development.py',
        'scripts/run_go2_polygon_floor_navigation_development.py')}


def prepare():
    assert not (BASE/ROOT).exists()
    frozen=json.loads(previous.PLAN.read_text())
    assert previous.source_hashes()==frozen['source_sha256']
    for number in range(1,5):
        assert (BASE/previous.root_name(number)/'frozen_readout_navigation_readout_v1.json').exists()
    verification=json.loads((BASE/'go2_polygon_floor_coverage_saved_views_v2.json').read_text())
    assert verification['all_obstacle_cells_identical'] and verification['queries_match_independent_pixel_diagnosis']
    for path,digest in verification['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    plan=frozen|dict(schema='polygon_floor_navigation_plan.v1',assignments=[[1,'jepa']],
        root_names=[ROOT],models={'jepa':frozen['models']['jepa']},source_sha256=sources(),
        planned_native_assignments=1,reference_root=str(REFERENCE),
        intervention='raw measured floor coverage tests overlapping projected polygon image quads, not the enclosing rectangle',
        actual_mapping_class='ProjectedPolygonFloorRoutingMap',model_changed=False,
        controller_changes_during_batch=False,model_changes_during_batch=False,
        unchanged=['JEPA readout','six candidates','tracking','view planning','footprint coverage filter',
            'raw obstacle map','clearance and stopping checks','2mm depth noise and ideal gyro','4800-tick budget'],
        focused_tests_passed=3,saved_classification_query_count=16,
        saved_view_result=str(BASE/'go2_polygon_floor_coverage_saved_views_v2.json'),
        no_extra_repetitions_to_obtain_success=True,
        limitations=['one execution on one exposed maze','asynchronous trajectories differ',
            'saved pixel classification does not establish navigation success','no JEPA advantage or generalization established'])
    previous.previous.save(PLAN,plan)
    print('PREPARED one polygon-floor JEPA mission',flush=True)


def evaluate():
    root=BASE/ROOT
    selected=SimpleNamespace(BASE=BASE,ROOT=ROOT,PLAN=PLAN,ASSIGNMENTS=((1,'jepa'),))
    result=bind(previous.evaluation.evaluate,study=selected,
        xy=bind(previous.evaluation.xy,validate_root=bind(previous.validate_root,BASE=BASE)))(1)
    previous.previous.save(root/'polygon_floor_navigation_readout_v1.json',dict(
        schema='polygon_floor_navigation_readout.v1',navigation=result,
        physical_backtracking=previous.physical_return_edges(root),
        pipeline_faults=json.loads((root/'pipeline_faults.json').read_text()),
        exposed_layout=True,repeatability_or_causal_JEPA_advantage_established=False))
    print('POLYGON_FLOOR_MISSION_EVALUATED',result,flush=True)


def run():
    output=BASE/ROOT;plan=json.loads(PLAN.read_text())
    if output.exists() or sources()!=plan['source_sha256']:raise ValueError('fixed sources and unused output required')
    assert plan['assignments']==[[1,'jepa']]
    assert hashlib.sha256(native.INVENTORY.read_bytes()).hexdigest()==plan['inventory_sha256']
    assert sorted(os.sched_getaffinity(0))==study.cohort.transfer.CPU_GROUPS[1]
    if shutil.disk_usage(BASE).free<4*1024**3:raise ValueError('four GiB recording headroom required')
    reference=json.loads((REFERENCE/'launch.json').read_text());runtimes=[]
    hardware=native.baseline.hardware()

    def load_model(assignment):
        assert assignment=='jepa'
        model=previous.readout.load_readout('jepa')
        assert state_digest(model.state_dict())==plan['models']['jepa']['model_sha256']
        return model,'jepa','full'

    def write(name,value):
        if name=='launch.json':
            value=reference|value|dict(experiment='polygon_floor_navigation_v1',study_arm='jepa',
                batch_assignment=1,fixed_sequential_assignments=plan['assignments'],
                planned_native_assignments=1,planned_layout_count=1,hardware=hardware,
                prospective_plan=str(PLAN),prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                extra_sources=plan['source_sha256'],actual_mapping_class=plan['actual_mapping_class'],
                reference_root_path=str(REFERENCE),controller_unchanged_from_repeatability_batch=False,
                world_model_changed=False,projected_polygon_floor_coverage=True,
                exposed_development_layout=True,new_independent_development_layout=False,
                final_evaluation=False,hardware_validated=False)
        bind(study.source.write,OUTPUT=output)(name,value)
        if name=='requests.json' and runtimes:
            for filename,record in (('visual_dispatch_events.json',runtimes[0].visual_dispatch_events),
                    ('planning_latency_stress.json',runtimes[0].clock_ns.rows),
                    ('live_planning_profile.json',runtimes[0].plan_profile_rows)):
                bind(study.source.write,OUTPUT=output)(filename,record)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(native.baseline.registration_ready).result()
        def runtime(model,**kwargs):
            kwargs['clock_ns']=native.PlanningLatencyClock(kwargs['clock_ns'],20_000_000)
            result=previous.Runtime(model,prediction_source='neural',registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)
            runtimes.append(result);return result
        bind(study.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=1,
            validate_root=bind(previous.validate_root,BASE=BASE),specification=native.layouts.specification,
            public_mission=native.layouts.public_mission,MODEL_ASSIGNMENT='jepa',MODEL_LOADER=load_model,
            PacedNativeSession=partial(native.FreshCameraSession,noise_layout_index=1,noise_sigma_mm=2),
            write=write,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(native.baseline.initialize_pose,str(output)),MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,initialize_mapping=initialize_mapping)()


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true');group.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.evaluate:evaluate()
    else:run()
