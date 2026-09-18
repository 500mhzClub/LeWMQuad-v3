"""Actual CPU worker deadlines on a paced prerecorded 10Hz camera stream.

Emitted requests are shadow outputs: recorded robot motion is never attributed
to these newly computed commands. This is not a native navigation experiment.
"""
from collections import deque, Counter
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from lewm.paced_multirate_controller_development import AcquiredFrame, PacedMultirateController
from scripts.compare_full_consensus_recorded_tracker_development import without_work_counts
from scripts import probe_delayed_action_models_development as source

OUTPUT = source.BASE/'go2_paced_multirate_recorded_prefix_v1_attempt_001'
COUNT = 61


def write(name,value):
    with (OUTPUT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def verify_evidence(published,expected):
    for frame in range(len(expected)):
        for actual,old in zip(published[frame],expected[frame],strict=True):
            if without_work_counts(json.loads(json.dumps(actual)))!=without_work_counts(old):
                raise ValueError(f'recorded measured evidence differs at {frame}')
    return dict(all_raw_and_registered_evidence_equal_except_work_counts=True)


def main():
    assert not OUTPUT.exists() and psutil.virtual_memory().available>16*1024**3
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    admission=json.loads((source.BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'/'launch.json').read_text())['input_admission']['correction_admission']
    model,condition,variant=source.source.load_assigned(admission,'seed_2026091001_full_jepa')
    before=source.source.trial.previous.state_digest(model.state_dict())
    reader=source.source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
    auxiliary=json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
    frames=[];expected=[];history=deque(maxlen=4);goal=None
    with closing(source.source.packets.read_rows(source.INPUT)) as rows:
        for index in range(COUNT):
            decision=next(rows)['decision'];p,d,f,now=reader.packet(index)
            rgb,aux=source.source.packets.rgb_packet(source.INPUT,index,p,
                source.source.public_acquisition(auxiliary[index]),now_ns=now)
            history.append(p)
            frames.append(AcquiredFrame(index,now,p,d,f,rgb,aux,tuple(history)))
            expected.append((decision['original_visual_evidence'],decision['evidence']))
            if goal is None:goal=decision['mission_receipt']['active_goal_initial_body_xy_m']
    OUTPUT.mkdir();owner=psutil.Process()
    write('launch.json',dict(owner=dict(pid=owner.pid,created=owner.create_time()),observations=COUNT,
        input=str(source.INPUT),model_state_sha256=before,goal_initial_xy=goal,
        camera_period_ns=100_000_000,request_period_ns=20_000_000,
        mapping_and_planning_period_ns=400_000_000,preloaded_recorded_acquisition=True,
        physical_execution=False,shadow_requests_only=True,
        source_sha256={p:source.source.digest(Path(p)) for p in (
            'lewm/paced_multirate_controller_development.py','lewm/multirate_routing_map_development.py',
            'lewm/delayed_action_planning_development.py','lewm/fresh_obstacle_dispatch_development.py',
            'scripts/run_paced_multirate_recorded_prefix_development.py')}))
    published={};requests=[];deliveries=[];controller=None
    try:
        epoch=frames[0].measured_ns;start=time.perf_counter_ns()
        def clock():return epoch+time.perf_counter_ns()-start
        def sink(frame,raw,registered):published[frame]=(raw,registered)
        controller=PacedMultirateController(model,goal_initial_xy=goal,condition=condition,
            variant=variant,clock_ns=clock,evidence_sink=sink)
        index=0;next_request=epoch;end=frames[-1].measured_ns+500_000_000
        while clock()<end:
            now=clock()
            while index<len(frames) and frames[index].measured_ns<=now:
                controller.submit(frames[index])
                deliveries.append(dict(frame=index,measured_ns=frames[index].measured_ns,delivered_ns=clock()))
                index+=1
            if now>=next_request:
                started=clock();request=controller.request(now_ns=started)
                missed=(started-next_request)//20_000_000
                requests.append(dict(**request,request_finished_ns=clock(),
                    scheduled_request_ns=next_request,skipped_request_ticks=int(missed)))
                next_request+=(missed+1)*20_000_000
            if controller.faults:raise RuntimeError(str(controller.faults))
            time.sleep(.001)
        controller.finish();finished=clock()
        assert len(published)==COUNT
        evidence_comparison=verify_evidence(published,expected)
        assert source.source.trial.previous.state_digest(model.state_dict())==before
        assert all(p.grad is None for p in model.parameters())
        write('stage_events.json',controller.events);write('planning.json',controller.planning)
        write('requests.json',requests);write('deliveries.json',deliveries)
        phases={stage:[e for e in controller.events if e['stage']==stage] for stage in controller.queues}
        report=dict(status='PACED_MULTIRATE_RECORDED_PREFIX_COMPLETE',observations=COUNT,
            **evidence_comparison,model_unchanged=True,
            stage_counts={k:len(v) for k,v in phases.items()},
            stage_max_completion_age_ms={k:max((e['completed_ns']-e['measured_ns'])/1e6 for e in v) for k,v in phases.items() if v},
            plans_on_time=sum(p.get('on_time') is True for p in controller.planning),
            plans_late=sum(p.get('on_time') is False for p in controller.planning),
            planner_other_reasons=dict(Counter(p['reason'] for p in controller.planning if 'reason' in p)),
            shadow_request_reasons=dict(Counter(r['reason'] for r in requests)),
            nonzero_shadow_requests=sum(any(v!=0 for v in r['requested_command']) for r in requests),
            skipped_request_ticks=sum(r['skipped_request_ticks'] for r in requests),
            max_request_compute_ms=max((r['request_finished_ns']-r['now_ns'])/1e6 for r in requests),
            elapsed_paced_s=(finished-epoch)/1e9,shared_host=True,
            preloaded_recorded_acquisition=True,physical_execution=False,shadow_requests_only=True,
            recorded_motion_attributed_to_new_requests=False,continuous_navigation_demonstrated=False,
            whole_body_contact_or_ground_support_checked=False)
        write('result.json',report);print(json.dumps(report),flush=True)
    except BaseException as error:
        if controller is not None:
            controller.stopped.set()
            for thread in controller.threads:thread.join(timeout=2.)
        write('failure.json',dict(reason=repr(error),published_frames=sorted(published),
            stage_events=[] if controller is None else controller.events,
            planning=[] if controller is None else controller.planning,requests=requests,deliveries=deliveries))
        raise


if __name__=='__main__':main()
