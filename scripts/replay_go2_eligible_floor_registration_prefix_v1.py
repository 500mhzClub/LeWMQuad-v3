"""Compare complete floor registration on the original public tracking prefix."""
from contextlib import closing
from itertools import islice
import json
import os
from pathlib import Path
import time

from lewm.causal_sensor_state import _identity
from lewm.eligible_floor_registration_development import EligibleFloorRegistration, MeasuredFloorTransportRegistration
from scripts import replay_go2_chained_anchor_controller_prefix_v1 as reference
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE='scripts/replay_go2_eligible_floor_registration_prefix_v1.py'
TEST='lewm/tests/test_eligible_floor_registration_development.py'
PROTOCOL='docs/go2_eligible_floor_registration_prefix_v1_2026-09-11.md'
BENCHMARK=Path('docs/go2_eligible_floor_cell_index_microbenchmark_v2_2026-09-11.json')
BENCHMARK_SHA='974d4e59ef64c6f7aaa8c783aef9c50ba2166aa9b55d11fdf9faaec4da8801d1'
OUTPUT=BASE/'go2_eligible_floor_registration_prefix_v1_attempt_001'
FRAMES=854


def main():
    env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k)!=v for k,v in env.items()):raise ValueError('fixed CPU environment required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive registration replay required')
    verify({str(BENCHMARK):BENCHMARK_SHA})
    bench=json.loads(BENCHMARK.read_text())
    inherited=reference.observer.prepared_sources()
    sources=reference.observer.merge_sources(inherited,bench['source_sha256'])
    sources=discover_sources((SOURCE,TEST,PROTOCOL,str(BENCHMARK)),sources);verify(sources)
    inputs=reference.observer.admit_worker(sources)
    hw=reference.resources()
    # The preceding full controller and timing replay must remain ended.
    owner=json.loads(Path('docs/go2_chained_anchor_controller_execution_2026-09-11.json').read_text())['owner']
    if reference.observer.owner_live(owner) or reference.observer.owner_live(reference.observer.CPU_OWNER):
        raise ValueError('preceding full CPU owners must be ended')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_artifact_sha256=inputs,
        owner_pid=os.getpid(),boot_id=reference.observer.BOOT,environment=env,hardware=hw,
        frames=FRAMES,original_class='MeasuredFloorTransportRegistration',candidate_class='EligibleFloorRegistration',
        model_loaded=False,raw_observer_reexecuted=False,native_execution=False))
    print('ELIGIBLE_REGISTRATION_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter()
    try:
        directory=reference.native.OUTPUT/reference.native.CASE[0]
        reader=reference.observer.IntentReturnRGBDReplay(directory)
        acquisitions=read_json(directory,'auxiliary_camera_audit.json')
        original=MeasuredFloorTransportRegistration();candidate=EligibleFloorRegistration()
        counts=0;totals=dict(original=0,candidate=0)
        with (OUTPUT/'comparison.jsonl').open('x') as output, closing(reference.observer.read_rows(directory)) as rows:
            for frame,row in enumerate(islice(rows,FRAMES)):
                if row['tick']!=frame or row['observation_index']!=frame:raise ValueError('ordered original observations required')
                p,d,f,now=reader.packet(frame)
                image,aux=reference.observer.packet(directory,frame,p,
                    reference.observer.public_acquisition(acquisitions[frame]),now_ns=now)
                saved_raw=row['decision']['original_visual_evidence']
                raw=dict(saved_raw,identity=_identity(tuple(saved_raw['identity'])))
                public=(p,d,f,image,aux,raw);before=reference.observer.fingerprint(public)
                order=('original','candidate') if frame%2==0 else ('candidate','original')
                receipts={};times={}
                for name in order:
                    t=time.perf_counter_ns()
                    receipts[name]=(original if name=='original' else candidate).observe(p,d,aux,raw,now_ns=now)
                    times[name]=(time.perf_counter_ns()-t)/1e6;totals[name]+=times[name]
                expected=reference.observer.canonical(row['decision']['evidence'])
                if any(reference.observer.canonical(v)!=expected for v in receipts.values()):
                    raise ValueError('entire original and candidate registration must match recorded evidence')
                if reference.observer.canonical(vars(original))!=reference.observer.canonical(vars(candidate)):
                    raise ValueError('complete registration reference, anchor and state must match')
                if before!=reference.observer.fingerprint(public):raise ValueError('registration mutated public inputs')
                output.write(json.dumps(dict(frame=frame,public_input_sha256=before,order=list(order),wall_ms=times,
                    recorded_evidence_exact=True,complete_registration_state_exact=True,public_inputs_unchanged=True))+'\n')
                output.flush();counts+=1
                if frame%100==0:print('ELIGIBLE_REGISTRATION_FRAME',frame,flush=True)
        if counts!=FRAMES:raise ValueError('complete fixed raw prefix required')
        if reference.observer.admit_worker(sources)!=inputs:raise ValueError('original raw worker inputs changed')
        verify(sources)
        ids={n:digest(OUTPUT/n) for n in ('launch.json','comparison.jsonl')};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='ELIGIBLE_FLOOR_REGISTRATION_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,frames=counts,total_registration_wall_ms=totals,
            total_time_reduction_percent=100*(totals['original']-totals['candidate'])/totals['original'],
            all_recorded_registration_receipts_exact=True,all_registration_states_exact=True,
            public_inputs_unchanged=True,model_loaded=False,raw_observer_reexecuted=False,
            full_controller_reexecuted=False,shared_host_timings=True,following_observation_consumed=False,
            native_execution=False,real_time_qualified=False,navigation_qualified=False,goal_achieved=False,
            wall_s=time.perf_counter()-started))
        print('ELIGIBLE_REGISTRATION_COMPLETE',digest(OUTPUT/'result.json'),counts,totals,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ELIGIBLE_REGISTRATION_REPLAY_FAILURE',
            reason=repr(error),automatic_retry=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
