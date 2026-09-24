"""Diagnose the negative registration timing on fixed original camera inputs."""
from contextlib import closing
from itertools import islice
import json
import os
from pathlib import Path
import time
from types import FunctionType
import numpy as np

from lewm import eligible_floor_cell_index_development as eligible
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from scripts import replay_go2_eligible_floor_registration_prefix_v1 as previous
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE='scripts/probe_go2_eligible_floor_index_recorded_workloads_v1.py'
OUTPUT=Path('docs/go2_eligible_floor_index_recorded_workload_probe_2026-09-11.json')
FRAMES=(0,100,400,800)
PAIRS=12
LAUNCH_SHA='225bbae0406553217cb0eddd1ccccb0174ea588babfe4244e83fd7dae43aee04'
RESULT_SHA='f4a6618ea9012d8243a5f5eabd07bdeee265d5d012273be3c52b1187e8e31b52'


def eligible_count(depth,valid,up):
    T=np.asarray(BODY_FROM_OPTICAL)
    u=(np.arange(640)+.5-320)/FOCAL;v=(np.arange(480)+.5-240)/FOCAL
    optical=np.stack((depth*u[None],depth*v[:,None],depth),axis=2)
    points=optical@T[:3,:3].T+T[:3,3]
    a,b,c,e=points[:-1,:-1],points[:-1,1:],points[1:,1:],points[1:,:-1]
    good=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,1:]&valid[1:,:-1]
    for p in (a,b,c,e):good&=p@up<-.15
    return int(good.sum())


def compare(depth,valid,up):
    fallbacks=[]
    def dense(*args):
        fallbacks.append(True)
        return eligible.dense_index(*args)
    function=eligible.observed_floor_cell_index
    candidate=FunctionType(function.__code__,function.__globals__|dict(dense_index=dense),
        function.__name__,function.__defaults__)
    before=previous.reference.observer.fingerprint((depth,valid,up));rows=[]
    for pair in range(PAIRS):
        order=('original','candidate') if pair%2==0 else ('candidate','original')
        values={};times={};n=len(fallbacks)
        for name in order:
            start=time.perf_counter_ns()
            values[name]=(eligible.dense_index if name=='original' else candidate)(depth,valid,up)
            times[name]=(time.perf_counter_ns()-start)/1e6
        for key in values['original']:
            if (values['original'][key].dtype!=values['candidate'][key].dtype
                    or values['original'][key].shape!=values['candidate'][key].shape
                    or values['original'][key].tobytes()!=values['candidate'][key].tobytes()):
                raise ValueError('original floor index must match exactly')
        rows.append(dict(pair=pair,order=list(order),wall_ms=times,dense_fallback=len(fallbacks)!=n))
    if before!=previous.reference.observer.fingerprint((depth,valid,up)):raise ValueError('inputs mutated')
    totals={name:sum(r['wall_ms'][name] for r in rows) for name in ('original','candidate')}
    return dict(input_sha256=before,eligible_cells=eligible_count(depth,valid,up),
        approved_ground_cells=int(values['original']['ground_cells'].sum()),total_cells=479*639,
        dense_fallback_calls=len(fallbacks),pairs=rows,total_wall_ms=totals,
        reduction_percent=100*(totals['original']-totals['candidate'])/totals['original'],
        all_output_arrays_byte_exact=True,inputs_unchanged=True)


def main():
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        raise ValueError('fixed CPU thread settings required')
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive recorded workload probe required')
    verify_artifacts(previous.OUTPUT,{'launch.json':LAUNCH_SHA,'result.json':RESULT_SHA})
    launch=read_json(previous.OUTPUT,'launch.json')
    sources=discover_sources((SOURCE,),launch['source_sha256']);verify(sources)
    inputs=previous.reference.observer.admit_worker(sources)
    if inputs!=launch['input_artifact_sha256']:raise ValueError('same original raw worker inputs required')
    directory=previous.reference.native.OUTPUT/previous.reference.native.CASE[0]
    reader=previous.reference.observer.IntentReturnRGBDReplay(directory)
    acquisitions=read_json(directory,'auxiliary_camera_audit.json');reports=[]
    T=np.asarray(BODY_FROM_OPTICAL)
    with closing(previous.reference.observer.read_rows(directory)) as rows:
        for frame,row in enumerate(islice(rows,max(FRAMES)+1)):
            if frame not in FRAMES:continue
            if row['tick']!=frame:raise ValueError('original ordered frame required')
            p,d,f,now=reader.packet(frame)
            image,aux=previous.reference.observer.packet(directory,frame,p,
                previous.reference.observer.public_acquisition(acquisitions[frame]),now_ns=now)
            evidence=row['decision']['evidence']
            anchor=evidence if 'floor_registration' in evidence else evidence['floor_transport']['anchor']
            up=np.asarray(anchor['floor_registration']['reference']['initial_up_body'])
            R=np.asarray(row['decision']['original_visual_evidence']['current_pose']['rotation_initial_body_from_current_body'])
            for name,packet,E in (('primary',d,T),('auxiliary',aux,body_from_optical())):
                Q=E[:3,:3]@T[:3,:3].T
                report=compare(packet['depth_m'],packet['valid'],Q.T@(R.T@up))
                reports.append(dict(frame=frame,camera=name,**report))
                print('RECORDED_INDEX_WORKLOAD',frame,name,report['eligible_cells'],
                    report['dense_fallback_calls'],report['reduction_percent'],flush=True)
    if len(reports)!=2*len(FRAMES):raise ValueError('all fixed camera populations required')
    verify(sources);verify_artifacts(previous.reference.native.OUTPUT,inputs)
    write_json(OUTPUT,dict(status='ELIGIBLE_FLOOR_INDEX_RECORDED_WORKLOADS_CHECKED',source_sha256=sources,
        registration_result_sha256=RESULT_SHA,registration_launch_sha256=LAUNCH_SHA,
        frames=list(FRAMES),pairs_per_camera=PAIRS,reports=reports,
        actual_recorded_depth_and_registration_up_used=True,shared_host_timings=True,
        private_fallback_counter_only=True,imported_module_globals_mutated=False,
        full_controller_replayed=False,native_execution=False,candidate_adopted=False,goal_achieved=False))
    print('RECORDED_INDEX_PROBE_COMPLETE',digest(OUTPUT),flush=True)


if __name__=='__main__':main()
