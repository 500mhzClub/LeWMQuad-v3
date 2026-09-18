"""Paced 300-feature runtime; assess pose error instead of claiming exact equality."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from lewm.eligible_floor_registration_development import bind
from lewm.process_mapped_runtime_development import (
    initialize_mapping,mapping_ready,initialize_pose_300,pose_ready,ProcessSeparatedRuntime)
from lewm.physical_execution_development import rotation_xyzw
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts import run_paced_multirate_recorded_prefix_development as source

OUTPUT=source.source.BASE/'go2_feature_budget300_paced_recorded_prefix_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def verify_evidence(published,expected):
    # The driver calls this only after the estimation workers stop.
    with np.load(source.source.INPUT/'physics_trace.npz',allow_pickle=False) as trace:
        poses=trace['base_pose_world']
    origin=poses[749];R0=rotation_xyzw(origin[3:]);rows=[]
    for frame in range(len(expected)):
        native=poses[749+50*frame];p=(native[:3]-origin[:3])@R0
        R=R0.T@rotation_xyzw(native[3:]);row=dict(frame=frame)
        for label,evidence in (('candidate_raw',published[frame][0]),('candidate_registered',published[frame][1]),
                ('original_raw',expected[frame][0]),('original_registered',expected[frame][1])):
            pose=evidence['current_pose'];assert pose['frame']==frame
            row[label+'_position_error_m']=float(np.linalg.norm(np.asarray(pose['position_initial_body_m'])-p))
            row[label+'_rotation_error_rad']=angle(np.asarray(pose['rotation_initial_body_from_current_body']).T@R)
        assert published[frame][0]['corner_budget_per_camera']==300
        rows.append(row)
    _write('pose_accuracy.json',dict(rows=rows,native_state_loaded_after_estimation=True))
    return dict(all_raw_and_registered_evidence_equal_except_work_counts=False,
        changed_feature_population=True,corner_budget_per_camera=300,
        pose_accuracy={k:dict(median=float(np.median([r[k] for r in rows])),maximum=max(r[k] for r in rows))
            for k in rows[0] if k!='frame'},native_state_loaded_only_after_estimation=True)


def write(name,value):
    if name=='launch.json':
        value=value|dict(mapping_and_pose_separate_processes=True,corner_budget_per_camera=300,
            processes_ready_before_stream=True,
            extra_sources={p:source.source.source.digest(Path(p)) for p in (
                'lewm/process_mapped_runtime_development.py','lewm/feature_budget_300_tracker_development.py',
                'scripts/run_feature_budget300_paced_prefix_development.py')})
    if name=='result.json':value=value|dict(status='FEATURE_BUDGET300_PACED_PREFIX_COMPLETE',
        mapping_and_pose_separate_processes=True,packet_transfer_included_in_stage_timing=True)
    _write(name,value)


def main():
    with (ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initialize_mapping) as mapping,
            ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initialize_pose_300) as pose):
        assert mapping.submit(mapping_ready).result() and pose.submit(pose_ready).result()
        def controller(*args,**kwargs):
            return ProcessSeparatedRuntime(*args,mapping_executor=mapping,pose_executor=pose,**kwargs)
        bind(source.main,OUTPUT=OUTPUT,write=write,PacedMultirateController=controller,
            verify_evidence=verify_evidence)()


if __name__=='__main__':main()
