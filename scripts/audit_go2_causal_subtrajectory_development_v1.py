#!/usr/bin/env python3
"""Independent scalar label/plan audit of the fixed development window derivation."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.causal_subtrajectory_learning_development import CORPUS_ROOT,DERIVATION_ROOT,remaining_plan
from lewm.counterfactual_learning_data_development import AuditedCounterfactualDataset
from lewm.counterfactual_maze_development import ACTIONS
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.build_go2_causal_subtrajectory_development_v1 import digest,write_json,checked_leaf,source_bindings


def check(value,message):
    if not value: raise ValueError(message)


def scalar_motion(origin,future):
    """Quaternion-vector identity independent of the builder's rotation matrix."""
    q=np.asarray(origin[3:],dtype=float); q=q/np.linalg.norm(q)
    vec=-q[:3]; delta=np.asarray(future[:3])-origin[:3]
    cross=2*np.cross(vec,delta); body=delta+q[3]*cross+np.cross(vec,cross)
    def yaw(pose):
        x,y,z,w=np.asarray(pose[3:])/np.linalg.norm(pose[3:])
        return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    difference=yaw(future)-yaw(origin)
    return np.array([body[0],body[1],math.atan2(math.sin(difference),math.cos(difference))])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    check(output==DERIVATION_ROOT and output.resolve()==output,'exact development audit root')
    check(not (output/'raw_artifact_audit.json').exists(),'audit already exists')
    launch=json.loads((output/'launch.json').read_text()); report=json.loads((output/'result.json').read_text())
    check(report['status']=='COMPLETE' and report['launch_sha256']==digest(output/'launch.json'),'launch/result binding')
    check(launch['source_sha256']==source_bindings(),'derivation source binding')
    check(launch['corpus_result_sha256']==report['corpus_result_sha256']==digest(CORPUS_ROOT/'result.json'),'input result binding')
    check(launch['corpus_audit_sha256']==digest(CORPUS_ROOT/'raw_artifact_audit.json'),'input audit binding')
    check(report['windows_sha256']==digest(output/'windows.json'),'window binding')
    rows=json.loads((output/'windows.json').read_text()); data=AuditedCounterfactualDataset(CORPUS_ROOT,'train')
    check(len(rows)==report['window_count'] and len({r['window_id'] for r in rows})==len(rows),'window population')
    by_scene={}
    for row in rows: by_scene.setdefault(row['scene_id'],[]).append(row)
    check(set(by_scene)==set(data.members)==set(report['branch_bindings']) and report['branch_count']==120,'branch population')
    maximum_label_error=0.; maximum_old_label_error=0.; old_labels_checked=0; packet_count=0
    motion_count=0; horizon_count=0; valid_contact=0; positive_contact=0
    for scene,member in data.members.items():
        trial=member['result']; directory=data.paths[scene]
        raw_path=checked_leaf(directory,'physics_trace.npz',trial['artifact_sha256']['physics_trace.npz'])
        with np.load(raw_path,allow_pickle=False) as archive:
            raw={k:archive[k] for k in ('timestamp_s','base_pose_world','physics_contact','phase','applied_command')}
        ns=[round(t*1e9) for t in raw['timestamp_s']]; physical={t:i for i,t in enumerate(ns)}
        start=trial['prefix_terminal_sample_index']; t0=ns[start]; stop=t0+4_000_000_000
        events=[ns[i] for i,c in enumerate(raw['physics_contact']) if c]
        first=min(events) if events else None
        expected_offsets=[offset for offset in range(0,4_000_000_000,500_000_000)
            if t0+offset in physical and (first is None or t0+offset<first)]
        actual=by_scene[scene]
        check([r['offset_ns'] for r in actual]==expected_offsets,'pre-contact window population')
        binding=report['branch_bindings'][scene]
        check(binding=={'member_result_sha256':member['result_sha256'],'physics_sha256':digest(raw_path),'windows':len(actual)},'member identity')
        canonical=member['prefix_match']['canonical_model_context_scene_id']; cache={}
        def packet(source,index):
            key=(source,index)
            if key not in cache: cache[key]=load_route_observation(data.paths[source],index)
            return cache[key]
        for row in actual:
            current=t0+row['offset_ns']; context=canonical if row['offset_ns']==0 else scene
            check(row['window_id']==f'{scene}-t{row["offset_ns"]}' and row['decision_ns']==current,'window clock/id')
            check(row['layout_id']==trial['layout_id'] and row['data_role']==trial['data_role']
                and row['action_index']==trial['action_index'] and row['context_scene_id']==context,'window source/role')
            check(data.members[context]['result']['data_role']==trial['data_role']
                and data.members[context]['result']['layout_id']==trial['layout_id'],'canonical layout/role')
            check(len(row['history_observation_indices'])==4,'history count')
            for index,lag in zip(row['history_observation_indices'],(-300_000_000,-200_000_000,-100_000_000,0),strict=True):
                p=packet(context,index)
                check(p['image']['measured_ns']==p['sensor_state']['decision_ns']==current+lag,'actual causal packet time')
            prior=p['sensor_state']['control']['applied_command']
            check(prior['valid'][-1].all(),'valid current past command')
            remaining=(stop-current)//100_000_000
            check(row['remaining_ticks']==remaining,'remaining duration')
            planned=remaining_plan(ACTIONS[trial['action_index']][1],prior['values'][-1],remaining)
            commands=planned['known_action_blocks'].numpy().reshape(40,3)*[.3,1.,.5]
            mask=planned['known_action_valid'].numpy().reshape(40)
            check(np.array_equal(mask,np.arange(40)<remaining) and np.all(commands[remaining:]==0),'unknown plan tail')
            # Every actually executed part agrees with a command inferred solely from the past.
            origin=physical[current]
            for tick in range(remaining):
                begin=origin+1+50*tick; end=min(begin+50,len(ns))
                if begin>=len(ns): break
                check(np.all(raw['phase'][begin:end]==1),'window outside original fixed-action phase')
                check(np.allclose(raw['applied_command'][begin:end],commands[tick],rtol=0,atol=1e-7),'prospective/actually applied command')
            check(len(row['targets'])==8,'horizon population')
            for i,target in enumerate(row['targets']):
                horizon=(i+1)*500_000_000; time=current+horizon; in_plan=time<=stop
                event=in_plan and first is not None and first<=time
                mv=in_plan and time in physical and (first is None or time<first)
                cv=in_plan and (time<=ns[-1] or event)
                check(target['horizon_ns']==horizon and target['in_plan']==in_plan,'horizon definition')
                check(target['motion_valid']==mv and target['contact_valid']==cv
                    and target['contact_by_horizon']==(bool(event) if cv else None),'censoring/contact semantics')
                horizon_count+=1; valid_contact+=cv; positive_contact+=bool(event)
                if mv:
                    expected=scalar_motion(raw['base_pose_world'][origin],raw['base_pose_world'][physical[time]])
                    error=float(np.max(np.abs(expected-target['delta_xy_yaw_current_body'])))
                    maximum_label_error=max(maximum_label_error,error); check(error<1e-10,'independent scalar motion')
                    future=packet(scene,target['future_observation_index'])
                    check(future['image']['measured_ns']==future['sensor_state']['decision_ns']==time,'actual future packet')
                    motion_count+=1
                    if row['offset_ns']==0 and trial['horizon_labels'][i]['motion_valid']:
                        old=trial['horizon_labels'][i]['delta_xy_yaw_start_body']
                        error=float(np.max(np.abs(expected-old))); maximum_old_label_error=max(maximum_old_label_error,error)
                        check(error<1e-10,'initial old-label consistency'); old_labels_checked+=1
                else: check(target['delta_xy_yaw_current_body'] is None and target['future_observation_index'] is None,'censored target content')
        packet_count+=len(cache)
        print(json.dumps({'event':'branch_audited','scene_id':scene,'windows':len(actual)}),flush=True)
    for role in ('train','validation'):
        selected=[r for r in rows if r['data_role']==role]
        expected={'layouts':len({r['layout_id'] for r in selected}),'windows':len(selected),
            'by_offset_ns':dict(Counter(str(r['offset_ns']) for r in selected)),
            'in_plan_horizons':sum(t['in_plan'] for r in selected for t in r['targets']),
            'motion_valid':sum(t['motion_valid'] for r in selected for t in r['targets']),
            'contact_valid':sum(t['contact_valid'] for r in selected for t in r['targets']),
            'contact_positive':sum(t['contact_by_horizon'] is True for r in selected for t in r['targets'])}
        check(expected==report['counts'][role],'aggregate counts')
    audit={'status':'PASS','audited_windows':len(rows),'audited_horizons':horizon_count,'valid_motion_targets':motion_count,
        'valid_contact_targets':valid_contact,'positive_contact_targets':positive_contact,'packet_checks':packet_count,
        'maximum_independent_motion_error':maximum_label_error,'old_initial_labels_checked':old_labels_checked,
        'maximum_old_initial_label_error':maximum_old_label_error,'study_result_sha256':digest(output/'result.json'),
        'audit_source_sha256':digest(Path(__file__)),'scope':'development temporal data semantics; no fitting or navigation claim'}
    write_json(output/'raw_artifact_audit.json',audit); print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
