#!/usr/bin/env python3
"""Recompute fresh Go2 development results from explicit saved raw artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.physical_execution_development import KINDS, WIDTHS, build_case, evaluate_execution
from lewm.safety.contact_hazard_ontology_v1 import is_disallowed_contact
from scripts import run_physical_graph_edge_handoff_qualification_v1 as BASE


def check(condition, message):
    if not condition:
        raise ValueError(message)


def recompute_contact_flags(native, topology, timestamps):
    """Scalar per-contact reference, independent of the development adapter."""
    offsets = native['frame_offsets']
    check(offsets.dtype.kind in 'iu' and offsets.shape == (len(timestamps)+1,), 'contact frame offsets shape/type')
    check(offsets[0] == 0 and np.all(np.diff(offsets) >= 0), 'contact offsets not monotonic')
    check(np.array_equal(native['frame_timestamp_s'],timestamps), 'contact timestamps do not match physics')
    fields = ('geom_a','geom_b','link_a','link_b','force_a','force_b','position','valid_mask')
    for key in fields:
        expected = (int(offsets[-1]),3) if key in ('force_a','force_b','position') else (int(offsets[-1]),)
        check(native[key].shape == expected, f'native field shape mismatch: {key}')
    check(native['valid_mask'].dtype == np.bool_, 'native mask is not boolean')
    robot, support, ground = (set(topology[key]) for key in ('robot_link_ids','support_link_ids','ground_link_ids'))
    flags = np.zeros(len(timestamps),dtype=bool)
    first = None
    for frame in range(len(timestamps)):
        for index in range(int(offsets[frame]),int(offsets[frame+1])):
            if not native['valid_mask'][index]:
                continue
            a,b = int(native['link_a'][index]),int(native['link_b'][index])
            if (a in robot) == (b in robot):
                continue
            robot_id, other_id, side = (a,b,'a') if a in robot else (b,a,'b')
            force = native[f'force_{side}'][index]
            check(np.isfinite(force).all() and np.isfinite(native['position'][index]).all(), 'nonfinite native contact')
            magnitude = math.hypot(*(float(component) for component in force))
            if is_disallowed_contact(robot_link_id=robot_id,environment_link_id=other_id,
                                      foot_link_ids=support,ground_link_ids=ground,
                                      self_contact=False,force_magnitude_n=magnitude):
                flags[frame] = True
                if first is None:
                    first = {'sample_index':frame, 'robot_link_id':robot_id, 'environment_link_id':other_id,
                             'force_magnitude_n':magnitude}
    return flags, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    output = args.output_dir.absolute()
    check(not any(part=='sealed' or part=='sealed_test.json' or part.startswith('sealed_') for part in output.parts), 'protected input forbidden')
    audit_path = output/'raw_artifact_audit.json'
    check(not audit_path.exists(), 'audit already exists; do not overwrite')
    launch = json.loads((output/'launch.json').read_text())
    report = json.loads((output/'result.json').read_text())
    specs = [build_case(kind,width) for kind in KINDS for width in WIDTHS]
    check(launch['case_specs']==specs, 'case population changed')
    check(report['status']=='COMPLETE' and report['completed_cases']==report['planned_cases']==8, 'study is not complete')
    check(hashlib.sha256((output/'launch.json').read_bytes()).hexdigest()==report['launch_sha256'], 'launch binding mismatch')
    for path,digest in {**launch['source_sha256'],**launch['gait_sha256']}.items():
        candidate = ROOT/path
        check(not Path(path).is_absolute() and '..' not in Path(path).parts, 'invalid bound source path')
        check(not any(part=='sealed' or part=='sealed_test.json' or part.startswith('sealed_') for part in candidate.parts), 'protected source forbidden')
        check(hashlib.sha256(candidate.read_bytes()).hexdigest()==digest, f'source/gait binding mismatch: {path}')
    rows = []
    for spec,supplied in zip(specs,report['cases'],strict=True):
        check(supplied['scene_id']==spec['scene_id'], 'case order/identity mismatch')
        case_dir=output/spec['scene_id']
        check(json.loads((case_dir/'result.json').read_text())==supplied, 'case/report result mismatch')
        allowed={'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','initial_rgb.png','final_rgb.png'}
        check(set(supplied['artifact_sha256']) <= allowed, 'unexpected case artifact binding')
        for leaf,digest in supplied['artifact_sha256'].items():
            check(hashlib.sha256((case_dir/leaf).read_bytes()).hexdigest()==digest, f'case artifact mismatch: {leaf}')
        with np.load(case_dir/'physics_trace.npz',allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        with np.load(case_dir/'native_contacts.npz',allow_pickle=False) as archive:
            native={key:archive[key] for key in archive.files}
        topology=json.loads((case_dir/'contact_topology.json').read_text())
        times=arrays['timestamp_s']
        check(len(times)>0 and np.isfinite(times).all() and np.allclose(np.diff(times),.002,rtol=0,atol=1e-10), 'physics time spacing')
        check(np.all(np.diff(arrays['phase'].astype(int))>=0) and set(arrays['phase']).issubset({0,1,2}), 'phase ordering')
        commands=arrays['applied_command']
        check(np.isfinite(commands).all() and np.all(np.abs(commands)<=np.array([.3,0,.5])+1e-7), 'command magnitude contract')
        check(np.all(np.abs(np.diff(commands,axis=0))<=np.array([.25,0,.35])+1e-7), 'command slew contract')
        flags,first=recompute_contact_flags(native,topology,times)
        check(np.array_equal(flags,arrays['physics_contact'].astype(bool)), 'recorded contact flags disagree with raw force reference')
        if first is not None:
            check(first['sample_index']==len(times)-1 and supplied['stop_reason']=='DISALLOWED_CONTACT', 'contact stop timing mismatch')
            check(supplied['first_disallowed_contact']['sample_index']==first['sample_index'], 'first contact index mismatch')
        active=arrays['phase']!=0
        edge=spec['geometry']['selected_directed_edge']
        crossing=None
        if active.any():
            try:
                crossing=BASE.canonical_port_crossing(arrays['base_pose_world'][active],np.zeros(np.count_nonzero(active),dtype=np.uint8),
                    edge['opening_segment_world'],edge['opening_normal_world'],[],sustained_samples=100)
            except BASE.ExperimentError as exc:
                check(str(exc) in ('teacher trace never crosses the canonical directed port',
                    'teacher enters a competing physical port first',
                    'teacher did not remain beyond the port for 100 physics samples'),str(exc))
        reduced=evaluate_execution(spec,arrays,stop_reason=supplied['stop_reason'],crossing=crossing)
        check(all(supplied[key]==value for key,value in reduced.items()), 'execution endpoint mismatch')
        from PIL import Image
        for name,image in supplied['images'].items():
            check(name in ('initial','final'), 'unknown image role')
            rgb=np.asarray(Image.open(case_dir/f'{name}_rgb.png'))
            check(rgb.shape==(480,640,3) and rgb.dtype==np.uint8, 'RGB shape/encoding')
            check(hashlib.sha256(rgb.tobytes()).hexdigest()==image['rgb_sha256'], 'RGB pixel binding mismatch')
            transform=np.asarray(image['world_from_optical'])
            check(transform.shape==(4,4) and np.allclose(transform[3],[0,0,0,1],rtol=0,atol=1e-12), 'optical transform shape')
            check(np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),rtol=0,atol=1e-10) and abs(np.linalg.det(transform[:3,:3])-1)<1e-10, 'optical transform is not proper')
            check(np.any(np.isclose(times,image['timestamp_s'],rtol=0,atol=1e-12)), 'image timestamp is not a recorded physical boundary')
        rows.append({'scene_id':spec['scene_id'],'status':reduced['status'],'physics_samples':len(times),
                     'first_native_disallowed_contact':first,'failed_checks':[key for key,value in reduced['checks'].items() if not value]})
    check(sum(row['status']=='SUCCESS' for row in rows)==report['successful_cases'], 'success count mismatch')
    audit={'status':'PASS','audited_cases':8,'successful_cases':report['successful_cases'],'cases':rows,
           'study_result_sha256':hashlib.sha256((output/'result.json').read_bytes()).hexdigest(),
           'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'scope':'raw packet/trajectory recomputation, not a new physical replication or novel-maze generalization test'}
    with audit_path.open('x') as stream:
        json.dump(audit,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
