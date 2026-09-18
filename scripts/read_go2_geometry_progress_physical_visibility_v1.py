"""Separate all-frame first-opaque-surface diagnostic; no artifact correction."""
import json
import numpy as np
from lewm.physical_first_surface_depth_development import evaluate_visibility
from lewm.geometry_progress_pilot_development import TRIALS, specification
from scripts.run_go2_geometry_progress_pilot_v1 import OUTPUT as INPUT
from scripts.read_go2_geometry_progress_science_v1 import OUTPUT as SCIENCE
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT=BASE/'go2_geometry_progress_physical_visibility_v1_attempt_001'
SCIENCE_IDS={'result.json':'e3255efaac6d4d759dfe6c9b5c5f44f3e3ca58c83d4184ecb132a0c94d22bc8b'}


def main():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists():raise ValueError('exclusive read-only physical visibility diagnostic')
    verify_artifacts(SCIENCE,SCIENCE_IDS);science=read_json(SCIENCE,'result.json')
    verify_artifacts(INPUT,science['collection_sha256']);collection=read_json(INPUT,'result.json')
    launch=read_json(INPUT,'launch.json');verify(launch)
    names=[]
    for c in TRIALS:
        names.extend(c+'/'+n for n in ('specification.json','camera_audit.json','depth_camera_audit.json'))
        names.extend(c+f'/native_depth_{i:04d}.npz' for i in range(collection['conditions'][c]['rgbd_frames']))
    bindings=science['collection_sha256']|{n:collection['artifact_sha256'][n] for n in names}
    verify_artifacts(INPUT,bindings)
    sources=discover_sources(('scripts/read_go2_geometry_progress_physical_visibility_v1.py',),science['source_sha256'])
    definition=launch|dict(source_sha256=sources);verify(definition);create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(input_sha256=bindings,science_sha256=SCIENCE_IDS,source_sha256=sources,
        scope='all-frame existing first-opaque-surface evaluator; no native execution, pixel repair or training eligibility',
        geometry_available_evidence_remains_failed=True,render_near_m=.05,stride=8))
    reports={}
    try:
        for c in TRIALS:
            spec=read_json(INPUT/c,'specification.json')
            if spec!=specification(c):raise ValueError('exact frozen scene required')
            cameras=read_json(INPUT/c,'camera_audit.json');depths=read_json(INPUT/c,'depth_camera_audit.json');rows=[]
            if len(cameras)!=len(depths):raise ValueError('complete camera/depth pairs required')
            for i,(cam,dc) in enumerate(zip(cameras,depths,strict=True)):
                if dc['native_near_m']!=.05:raise ValueError('original five-centimetre renderer required')
                with np.load(INPUT/c/f'native_depth_{i:04d}.npz',allow_pickle=False) as z:native=z['optical_depth_m']
                score=evaluate_visibility(native,spec['geometry']['wall_boxes'],cam['world_from_optical'],render_near_m=.05)
                rows.append(dict(frame=i,decision_ns=round(cam['timestamp_s']*1e9),physical_sample_index=cam['physical_sample_index'],**score))
            reports[c]=dict(rows=rows,failed_frames=[r['frame'] for r in rows if not r['passes_sampled_physical_visibility']],
                failed_input_history_frames=[r['frame'] for r in rows[:4] if not r['passes_sampled_physical_visibility']],
                failed_future_target_frames=[r['frame'] for r in rows if r['frame'] in range(8,44,5) and not r['passes_sampled_physical_visibility']])
            print('GEOMETRY_PHYSICAL_VISIBILITY',c,reports[c]['failed_frames'],flush=True)
        allrows=[r for p in reports.values() for r in p['rows']]
        verify(definition);verify_artifacts(INPUT,bindings);verify_artifacts(SCIENCE,SCIENCE_IDS)
        write_json(OUTPUT/'result.json',dict(status='GEOMETRY_PROGRESS_PHYSICAL_VISIBILITY_READOUT_COMPLETE',
            episodes=len(reports),frames=len(allrows),failed_frames=sum(not r['passes_sampled_physical_visibility'] for r in allrows),
            clipped_opaque_rays=sum(r['clipped_opaque_rays'] for r in allrows),
            false_public_valid_near_rays=sum(r['false_public_valid_near_rays'] for r in allrows),
            failed_input_history_frames=sum(len(p['failed_input_history_frames']) for p in reports.values()),
            failed_future_target_frames=sum(len(p['failed_future_target_frames']) for p in reports.values()),
            conditions=reports,source_sha256=sources,launch_sha256=digest(OUTPUT/'launch.json'),
            all_raw_artifacts_unchanged=True,training_eligibility_granted=False,model_trained=False,
            navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_VISIBILITY_DIAGNOSTIC_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
