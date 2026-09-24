"""Read-only all-frame physical visibility audit; preserve legacy observations."""
import shutil
import numpy as np
from lewm.physical_first_surface_depth_development import evaluate_visibility
from lewm.independent_pulse_context_development import TRIALS,specification
from scripts.probe_go2_near_field_visibility_v1 import PILOT,OUTPUT as BENCH
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts

OUTPUT=BASE/'go2_independent_pulse_context_physical_visibility_audit_v1_attempt_001'
PROTOCOL='docs/go2_pilot_physical_visibility_audit_v1_2026-09-06.md'
PILOT_IDS={'launch.json':'bb06bb68d6cc8702c224d1a5b78d5b4285b2c3dcf6636e9c029f425d0519319d',
    'result.json':'132bcf46e4c62d892765b8c61a1a38563aa2aeeacc5fa2e9d1896f27e8f567ae'}
BENCH_IDS={'launch.json':'429c73911834c7c0243d9b68aa9297848ee248afa7f19ca5965b7c35e88c90e6',
    'result.json':'6eba751f612df0ad40d56b26fb4a62923528adfbfb3dd1fa897414641371504b'}


def preflight():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive read-only audit output required')
    verify_artifacts(PILOT,PILOT_IDS);verify_artifacts(BENCH,BENCH_IDS)
    old=read_json(BENCH,'launch.json');verify(old)
    result=read_json(PILOT,'result.json')
    if result['planned_trials']!=list(TRIALS) or set(result['conditions'])!=set(TRIALS):
        raise ValueError('complete frozen twelve-case pilot roster required')
    names=[]
    for trial in TRIALS:
        n=result['conditions'][trial]['rgbd_frames']
        names.extend(trial+'/'+p for p in ('specification.json','camera_audit.json','depth_camera_audit.json'))
        names.extend(f'{trial}/{prefix}_{i:04d}.npz' for i in range(n) for prefix in ('native_depth','depth'))
    bindings=PILOT_IDS|{n:result['artifact_sha256'][n] for n in names};verify_artifacts(PILOT,bindings)
    sources=discover_sources((PROTOCOL,'scripts/audit_go2_pilot_physical_visibility_v1.py'),old['source_sha256'])
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch.update(source_sha256=sources,pilot_artifact_sha256=bindings,bench_identity=BENCH_IDS,
        expected_cases=12,expected_frames=390,physics_steps=0,minimum_free_bytes=40*1024**3,maximum_new_bytes=8*1024**2)
    verify(launch)
    if shutil.disk_usage(BASE.parent).free<launch['minimum_free_bytes']+launch['maximum_new_bytes']:
        raise ValueError('audit storage reserve required')
    return launch,result


def main():
    launch,pilot=preflight();create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);reports={}
    try:
        for trial in TRIALS:
            directory=PILOT/trial;spec=read_json(directory,'specification.json')
            if spec!=specification(trial):raise ValueError('frozen scene specification changed')
            cameras=read_json(directory,'camera_audit.json');depths=read_json(directory,'depth_camera_audit.json')
            assert len(cameras)==len(depths)==pilot['conditions'][trial]['rgbd_frames']
            rows=[]
            for i,(camera,dc) in enumerate(zip(cameras,depths,strict=True)):
                assert dc['native_near_m']==.05 and dc['native_far_m']==200.
                native=read_npz(directory,f'native_depth_{i:04d}.npz')['optical_depth_m']
                public=read_npz(directory,f'depth_{i:04d}.npz')
                mask=np.isfinite(native)&(native>=.2)&(native<=5.)
                np.testing.assert_array_equal(public['valid'],mask)
                np.testing.assert_array_equal(public['depth_m'],np.where(mask,native,np.float32(0.)))
                report=evaluate_visibility(native,spec['geometry']['wall_boxes'],camera['world_from_optical'],render_near_m=.05)
                rows.append(dict(frame=i,physical_sample_index=camera['physical_sample_index'],**report))
            failed=[r['frame'] for r in rows if not r['passes_sampled_physical_visibility']]
            reports[trial]=dict(frames=len(rows),failed_frames=failed,rows=rows,
                clipped_opaque_rays=sum(r['clipped_opaque_rays'] for r in rows),
                false_public_valid_near_rays=sum(r['false_public_valid_near_rays'] for r in rows))
            write_json(OUTPUT/(trial+'.json'),reports[trial]);print('PILOT_PHYSICAL_VISIBILITY',trial,len(rows),failed,flush=True)
        assert len(reports)==12 and sum(r['frames'] for r in reports.values())==390
        verify(launch);verify_artifacts(PILOT,launch['pilot_artifact_sha256']);verify_artifacts(BENCH,BENCH_IDS)
        bindings={c+'.json':digest(OUTPUT/(c+'.json')) for c in TRIALS}
        used=(OUTPUT/'launch.json').stat().st_size+sum((OUTPUT/n).stat().st_size for n in bindings)
        if used>launch['maximum_new_bytes']:raise ValueError('metadata budget exceeded')
        result=dict(status='COMPLETE_READ_ONLY_PILOT_PHYSICAL_VISIBILITY_AUDIT',cases=12,frames=390,
            failed_frames=sum(len(r['failed_frames']) for r in reports.values()),
            cases_with_visibility_failure=[c for c,r in reports.items() if r['failed_frames']],
            clipped_opaque_rays=sum(r['clipped_opaque_rays'] for r in reports.values()),
            false_public_valid_near_rays=sum(r['false_public_valid_near_rays'] for r in reports.values()),
            artifact_sha256=bindings,new_artifact_bytes=used,physics_steps=0,original_artifacts_unchanged=True,
            training_eligibility_granted=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print(result,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_READ_ONLY_VISIBILITY_AUDIT_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
