"""Posthoc coverage diagnostic only; preserve frozen sample and batch decisions."""
from pathlib import Path
import numpy as np
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.diagnose_go2_recorded_raster_failures_v1 import BATCH,BATCH_IDS
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz

OUTPUT=Path('docs/go2_recorded_raster_footprint_diagnostic_2026-09-06.json')
SOURCES=('scripts/check_go2_recorded_raster_footprint_diagnostic_v1.py',
         'lewm/raster_footprint_visibility_development.py','lewm/tests/test_raster_footprint_visibility_development.py')


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive posthoc diagnostic')
    sources={p:digest(Path(p)) for p in SOURCES}
    verify_artifacts(BATCH,BATCH_IDS);launch=read_json(BATCH,'launch.json');verify(launch)
    failure=read_json(BATCH,'failure.json')
    bindings=BATCH_IDS|{n:failure[k][n] for n,k in (
        ('episode_063_commit.json','commits'),('episode_063_raw_precheck.json','prechecks'))}
    verify_artifacts(BATCH,bindings)
    commit=read_json(BATCH,'episode_063_commit.json');directory=BATCH/commit['trial']
    names=['specification.json','camera_audit.json']+[f'native_depth_{i:04d}.npz' for i in (10,11,12)]
    bindings|={commit['trial']+'/'+n:commit['artifact_sha256'][commit['trial']+'/'+n] for n in names}
    verify_artifacts(BATCH,bindings)
    spec=read_json(directory,'specification.json');cameras=read_json(directory,'camera_audit.json')
    old=read_json(BATCH,'episode_063_raw_precheck.json')['report'];rows=[]
    for i in (10,11,12):
        native=read_npz(directory,f'native_depth_{i:04d}.npz')['optical_depth_m']
        report=evaluate_footprint(native,spec['geometry']['wall_boxes'],cameras[i]['world_from_optical'],render_near_m=.005)
        assert report['original_strict_score']==old['depth_checks'][i]['physical_visibility']
        rows.append(dict(frame=i,report=report))
    verify(launch);verify_artifacts(BATCH,bindings)
    assert sources=={p:digest(Path(p)) for p in SOURCES}
    write_json(OUTPUT,dict(status='POSTHOC_RASTER_FOOTPRINT_ACCOUNTING_ONLY',source_sha256=sources,
        input_sha256=bindings,rows=rows,original_batch_failure_preserved=True,
        training_eligibility_changed=False,qualification_granted=False,goal_achieved=False))
    print('FOOTPRINT_DIAGNOSTIC',digest(OUTPUT),rows,flush=True)


if __name__=='__main__':main()
