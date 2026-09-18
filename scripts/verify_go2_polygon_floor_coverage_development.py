"""Compare old/new current-frame maps on the diagnosed retained camera views."""
import json
import hashlib
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from lewm.projected_polygon_floor_coverage_development import ProjectedPolygonFloorRoutingMap,warmup
from lewm.two_cm_floor_extent_development import configure
from scripts.diagnose_alignment_route_switches_development import saved_pose
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.replay_go2_no_early_release_map_entry_development import RecordedCurrentPlaneMap
from scripts import run_go2_frozen_readout_navigation_development as run


class RecordedPolygonMap(ProjectedPolygonFloorRoutingMap):
    _read_pose=staticmethod(saved_pose)


def main():
    output=run.BASE/'go2_polygon_floor_coverage_saved_views_v2.json'
    if output.exists():raise ValueError('preserve completed diagnostic')
    probe=json.loads((run.BASE/'go2_frozen_readout_floor_view_probe_v1.json').read_text())
    cases=defaultdict(lambda:defaultdict(list))
    for row in probe['rows']:cases[row['assignment']][row['frame']].append(row)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1);configure();warmup()
    rows=[];began=time.monotonic()
    for assignment,frames in cases.items():
        root=run.BASE/run.root_name(assignment)
        poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
        # Public replay checks depth retirement and both delivered noise digests.
        reader=NoisyPublicReplay(root/'native')
        old=RecordedCurrentPlaneMap();new=RecordedPolygonMap()
        for frame in sorted({0,*frames}):
            policy,depth,_,_,auxiliary,now=reader.packet(frame)
            start=time.perf_counter();a=old.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
            old_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();b=new.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
            new_ms=(time.perf_counter()-start)*1000
            assert a.map_from_initial==b.map_from_initial and a.floor_height==b.floor_height
            assert a.current_floor<=b.current_floor and a.floor<=b.floor
            assert a.occupied==b.occupied and a.fine_occupied==b.fine_occupied
            assert a.current_occupied==b.current_occupied and a.current_fine_occupied==b.current_fine_occupied
            queries=[]
            for case in frames.get(frame,[]):
                cell=tuple(case['cell']);views=case['cameras'].values()
                old_expected=any(v['covered'] for v in views)
                new_expected=any(v['covered'] or (v['classifier']=='raw_valid_quads'
                    and v.get('failure_only_outside_projected_square',False)) for v in views)
                assert (cell in a.current_floor)==old_expected
                assert (cell in b.current_floor)==new_expected
                queries.append(dict(cell=cell,old_covered=old_expected,new_covered=new_expected))
            rows.append(dict(assignment=assignment,frame=frame,old_ms=old_ms,new_ms=new_ms,
                old_current_floor_cells=len(a.current_floor),new_current_floor_cells=len(b.current_floor),queries=queries))
            print('POLYGON_VIEW_VERIFIED',assignment,frame,'ms',round(old_ms,2),round(new_ms,2),queries,flush=True)
    run.previous.save(output,dict(schema='polygon_floor_coverage_saved_views.v1',rows=rows,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
            'lewm/projected_polygon_floor_coverage_development.py',
            'scripts/verify_go2_polygon_floor_coverage_development.py')},
        old_floor_cells_preserved=True,all_obstacle_cells_identical=True,
        basis_and_floor_height_identical=True,queries_match_independent_pixel_diagnosis=True,
        delivered_noisy_depth_digests_verified=True,native_state_or_wall_geometry_used=False,
        historical_accumulated_map_replayed=False,current_frame_comparison_only=True,
        native_navigation_executed=False,wall_seconds=time.monotonic()-began))


if __name__=='__main__':main()
