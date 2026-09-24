"""Separate tracker-budget effects on the exact failed fine-obstacle trajectory."""
from pathlib import Path
import json
import time
import argparse
import cv2
import torch
from lewm.feature_budget_100_tracker_development import FeatureBudget100VisualMotion
from lewm.feature_budget_150_tracker_development import FeatureBudget150VisualMotion
from lewm.feature_budget_300_tracker_development import FeatureBudget300VisualMotion
from lewm.optional_plane_refinement_development import refine_if_supported
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneFloorRegistration
from scripts.in_memory_public_replay_development import PublicReplay
from lewm.partial_floor_height_development import PartialHeightRegistration


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',default='go2_fine_obstacle_round_trip_native_layout00_v1_attempt_001')
    parser.add_argument('--partial-height',action='store_true')
    parser.add_argument('--optional-refinement',action='store_true')
    parser.add_argument('--budgets',default='100,150')
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary artifact basename required')
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')/args.root_name
    output=root/'paired_tracker_registration_diagnostic.json'
    if output.exists():raise ValueError('preserve diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native')
    count=json.loads((root/'failure.json').read_text())['acquired_frames']
    budgets=[int(x) for x in args.budgets.split(',')]
    types={100:FeatureBudget100VisualMotion,150:FeatureBudget150VisualMotion,300:FeatureBudget300VisualMotion}
    if len(set(budgets))!=len(budgets) or not set(budgets)<=set(types):raise ValueError('distinct supported budgets required')
    trackers={k:types[k]() for k in budgets}
    if args.optional_refinement:
        for tracker in trackers.values():tracker.model._refine_candidate=refine_if_supported
    registration_type=PartialHeightRegistration if args.partial_height else SampledPlaneFloorRegistration
    registrations={k:registration_type() for k in trackers}
    failures={};rows=[]
    for frame in range(count):
        p,d,f,rgb,a,now=reader.packet(frame);row=dict(frame=frame,arms={})
        for budget in (budgets if frame%2==0 else budgets[::-1]):
            if budget in failures:continue
            raw=None
            try:
                began=time.perf_counter()
                raw=trackers[budget].observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=a)
                tracking_s=time.perf_counter()-began
                if raw.get('current_pose') is None:raise ValueError('visual pose unavailable')
                began=time.perf_counter();registered=registrations[budget].observe(p,d,a,raw,now_ns=now)
                row['arms'][str(budget)]=dict(tracking_s=tracking_s,registration_s=time.perf_counter()-began,
                    raw_pose=raw['current_pose'],registered_pose=registered['current_pose'])
            except Exception as error:
                failures[budget]=dict(frame=frame,reason=repr(error))
                with (root/f'tracker_{budget}_failure_evidence.json').open('x') as out:
                    json.dump(dict(frame=frame,raw=raw,registration_anchor=registrations[budget].anchor),out)
        rows.append(row)
        if frame%40==0:print('PAIRED_REGISTRATION_FRAME',frame,flush=True)
    report=dict(public_sensor_replay_only=True,native_state_read=False,frames=count,failures=failures,
        accepted={k:sum(str(k) in row['arms'] for row in rows) for k in trackers},rows=rows)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('PAIRED_REGISTRATION_COMPLETE',report['accepted'],failures,flush=True)


if __name__=='__main__':main()
