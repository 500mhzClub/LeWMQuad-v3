"""Paired association benchmark on the six primary chains at slow frame 990."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil

from lewm.eligible_floor_registration_development import bind
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.chained_corner_flow_association_development import chained_points as original
from lewm.batched_patch_agreement_development import tracked_points
from scripts import profile_stop_conditioned_early_decisions_development as source

candidate = bind(original, tracked_points=tracked_points)
OUTPUT = Path('docs/go2_batched_patch_recorded_chains_2026-09-13.json')
REFERENCES = (978, 975, 973, 971, 968, 966)


def main():
    assert not OUTPUT.exists()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT); images = {}
    for frame in range(min(REFERENCES), 991):
        policy, depth, _, now = reader.packet(frame)
        images[frame] = (frame, now, CornerSupportFeatureFrame(policy['image']['rgb'], depth))
    rows = []
    for repeat in range(5):
        for index, ref in enumerate(REFERENCES):
            frames = [images[f] for f in range(ref, 991)]; outputs = {}; times = {}
            for label, function in ((('original', original), ('candidate', candidate))
                    if (repeat+index)%2 == 0 else (('candidate', candidate), ('original', original))):
                started = time.perf_counter(); outputs[label] = function(frames)
                times[label] = time.perf_counter()-started
            for a, b in zip(outputs['original'][0], outputs['candidate'][0], strict=True):
                np.testing.assert_array_equal(a, b)
            assert outputs['original'][1] == outputs['candidate'][1]
            rows.append(dict(repeat=repeat, reference_frame=ref, current_frame=990,
                intervals=990-ref, endpoint_pairs=len(outputs['candidate'][0][0]),
                all_arrays_and_receipts_equal=True, original_s=times['original'], candidate_s=times['candidate']))
    old = sum(r['original_s'] for r in rows); new = sum(r['candidate_s'] for r in rows)
    report = dict(status='BATCHED_PATCH_RECORDED_CHAINS_COMPLETE', rows=rows,
        original_total_s=old, candidate_total_s=new, reduction_percent=100*(1-new/old),
        original_mean_six_chain_ms=old/5*1000, candidate_mean_six_chain_ms=new/5*1000,
        input=str(source.INPUT), references=list(REFERENCES), independent_image_chains=6,
        paired_repeats=5, references_selected_from_completed_profile=True,
        hardware=dict(available_ram_bytes=psutil.virtual_memory().available, shared_host=True),
        sources={p:source.digest(Path(p)) for p in (
            'lewm/batched_patch_agreement_development.py',
            'scripts/compare_batched_patch_recorded_chains_development.py')},
        native_pose_loaded=False, pose_fit_or_tracker_executed=False, native_execution=False,
        continuous_execution_qualified=False, adopted=False)
    with OUTPUT.open('x') as out: json.dump(report, out, indent=2); out.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__': main()
