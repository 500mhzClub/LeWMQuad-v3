"""Compare candidate selection on exact noisy packets from retained failures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates
from lewm.gyro_conditioned_partial_floor_candidates_development import GyroConditionedPartialFloorCandidates
from lewm.partial_floor_height_development import fit_gyro_height
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    args = parser.parse_args()
    root = path(args.root_name)
    if (root/'depth_retention.json').exists():
        raise ValueError('retained exact depth required')
    output = root/'gyro_conditioned_partial_floor_probe_v1'
    output.mkdir()
    configure()
    reader = NoisyPublicReplay(root/'native')
    receipts = {r['frame']:r for r in read(root, 'independent_depth_receipts.json')}
    rows = []
    for frame in args.frames:
        _, primary, _, _, auxiliary, now = reader.packet(frame)
        recorded = receipts[frame]
        if recorded['measured_ns'] != now:
            raise ValueError('same recorded observation required')
        up = recorded['joint_plane']['up_body']
        outcomes = {}
        for name, cls in [('original', PairedHeightCandidates),
                          ('gyro_conditioned', GyroConditionedPartialFloorCandidates)]:
            selector = cls(primary, auxiliary)
            clouds = [selector(p['depth_m'], p['valid'], E, up)[0]
                for p, E in zip((primary, auxiliary), selector.mounts)]
            plane = fit_gyro_height(*clouds, up)
            outcomes[name] = dict(plane=plane, selection=selector.receipt)
        # Match actual old observer outcomes before interpreting a changed fit.
        before = outcomes['original']['plane']
        keys = ('available', 'reason', 'candidate_count', 'partial_height_maximum_residual_m')
        matched = all(before.get(k) == recorded['joint_plane'].get(k) for k in keys)
        if not matched:
            raise ValueError(f'original plane probe differs from recorded frame {frame}')
        rows.append(dict(frame=frame, original_recorded_outcome_exact=True, **outcomes))
    result = dict(rows=rows, frames=len(rows),
        newly_available_frames=[r['frame'] for r in rows if not r['original']['plane']['available']
            and r['gyro_conditioned']['plane']['available']],
        lost_available_frames=[r['frame'] for r in rows if r['original']['plane']['available']
            and not r['gyro_conditioned']['plane']['available']],
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__, 'lewm/gyro_conditioned_partial_floor_candidates_development.py')},
        current_packets_only=True, original_noise_recipe=True, native_pose_read=False,
        original_acceptance_thresholds_unchanged=True, floor_identity_certified=False,
        registration_and_navigation_not_tested=True, hardware_validated=False)
    with (output/'result.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')}))


if __name__ == '__main__': main()
