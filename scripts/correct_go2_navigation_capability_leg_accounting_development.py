"""Read-only leg segmentation erratum; retain source outcomes and first reader."""
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from scripts.run_go2_navigation_capability_development import PROTOCOL, save, sha


def legs(trace, first, outbound_end, return_end, beacon_success, home_success, shortest):
    """Phase boundaries come from logged arrivals, independent of verification.

    A rejected outbound arrival still started a return trajectory. Its boundary
    must not become the episode's final sample, which erased that return path.
    """
    def leg(start, end, success, minimum):
        if end < start:
            raise ValueError('Non-monotone logged leg boundaries')
        length = float(np.linalg.norm(np.diff(trace['base_pose_world'][start:end+1, :2], axis=0), axis=1).sum())
        return dict(success=bool(success), actual_path_m=length, shortest_path_m=minimum,
            spl=float(minimum/max(length, minimum)) if success else 0.,
            elapsed_s=float(trace['timestamp_s'][end]-trace['timestamp_s'][start]))
    end = len(trace['timestamp_s'])-1
    outbound = leg(first, outbound_end if outbound_end is not None else end, beacon_success, shortest[0])
    home = leg(outbound_end if outbound_end is not None else end,
               return_end if return_end is not None else end,
               home_success and outbound_end is not None, shortest[1])
    return outbound, home


def main():
    protocol = json.loads(PROTOCOL.read_text()); base = Path(protocol['output_root']); output.install(base)
    for arm in ['C0', 'C1', 'C2', 'C3', 'C4']:
        root = base/f'runs/v0_pilot_{arm}_dev00_ep0_attempt{2 if arm == "C0" else 1:03d}'
        original = json.loads((root/'episode_evaluation.json').read_text())
        episode = json.loads((root/'episode.json').read_text())
        with np.load(root/'native/physics_trace.npz', allow_pickle=False) as a:
            trace = {k:a[k] for k in ['base_pose_world', 'timestamp_s']}
        metadata = root/'native/in_memory_camera_observations.json'
        if metadata.exists():
            frames = json.loads(metadata.read_text())['frames']
        else:
            frames = json.loads((root/'acquisitions.json').read_text())
            stamps = np.rint(trace['timestamp_s']*1e9).astype(np.int64)
            frames = [r|dict(physical_sample_index=int(np.searchsorted(stamps, r['measured_ns']))) for r in frames]
            assert all(stamps[r['physical_sample_index']] == r['measured_ns'] for r in frames)
        lookup = {r['frame']:r['physical_sample_index'] for r in frames}
        boundaries = {r['phase']:lookup[r['frame']] for r in original['arrivals']}
        outbound, home = legs(trace, frames[0]['physical_sample_index'], boundaries.get('OUTBOUND'),
            boundaries.get('RETURN'), original['beacon_success'], original['home_success'],
            [episode['shortest_outbound_m'], episode['shortest_return_m']])
        corrected = original|dict(schema='navigation_capability_episode_evaluation.v2', outbound=outbound, return_leg=home)
        taxonomy = dict(corrected['failure_and_stall_taxonomy'])
        failed_arrivals = [r for r in original['arrivals'] if not r['passed']]
        if failed_arrivals:
            taxonomy['observed arrival rejected by physical verification'] = len(failed_arrivals)
        corrected['failure_and_stall_taxonomy'] = taxonomy
        corrected['leg_accounting_erratum'] = dict(original_sha256=sha(root/'episode_evaluation.json'),
            implementation_sha256=sha(__file__), boundaries='Logged phase arrivals, including physically rejected arrivals',
            corrected_fields=['outbound', 'return_leg', 'failure_and_stall_taxonomy'],
            outcome_flags_and_safety_unchanged=True, original_preserved=True, new_physics=False,
            prior_leg_values=dict(outbound=original['outbound'], return_leg=original['return_leg']))
        if arm != 'C4':
            assert outbound == original['outbound'] and home == original['return_leg']
        save(root/'episode_evaluation_leg_accounting_v2.json', corrected)
        print(output.dumps(dict(controller=arm, outbound=outbound, return_leg=home, failed_arrivals=failed_arrivals)))


if __name__ == '__main__':
    main()
