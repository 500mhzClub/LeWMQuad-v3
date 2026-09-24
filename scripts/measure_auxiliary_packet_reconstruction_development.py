"""Small recorded-frame timing experiment; no capture or controller execution.

Compare the existing duplicate archive reconstruction, the prepared single-read
composition, and construction from already acquired arrays. This measures only
auxiliary packet construction, including its existing public validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from lewm.auxiliary_downward45_depth_observation_development import from_native_depth
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb, validate_rgb
from scripts import extended_budget_anchored_maze_development as packets
from scripts.novel_maze_auxiliary_rgb_packet_development import public_acquisition

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
ROOT = BASE/'go2_measured_plane_chained_maze02_v1_attempt_001'
CASE = 'no_rgb_direct_measured_plane_chained_maze_02'
FRAMES = (0, 3062, 4003, 4013)


def same(a, b):
    if type(a) is not type(b):
        raise ValueError('packet type changed')
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError('packet array bits changed')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('packet fields changed')
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError('packet length changed')
        for x, y in zip(a, b):
            same(x, y)
    elif a != b:
        raise ValueError('packet value changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve a new result file without overwriting any previous experiment.
    with args.output.open('x') as destination:
        source_result = json.loads((ROOT/'result.json').read_text())
        if source_result['status'] != 'MEASURED_PLANE_CHAINED_MAZE02_V1_COMPLETE':
            raise ValueError('completed recorded development input required')
        directory = ROOT/CASE
        acquisitions = json.loads((directory/'auxiliary_camera_audit.json').read_text())
        reader = packets.ExtendedBudgetRGBDReplay(directory)
        rows = []
        for frame in FRAMES:
            policy, _, _, now = reader.packet(frame)
            acquisition = public_acquisition(acquisitions[frame])
            depth_acquisition = {k: v for k, v in acquisition.items() if k != 'rgb_sha256'}
            with np.load(directory/f'auxiliary_depth_{frame:04d}.npz', allow_pickle=False) as archive:
                native = archive['native_optical_depth_m']
            with Image.open(directory/f'auxiliary_rgb_{frame:04d}.png') as image:
                rgb = np.array(image)

            def duplicate_read():
                depth = packets.depth_packet(directory, frame, policy, depth_acquisition, now_ns=now)
                image, _ = packets.rgb_packet(directory, frame, policy, acquisition, now_ns=now)
                validate_rgb(image, depth, policy, now_ns=now)
                return image, depth

            def single_read():
                image, depth = packets.rgb_packet(directory, frame, policy, acquisition, now_ns=now)
                validate_rgb(image, depth, policy, now_ns=now)
                return image, depth

            def acquired_arrays():
                if (hashlib.sha256(native.tobytes()).hexdigest() != acquisition['native_depth_sha256']
                        or hashlib.sha256(rgb.tobytes()).hexdigest() != acquisition['rgb_sha256']):
                    raise ValueError('recorded acquisition pixels changed')
                depth = from_native_depth(native, policy, measured_ns=acquisition['measured_ns'],
                    available_ns=acquisition['measured_ns'], now_ns=now)
                image = from_captured_rgb(rgb, depth, policy, measured_ns=acquisition['measured_ns'],
                    available_ns=acquisition['measured_ns'], now_ns=now)
                validate_rgb(image, depth, policy, now_ns=now)
                return image, depth

            methods = (duplicate_read, single_read, acquired_arrays)
            reference = duplicate_read()
            for method in methods:
                same(reference, method())  # Warm each method before timing.
            times = {method.__name__: [] for method in methods}
            # Alternate order to reduce systematic warm-cache/order effects.
            for repeat in range(12):
                for offset in range(3):
                    method = methods[(repeat+offset) % 3]
                    started = time.perf_counter()
                    actual = method()
                    times[method.__name__].append((time.perf_counter()-started)*1000)
                    same(reference, actual)  # Equality is outside the timer.
            rows.append(dict(frame=frame, native_depth_sha256=acquisition['native_depth_sha256'],
                rgb_sha256=acquisition['rgb_sha256'], packet_bits_equal=True, timings_ms=times))
        summaries = {}
        for method in rows[0]['timings_ms']:
            values = [v for row in rows for v in row['timings_ms'][method]]
            summaries[method] = dict(samples=len(values), median_ms=float(np.median(values)),
                p95_ms=float(np.quantile(values, .95)), minimum_ms=min(values), maximum_ms=max(values))
        report = dict(frames=FRAMES, rows=rows, summary=summaries,
            original_result_sha256=hashlib.sha256((ROOT/'result.json').read_bytes()).hexdigest(),
            all_sample_packet_bits_equal=True, repeated_warm_cache_component_measurement=True,
            concurrent_job='stop-conditioned maze02 full raw audit',
            includes_capture=False, includes_archive_write=False, includes_primary_packet=False,
            includes_controller=False, native_execution=False, production_path_changed=False,
            sensor_latency_measured=False, real_time_qualified=False)
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summaries, packet_bits_equal=True)))


if __name__ == '__main__':
    main()
