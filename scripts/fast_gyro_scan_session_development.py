"""Capture high-rate measurements live, separately from unchanged model inputs."""
import json

import numpy as np

from lewm.fast_gyro_development import FastGyroBuffer, SCHEMA_ID, SCHEMA, CALIBRATION
from lewm.simulated_fast_gyro_development import IdealFastGyro
from scripts.run_go2_multijunction_route_development_v1 import RouteSession


class FastGyroSession(RouteSession):
    def __init__(self, spec, output):
        self.fast_sensor = IdealFastGyro()
        self.fast_buffer = FastGyroBuffer((0, 0, 0))
        self.fast_rows, self.fast_packets = [], []
        super().__init__(spec, output)

    def _sample(self, requested, applied, timestamp_s):
        before = len(self.samples)
        try:
            return super()._sample(requested, applied, timestamp_s)
        finally:
            # Native-stop rows are retained but never followed by more physics.
            if len(self.samples) > before:
                row = self.samples[-1]
                ns = int(round(float(row['timestamp_s']) * 1e9))
                values, valid = self.fast_sensor.sample(measured_ns=ns, quaternion_xyzw=row['base_pose_world'][3:],
                                                       angular_velocity_world=row['base_twist_world'][3:])
                self.fast_buffer.append(values, valid, measured_ns=ns, available_ns=ns)
                self.fast_rows.append({'measured_ns': np.int64(ns), 'available_ns': np.int64(ns),
                                       'values': values.copy(), 'valid': valid.copy()})

    def capture_current(self):
        before = len(self.model_manifest)
        index = super().capture_current()
        if len(self.model_manifest) > before:
            fast = self.fast_buffer.packet(now_ns=int(self.model_manifest[index]['image_ns']))
            self.fast_packets.append({k: np.asarray(fast[k]).copy() for k in ('values', 'valid', 'measured_ns', 'available_ns')})
        return index

    def persist_observations(self, output):
        super().persist_observations(output)
        for name, rows in (('fast_gyro_samples.npz', self.fast_rows), ('fast_gyro_histories.npz', self.fast_packets)):
            np.savez_compressed(output / name, **({k: np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}))


def load_fast_packet(directory, index):
    manifest = json.loads((directory / 'policy_observations.json').read_text())
    if type(index) is not int or not 0 <= index < len(manifest['frames']):
        raise ValueError('existing actual observation index required')
    with np.load(directory / 'fast_gyro_histories.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'values', 'valid', 'measured_ns', 'available_ns'}:
            raise ValueError('exact fast gyro arrays required')
        row = {k: archive[k][index] for k in archive.files}
    return {'schema': SCHEMA_ID, 'identity': (0, 0, 0), 'decision_ns': manifest['frames'][index]['decision_ns'],
            'calibration_id': CALIBRATION, 'channels': SCHEMA.channels, 'units': SCHEMA.units, **row}
