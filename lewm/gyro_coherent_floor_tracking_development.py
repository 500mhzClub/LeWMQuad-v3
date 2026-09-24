"""Development raw tracker using gyro-consistent paired floor constraints."""
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.retained_depth_cache_tracking_development import RetainedDepthCachePose, RetainedDepthCacheMotion
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates
from lewm.gyro_coherent_floor_constraint_development import fit_pair


class GyroCoherentFloorPose(RetainedDepthCachePose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coherent_raw_clouds = {}

    def _raw_floor(self, features, receipt):
        frame = receipt['frame']
        if frame in self._coherent_raw_clouds:
            owner, points = self._coherent_raw_clouds[frame]
            if owner is not features:
                raise SensorContractError('paired floor cache acquisition differs')
            return points
        packets = [features[camera].depth for camera in ('primary', 'auxiliary')]
        selector = PairedHeightCandidates(*packets)
        up = receipt['floor_candidate_selection']['seed_up_body']
        clouds = [selector(p['depth_m'], p['valid'], E, up)[0]
            for p, E in zip(packets, selector.mounts)]
        points = np.concatenate(clouds)
        if len(points) != receipt['joint_plane']['candidate_count']:
            raise SensorContractError('original paired raw floor population differs')
        self._coherent_raw_clouds[frame] = (features, points)
        return points

    def _refine_candidate(self, candidate, reference_plane, current_plane, **kwargs):
        result = super()._refine_candidate(candidate, reference_plane, current_plane, **kwargs)
        reference = candidate['reference']
        features, receipt = self._planes[reference.frame]
        frame, current, pending = self._pending_plane
        if features is not reference.features or frame != self.frame:
            raise SensorContractError('exact current and retained floor acquisitions required')
        constraint = fit_pair(self._raw_floor(features, receipt), self._raw_floor(current, pending),
            kwargs['gyro'],
            reference_pool_count=receipt['floor_candidate_selection']['raw_pool_count'],
            current_pool_count=pending['floor_candidate_selection']['raw_pool_count'],
            reference_up=receipt['joint_plane']['up_body'],
            current_up=pending['joint_plane']['up_body'],
            minimum_second_eigenvalue_m2=max(reference_plane['minimum_second_eigenvalue_m2'],
                current_plane['minimum_second_eigenvalue_m2']))
        constraint.update(reference_frame=reference.frame, current_frame=self.frame,
            reference_depth_sha256=receipt['depth_sha256'], current_depth_sha256=pending['depth_sha256'])
        result['registration']['gyro_coherent_floor_constraint'] = constraint
        return result

    def observe(self, *args, **kwargs):
        try:
            result = super().observe(*args, **kwargs)
            self.last_coherent_floor_constraint = (result.get('registration') or {}).get(
                'gyro_coherent_floor_constraint')
            return result
        finally:
            retained = set(self._planes) | {self.frame}
            self._coherent_raw_clouds = {f:v for f,v in self._coherent_raw_clouds.items() if f in retained}


class GyroCoherentFloorMotion(RetainedDepthCacheMotion):
    def __init__(self, *, identity=(0,0,0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = GyroCoherentFloorPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            gyro_coherent_paired_floor_constraint=True,
            local_depth_estimator_and_pose_acceptance_unchanged=False,
            image_consensus_and_motion_thresholds_unchanged=True)
