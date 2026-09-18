"""Preserve candidate rejection causes during an otherwise identical sensor replay."""
import json
from lewm.gyro_conditioned_pair_pose_development import GyroConditionedPairPose
from scripts import replay_gyro_conditioned_pair_pose_development as replay


class DiagnosticPose(GyroConditionedPairPose):
    def observe(self, *args, **kwargs):
        self.current_pair_failures = []
        try:
            return super().observe(*args, **kwargs)
        except Exception as error:
            raise ValueError('current_pair_failures='+json.dumps(self.current_pair_failures)) from error

    def _candidate(self, ref, current, G):
        try:
            return super()._candidate(ref, current, G)
        except Exception as error:
            chain = []; cause = error
            while cause is not None:
                chain.append(str(cause)); cause = cause.__cause__
            self.current_pair_failures.append(dict(frame=self.frame,
                reference_frame=ref.frame, camera=self.camera, chain=chain))
            raise


if __name__ == '__main__':
    replay.GyroConditionedPairPose = DiagnosticPose
    replay.main()
