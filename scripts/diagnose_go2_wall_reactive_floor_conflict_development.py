"""Reproduce the reactive wall mission's floor rejection from noisy sensors."""
from copy import deepcopy
import sys
from lewm.eligible_floor_registration_development import bind
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates
from lewm.robust_height_floor_tracking_development import RobustHeightFloorRegistration
from scripts import replay_go2_gyro_coherent_floor_development as replay
from scripts.diagnose_go2_depth_noise_failures_development import registration_diagnostic,save
from scripts.run_go2_async_wall_reactive_control_development import ROOT

OUTPUT=replay.path(ROOT)/'gyro_coherent_floor_local_view_revisit_replay_v1'


class DiagnosingRegistration(RobustHeightFloorRegistration):
    def observe(self,policy,primary,auxiliary,raw,*,now_ns):
        try:
            return super().observe(policy,primary,auxiliary,raw,now_ns=now_ns)
        except ValueError as error:
            if str(error)!='current measured candidate conflicts with transported floor reference':
                raise
            save(OUTPUT,'terminal_raw_snapshot.json',raw)
            save(OUTPUT,'terminal_floor_state.json',dict(anchor=self.anchor,
                reference=self.reference,frame=self.frame,failed=self.failed))
            selector=PairedHeightCandidates(primary,auxiliary)
            diagnostic=bind(registration_diagnostic,measured_candidates=selector)(
                self,raw,primary,auxiliary,now_ns)
            diagnostic.update(frame=raw['current_pose']['frame'],
                floor_candidate_selection=deepcopy(selector.receipt),
                same_robust_candidate_selector=True,thresholds_changed=False,
                original_rejection_reproduced=True,native_pose_used=False)
            save(OUTPUT,'registration_diagnostic.json',diagnostic)
            raise


def main():
    sys.argv=['replay','--root-name',ROOT,'--variant','local_view_revisit']
    bind(replay.main,RobustHeightFloorRegistration=DiagnosingRegistration)()


if __name__=='__main__':main()
