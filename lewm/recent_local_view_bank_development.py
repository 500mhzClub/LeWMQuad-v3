"""Keep the newest accepted reference in each existing local heading bin."""
from lewm.cadenced_view_revisit_tracking_development import (
    CadencedViewRevisitPose, CadencedViewRevisitMotion)
from lewm.local_view_revisit_tracking_development import view_bin


class RecentLocalViewBankPose(CadencedViewRevisitPose):
    def _remember(self, current, rotation, gyro, position, now):
        super()._remember(current, rotation, gyro, position, now)
        reference = next(r for r in self.references if r.frame == self.frame)
        self.local_view_bank[view_bin(gyro)] = (reference, self._planes[self.frame])


class RecentLocalViewBankMotion(CadencedViewRevisitMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = RecentLocalViewBankPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            local_view_bank_retention='latest_accepted_reference_per_heading_bin',
            maximum_local_view_bank_entries=8,
            matching_and_revisit_eligibility_thresholds_unchanged=True)
