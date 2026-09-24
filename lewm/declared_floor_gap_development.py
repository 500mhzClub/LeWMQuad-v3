"""Declared floor-pose publication outage; RGB/gyro and raw depth stay live."""
from lewm.floor_reacquisition_development import ReacquiringFloorRegistration,UNAVAILABLE

GAP_FRAMES=(405,406,407,408)


class DeclaredFloorGapRegistration(ReacquiringFloorRegistration):
    def observe(self,policy,primary,auxiliary,raw,*,now_ns):
        anchor,reference=self.anchor,self.reference
        evidence=super().observe(policy,primary,auxiliary,raw,now_ns=now_ns)
        frame=raw['current_pose']['frame']
        if frame not in GAP_FRAMES:return evidence
        if anchor is None:raise ValueError('declared gap requires an admitted earlier floor anchor')
        # Validate the incoming frame, but withhold its floor pose and prevent
        # that withheld observation from becoming a hidden future floor anchor.
        self.anchor,self.reference=anchor,reference
        return dict(status=UNAVAILABLE,frame=frame,measured_ns=now_ns,current_pose=None,
            reason='DECLARED_FOUR_FRAME_FLOOR_POSE_PUBLICATION_GAP',
            anchor_frame=anchor['current_pose']['frame'],fault_injection=True,
            underlying_geometry_also_rejected=evidence.get('status')==UNAVAILABLE,
            rejected_measurement_used_for_mapping=False,rejected_measurement_used_for_arrival=False)


def initialize_registration():
    from lewm.floor_reacquisition_development import initialize_registration as previous
    from lewm import process_registered_round_trip_development as process
    previous()
    process._registration=DeclaredFloorGapRegistration()
