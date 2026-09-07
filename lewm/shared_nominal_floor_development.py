"""One raw paired update supplies both floor poses and full nominal evidence."""
from lewm.causal_sensor_state import SensorContractError
from lewm.certified_registration_reuse_development import LeanCertifiedRgbdMomentSensitivity
from lewm.correlated_floor_evidence_development import PairedFloorEvidence


class SharedNominalFloorEvidence(PairedFloorEvidence):
    def __init__(self, source_ids, *, difference_step=1e-3):
        super().__init__(source_ids, difference_step=difference_step)
        self.observer = LeanCertifiedRgbdMomentSensitivity(source_ids, difference_step=difference_step)

    def observe(self, policy, depth, fast, loadings):
        result = super().observe(policy, depth, fast, loadings)
        return result | {'nominal_depth_state': self.nominal_depth_state(now_ns=depth['measured_ns'])}

    def nominal_depth_state(self, *, now_ns):
        if self.failed or self.current is None or now_ns != self.current['measured_ns']:
            raise SensorContractError('fresh active floor/nominal history required')
        return self.observer.nominal_depth_state(now_ns=now_ns)
