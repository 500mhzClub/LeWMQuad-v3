"""Load an explicitly assigned final fit from a fully admitted matched launch."""
from scripts.run_go2_family_transition_fits_v1 import OUTPUT as FITS, ROSTER
from scripts.cumulative_pulse_snapshot_development import load_snapshot
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json


def load_assigned(launch, name):
    if name not in ROSTER or set(launch['snapshots']) != set(ROSTER):
        raise ValueError('complete six-model prospective assignment required')
    if launch['all_six_admission']['all_six_ledgers_and_raw_scores_reconstructed'] is not True:
        raise ValueError('complete parent fit admission required')
    verify_artifacts(FITS, launch['fit_artifact_sha256'])
    request=read_json(FITS,name+'_request.json')
    snapshot=read_json(FITS,name+'_fit.json')['snapshot']
    if (snapshot != launch['snapshots'][name]
            or name != f"seed_2026091001_{request['variant']}_{request['condition']}"
            or snapshot['configuration']['condition'] != request['condition']
            or snapshot['binding']['input_variant'] != request['variant']):
        raise ValueError('assigned trained condition and input treatment mismatch')
    model=load_snapshot(FITS,snapshot['filename'],sha256=snapshot['sha256'],
        expected_binding=snapshot['binding'],expected_config=snapshot['configuration']).model
    return model, request['condition'], request['variant']
