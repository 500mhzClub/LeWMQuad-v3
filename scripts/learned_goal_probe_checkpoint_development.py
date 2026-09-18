"""First preregistered historical full-JEPA seed; no outcome-based selection."""
from scripts.read_go2_independent_pulse_parallel_science_v1 import authenticate, read
from scripts.run_go2_independent_pulse_parallel_study_v1 import OUTPUT
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.cumulative_pulse_snapshot_development import load_snapshot

RESULT_SHA = '588f24def6ec8810ae5a3411277576b0d965c77bf6ffdb8e18cfd80dce7b8122'
DEFINITION_SHA = '8d8c3456054a284aa83031ea417d8c433beddbcc04a8b47d3164f120bc0ae5d8'
FIT_NAME = 'seed_2026091101_full_jepa_fit.json'
FIT_SHA = '9d30bd4bcce402d897279fd23001103cc6db31935c22281ce3e3f0b74d23925b'
SNAPSHOT_NAME = 'seed_2026091101_full_jepa.pt'
SNAPSHOT_SHA = '59cec8efec26a2318bdfecb0315f84a17a244f7d03f4029d9f408cdcd3b6abc7'


def authenticate_study():
    terminal, _ = authenticate(RESULT_SHA, DEFINITION_SHA)
    assert terminal['output_sha256'][FIT_NAME] == FIT_SHA
    assert terminal['output_sha256'][SNAPSHOT_NAME] == SNAPSHOT_SHA
    return dict(root=str(OUTPUT), result_sha256=RESULT_SHA, definition_sha256=DEFINITION_SHA,
        artifact_sha256={FIT_NAME: FIT_SHA, SNAPSHOT_NAME: SNAPSHOT_SHA},
        snapshot=read(FIT_NAME)['snapshot'], selection_rule='first preregistered seed; full input; JEPA',
        domain_gap='short-pulse training does not validate four-second moving replans',
        training_performed=False, checkpoint_performance_selection=False)


def load_model():
    verify_artifacts(OUTPUT, {'result.json': RESULT_SHA, FIT_NAME: FIT_SHA, SNAPSHOT_NAME: SNAPSHOT_SHA})
    snapshot = read(FIT_NAME)['snapshot']
    assert snapshot['filename'] == SNAPSHOT_NAME and snapshot['sha256'] == SNAPSHOT_SHA
    trainer = load_snapshot(OUTPUT, SNAPSHOT_NAME, sha256=SNAPSHOT_SHA,
        expected_binding=snapshot['binding'], expected_config=snapshot['configuration'])
    return trainer.model
