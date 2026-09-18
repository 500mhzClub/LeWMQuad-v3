"""Completed observer verification rejects rewritten raw identities or comparisons."""
from copy import deepcopy

import pytest

from scripts import verify_go2_chained_anchor_observer_completion_v1 as check


def rows():
    evidence = dict(decision_ns=1_500_000_000, status='CURRENT_VISUAL_POSE', current_pose={'frame': 0})
    recorded = dict(tick=0, observation_index=0, decision=dict(original_visual_evidence=evidence,
                                                            requested_command=[0., 0., 0.]))
    observed = dict(tick=0, public_packet_sha256='a'*64, public_inputs_unchanged=True,
        original_requested_command=[0., 0., 0.], command_selected=False,
        original=deepcopy(evidence), candidate=deepcopy(evidence))
    observed['comparison'] = check.run.compare(evidence, observed['original'], observed['candidate'], frame=0)
    return recorded, observed


def test_intact_actual_packet_and_visual_comparison_passes():
    a, b = rows()
    assert not check.check_row(a, b, frame=0, public_sha='a'*64)['stop']


@pytest.mark.parametrize('key,value', [('public_packet_sha256', 'b'*64), ('public_inputs_unchanged', False),
    ('original_requested_command', [.2, 0., 0.]), ('command_selected', True), ('tick', 1)])
def test_raw_identity_and_command_rewrites_rejected(key, value):
    a, b = rows()
    b[key] = value
    with pytest.raises(ValueError, match='actual ordered'): check.check_row(a, b, frame=0, public_sha='a'*64)


def test_comparison_claim_cannot_override_actual_evidence():
    a, b = rows()
    b['comparison']['candidate_original_fields_exact'] = False
    with pytest.raises(ValueError, match='comparison must reconstruct'):
        check.check_row(a, b, frame=0, public_sha='a'*64)


def test_original_saved_visual_rewrite_rejected():
    a, b = rows()
    b['original']['current_pose']['frame'] = 1
    with pytest.raises(ValueError, match='did not reproduce'):
        check.check_row(a, b, frame=0, public_sha='a'*64)
