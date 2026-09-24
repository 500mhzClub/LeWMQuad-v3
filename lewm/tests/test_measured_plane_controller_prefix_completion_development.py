"""Reject fabricated comparisons, truncated streams and confounded outcomes."""
from copy import deepcopy
import pytest
from scripts import verify_go2_measured_plane_controller_prefix_v1 as check
from lewm.tests.test_measured_plane_controller_prefix_comparison_development import example
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def fixture(monkeypatch):
    old,new,_,observer = example()
    old['tick'] = new['tick'] = 0
    new['requested_command'] = [.2,0.,0.]
    new['new_selection']['action'] = 'forward'
    recorded,tape = endpoint(0,old)
    observer.update(tick=0,comparison=dict(raw_packet_sha256='public'))
    row = dict(tick=0,original=old,decision=new,
        comparison=check.job.compare(old,new,recorded['decision'],observer,frame=0),
        public_packet_sha256='public',public_inputs_unchanged=True,
        original_requested_command=tape['requested_command'])
    monkeypatch.setattr(check.run,'fingerprint',lambda _: 'public')
    return row,recorded,observer,[tape]


def test_complete_boundary_verified_without_reading_following_outcome(monkeypatch):
    row,recorded,observer,tape = fixture(monkeypatch)
    def once(value):
        yield value
        raise AssertionError('following outcome was read')
    report = check.check_stream(iter([row]),once(recorded),once(observer),lambda _:None,tape)
    assert report['frames'] == 1
    assert report['boundary_comparison']['requested_command_changed'] is True
    assert report['navigation_recovered'] is False


@pytest.mark.parametrize('fault',['extra','truncated','original','observer','packet','receipt','command','tick'])
def test_incomplete_or_fabricated_evidence_rejected(monkeypatch,fault):
    row,recorded,observer,tape = fixture(monkeypatch)
    rows = [row]
    if fault == 'extra': rows.append(deepcopy(row))
    elif fault == 'truncated': rows = []
    elif fault == 'original': recorded['decision'] = dict(recorded['decision'],failure='altered')
    elif fault == 'observer': observer['candidate'] = {'pose':'unverified'}
    elif fault == 'packet': row['public_packet_sha256'] = 'altered'
    elif fault == 'receipt': row['comparison']['navigation_recovered'] = True
    elif fault == 'command': row['original_requested_command'] = [.2,0.,0.]
    else: row['tick'] = 1
    with pytest.raises(ValueError):
        check.check_stream(iter(rows),iter([recorded]),iter([observer]),lambda _:None,tape)


def test_completion_requires_actual_ended_owner(monkeypatch):
    launch = dict(boot_id=check.run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner={})
    monkeypatch.setattr(check.run,'owner_live',lambda _:True)
    with pytest.raises(ValueError,match='ended'): check.ended(launch)
    monkeypatch.setattr(check.run,'owner_live',lambda _:False)
    check.ended(launch)
    launch['boot_id'] = 'different-boot'
    with pytest.raises(ValueError,match='boot'): check.ended(launch)
